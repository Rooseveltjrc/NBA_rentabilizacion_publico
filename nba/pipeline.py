"""Orquestación de punta a punta del NBA de campañas.

  1. Calidad de datos  ->  2. Pesos y scores (ajuste con resultados)  ->  3. Estrategia
  ->  4. Optimización  ->  5. Diseño experimental  ->  6. Salidas
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import calidad, estrategia, experimentacion, scoring
from .data import ID, cargar_entradas, leer_tabla
from .optimizer import asignar_greedy, asignar_nba


@dataclass
class ConfigNBA:
    umbral_no_contacto: float = 0.0
    excluir_compradores: bool = True       # quien ya compró la campaña deja de ser elegible a ella
    control_adicional_al_minimo: bool = True  # el grupo control se suma al mínimo de negocio
    detener_si_error_calidad: bool = True
    holgura_riesgo: float = 1.2            # elegibles/mínimo por debajo de esto -> campaña en RIESGO
    calidad: calidad.UmbralesCalidad = field(default_factory=calidad.UmbralesCalidad)
    scoring: scoring.ConfigScoring = field(default_factory=scoring.ConfigScoring)
    experimento: experimentacion.ConfigExperimento = field(default_factory=experimentacion.ConfigExperimento)

    @classmethod
    def desde_json(cls, path) -> "ConfigNBA":
        """Lee config/parametros.json; claves ausentes conservan su valor por defecto."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        sub = {"calidad": calidad.UmbralesCalidad, "scoring": scoring.ConfigScoring,
               "experimento": experimentacion.ConfigExperimento}
        kw = {k: (sub[k](**v) if k in sub else v) for k, v in d.items() if not k.startswith("_")}
        return cls(**kw)


def _log(msg, t0=None):
    print(msg + (f"  ({time.time() - t0:.1f}s)" if t0 else ""), flush=True)


def ejecutar(clientes, analitica, estrategia_tbl, resultados=None, pesos=None,
             cfg: ConfigNBA | None = None, salida: str | Path | None = "salida") -> dict:
    cfg = cfg or ConfigNBA()
    out: dict = {}
    clientes, analitica, estrategia_tbl, resultados, pesos = map(
        leer_tabla, (clientes, analitica, estrategia_tbl, resultados, pesos))

    # 1. Calidad --------------------------------------------------------------------
    t = time.time()
    rep, tablas = calidad.validar_entradas(clientes, analitica, estrategia_tbl, resultados, pesos, cfg.calidad)
    out["calidad"] = rep.df
    _log("\n[1] " + rep.resumen(), t)
    if not rep.aprobado and cfg.detener_si_error_calidad:
        _guardar(out, salida)
        raise ValueError("La calidad de datos no supera los mínimos; ver reporte_calidad.csv")
    clientes, analitica, estrategia_tbl = tablas["clientes"], tablas["analitica"], tablas["estrategia"]
    resultados, pesos = tablas["resultados"], tablas["pesos"]
    campanas = estrategia_tbl.campana.astype(str).tolist()

    # 2. Pesos y scores -------------------------------------------------------------
    t = time.time()
    variables = scoring.mapa_variables(analitica, campanas)
    pesos_neg = scoring.pesos_negocio(pesos, variables)
    out["pesos"], out["calibracion"] = scoring.ajustar_pesos(analitica, resultados, pesos_neg, cfg.scoring)
    scores = scoring.calcular_scores(analitica, out["pesos"], out["calibracion"])
    if resultados is not None:
        out["embudo"] = scoring.embudo(resultados)
    _log(f"\n[2] Scores: {out['calibracion'].calibrada.sum()}/{len(campanas)} campañas ajustadas con resultados", t)
    print(out["calibracion"][["campana", "contactados", "compras", "auc_negocio", "auc_ajustado", "motivo"]]
          .to_string(index=False))

    # Restricciones de contacto derivadas de resultados
    no_llamar, compradores = scoring.restricciones_contacto(resultados)
    clientes = clientes.copy()
    if cfg.excluir_compradores:
        for c, ids in compradores.items():
            if c in clientes:
                clientes.loc[clientes[ID].isin(ids), c] = 0

    # 3. Estrategia -----------------------------------------------------------------
    t = time.time()
    e = cargar_entradas(clientes, scores, estrategia_tbl)
    ev = estrategia.evaluar_estrategia(e, cfg.holgura_riesgo)
    out["estrategia"], out["solapamiento"] = ev["por_campana"], ev["solapamiento"]
    out["estrategia_global"] = ev["global"]
    _log("\n[3] Estrategia: " + json.dumps(ev["global"], ensure_ascii=False, default=str), t)
    print(ev["por_campana"][["campana", "elegibles", "elegibles_exclusivos", "minimo",
                             "cobertura_elegibles_vs_minimo", "estado"]].to_string(index=False))
    if not ev["global"]["factible"]:
        _guardar(out, salida)
        raise ValueError(ev["global"]["detalle"])

    minimos_negocio = dict(zip(e.campanas, e.minimos.tolist()))
    if cfg.control_adicional_al_minimo:
        # control requerido por potencia, a partir de la conversión esperada de los mejores elegibles
        ce = cfg.experimento
        extra = np.zeros(len(e.campanas), dtype=np.int64)
        for j, c in enumerate(e.campanas):
            m = int(e.minimos[j])
            if m == 0:
                continue
            p = ev["por_campana"].conv_esperadas_techo_minimo[j] / m
            req = experimentacion.control_minimo(m, p, ce.mde_relativo, ce.alpha, ce.potencia)
            req = ce.min_control if not np.isfinite(req) else max(ce.min_control, req)
            extra[j] = min(req, int(m * ce.max_pct_control / (1 - ce.max_pct_control)))
        e.minimos = np.minimum(e.minimos + extra, e.elegible.sum(0))
        ok, msg = estrategia.verificar_factibilidad(e)
        if not ok:   # si no caben los controles, se vuelve a los mínimos de negocio
            print("[aviso] los grupos control no caben sobre los mínimos; se usan mínimos de negocio")
            e.minimos = np.array([minimos_negocio[c] for c in e.campanas])

    # 4. Optimización ---------------------------------------------------------------
    t = time.time()
    r = asignar_nba(e, umbral_no_contacto=cfg.umbral_no_contacto)
    g = asignar_greedy(e)
    out["resumen_asignacion"] = r.resumen().assign(minimo_negocio=lambda d: d.campana.map(minimos_negocio))
    out["comparativo"] = {"conv_esperadas_nba": round(r.valor_total, 1),
                          "conv_esperadas_greedy": round(g.valor_total, 1),
                          "mejora_pct": round(100 * (r.valor_total / g.valor_total - 1), 2)}
    _log(f"\n[4] Optimización: {out['comparativo']}", t)

    # 5. Experimento ----------------------------------------------------------------
    t = time.time()
    asign = r.tabla_salida()
    asign["no_llamar"] = asign[ID].isin(no_llamar)
    p_trat = asign.groupby("campana_nba").score_nba.mean().to_dict()
    asign, out["diseno_experimental"], out["balance"] = experimentacion.disenar_experimento(
        asign, p_trat, cfg.experimento, minimos=minimos_negocio)
    asign["contactar"] = asign.grupo.notna() & (asign.grupo != "control")
    out["asignacion"] = asign
    _log("\n[5] Diseño experimental", t)
    print(out["diseno_experimental"].to_string(index=False))
    print(f"    Balance: {out['balance'].balanceado.mean():.0%} de chequeos balanceados")

    out["resumen_asignacion"]["contactados_tratamiento"] = out["resumen_asignacion"].campana.map(
        asign[asign.contactar].campana_nba.value_counts()).fillna(0).astype(int)
    out["resumen_asignacion"]["cumple_minimo_negocio"] = \
        out["resumen_asignacion"].contactados_tratamiento >= out["resumen_asignacion"].minimo_negocio
    print("\n" + out["resumen_asignacion"][["campana", "minimo_negocio", "asignados", "contactados_tratamiento",
                                            "cumple_minimo_negocio", "score_medio", "precio_sombra"]]
          .to_string(index=False))
    out["config"] = asdict(cfg)
    _guardar(out, salida)
    return out


def _guardar(out: dict, salida):
    if salida is None:
        return
    d = Path(salida)
    d.mkdir(parents=True, exist_ok=True)
    nombres = {"calidad": "reporte_calidad", "pesos": "pesos_ajustados", "calibracion": "calibracion_scores",
               "embudo": "embudo_resultados", "estrategia": "evaluacion_estrategia",
               "solapamiento": "solapamiento_campanas", "resumen_asignacion": "resumen_asignacion",
               "diseno_experimental": "diseno_experimental", "balance": "balance_experimental"}
    for k, nombre in nombres.items():
        if k in out:
            out[k].to_csv(d / f"{nombre}.csv", index=(k == "solapamiento"))
    if "asignacion" in out:
        out["asignacion"].to_parquet(d / "asignacion_nba.parquet", index=False)
    meta = {k: out[k] for k in ("estrategia_global", "comparativo", "config") if k in out}
    (d / "resumen.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))

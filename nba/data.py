"""Entradas del NBA.

Convención de las tablas:
  clientes:   id_cliente, <campana_1>, <campana_2>, ...        (1/0 = cumple condiciones)
  analitica:  id_cliente, <campana>__<variable>, ...           (scores/variables en [0, 1])
              (una columna llamada exactamente <campana> se toma como variable "score")
  pesos:      campana, variable, peso                          (opcional; por defecto iguales)
  estrategia: campana, cantidad_minima [, cantidad_maxima, valor]
  resultados: id_cliente, campana, recibio_mensaje, abrio_comunicacion, compro,
              gestion_llamada, no_contesto_llamada, contesto_no_llamar, compro_llamada [, grupo]

`cargar_entradas` recibe la analítica ya combinada (una columna de score por campaña);
la combinación ponderada de variables la hace `nba.scoring`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ID = "id_cliente"


@dataclass
class EntradasNBA:
    ids: np.ndarray            # (n,) id de cada cliente
    campanas: list[str]        # (c,) nombre de cada campaña, mismo orden que las columnas
    elegible: np.ndarray       # (n, c) bool
    score: np.ndarray          # (n, c) float32
    minimos: np.ndarray        # (c,) int
    maximos: np.ndarray        # (c,) int  (n si negocio no define tope)
    valor: np.ndarray          # (c,) float, peso de negocio por campaña (1 por defecto)


SEP = "__"
FLAGS_RESULTADO = ["recibio_mensaje", "abrio_comunicacion", "compro", "gestion_llamada",
                   "no_contesto_llamada", "contesto_no_llamar", "compro_llamada"]


def leer_tabla(path: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if path is None or isinstance(path, pd.DataFrame):
        return path
    return _leer(path)


def _leer(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def cargar_entradas(clientes, analitica, estrategia) -> EntradasNBA:
    """Acepta DataFrames o rutas (.csv / .parquet)."""
    cli = clientes if isinstance(clientes, pd.DataFrame) else _leer(clientes)
    ana = analitica if isinstance(analitica, pd.DataFrame) else _leer(analitica)
    est = estrategia if isinstance(estrategia, pd.DataFrame) else _leer(estrategia)

    campanas = est["campana"].astype(str).tolist()
    faltan = [c for c in campanas if c not in cli.columns or c not in ana.columns]
    if faltan:
        raise ValueError(f"Campañas de estrategia sin columna en clientes/analítica: {faltan}")
    if cli[ID].duplicated().any() or ana[ID].duplicated().any():
        raise ValueError("id_cliente duplicado en clientes o analítica")

    # Alinear analítica al orden de clientes; cliente sin score -> 0 (no se pierde elegibilidad)
    ana = ana.set_index(ID).reindex(cli[ID])
    sin_score = int(ana[campanas].isna().all(axis=1).sum())
    if sin_score:
        print(f"[aviso] {sin_score:,} clientes sin registro en analítica; score=0")

    elegible = cli[campanas].fillna(0).to_numpy(dtype=np.int8) == 1
    score = ana[campanas].fillna(0).to_numpy(dtype=np.float32)
    if (score < 0).any() or (score > 1).any():
        raise ValueError("Hay scores fuera de [0, 1]")

    n = len(cli)
    minimos = est["cantidad_minima"].fillna(0).to_numpy(dtype=np.int64)
    maximos = (est["cantidad_maxima"].fillna(n).to_numpy(dtype=np.int64)
               if "cantidad_maxima" in est else np.full(len(campanas), n, dtype=np.int64))
    valor = (est["valor"].fillna(1.0).to_numpy(dtype=np.float32)
             if "valor" in est else np.ones(len(campanas), dtype=np.float32))
    if (minimos > maximos).any():
        raise ValueError("cantidad_minima > cantidad_maxima en alguna campaña")

    return EntradasNBA(cli[ID].to_numpy(), campanas, elegible, score, minimos, maximos, valor)


def generar_datos_sinteticos(n_clientes=100_000, n_campanas=10, seed=0):
    """Genera las tres tablas con estructura realista para pruebas."""
    rng = np.random.default_rng(seed)
    campanas = [f"camp_{i:02d}" for i in range(n_campanas)]
    tasa_eleg = rng.uniform(0.05, 0.6, n_campanas)          # campañas más y menos restrictivas
    elegible = rng.random((n_clientes, n_campanas)) < tasa_eleg
    # Score con componente de cliente (propensión general) + componente por campaña
    base = rng.beta(2, 5, (n_clientes, 1))
    score = np.clip(0.6 * base + 0.4 * rng.beta(2, 3, (n_clientes, n_campanas)), 0, 1)

    ids = np.arange(1, n_clientes + 1)
    clientes = pd.DataFrame(elegible.astype(np.int8), columns=campanas).assign(**{ID: ids})
    analitica = pd.DataFrame(score.astype(np.float32), columns=campanas).assign(**{ID: ids})
    # Mínimos: entre 5% y 40% de los elegibles de cada campaña
    minimos = elegible.sum(0) * rng.uniform(0.05, 0.4, n_campanas)
    minimos = (minimos * min(1.0, 0.6 * n_clientes / minimos.sum())).astype(int)  # sum <= 60% base
    estrategia = pd.DataFrame({"campana": campanas, "cantidad_minima": minimos})
    return clientes[[ID] + campanas], analitica[[ID] + campanas], estrategia


def _sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generar_escenario_completo(n_clientes=200_000, n_campanas=8, frac_historico=0.15, seed=0):
    """Escenario sintético con las 5 tablas. Cada campaña tiene 3 variables analíticas con
    distinto poder predictivo real (desconocido para negocio), para que el ajuste de pesos
    con resultados tenga algo que descubrir."""
    rng = np.random.default_rng(seed)
    campanas = [f"camp_{i:02d}" for i in range(n_campanas)]
    variables = ["modelo_propension", "var_recencia", "var_afinidad"]
    ids = np.arange(1, n_clientes + 1)

    elegible = rng.random((n_clientes, n_campanas)) < rng.uniform(0.05, 0.6, n_campanas)
    clientes = pd.DataFrame(elegible.astype(np.int8), columns=campanas)
    clientes.insert(0, ID, ids)

    analitica = {ID: ids}
    p_real = np.zeros((n_clientes, n_campanas))
    base = rng.beta(2, 5, n_clientes)
    for j, c in enumerate(campanas):
        X = np.column_stack([np.clip(0.5 * base + 0.5 * rng.beta(2, 4, n_clientes), 0, 1),
                             rng.beta(2, 2, n_clientes), rng.beta(1.5, 3, n_clientes)])
        beta_real = rng.dirichlet([2, 1, 1]) * rng.uniform(4, 7)     # pesos reales
        p_real[:, j] = _sigmoide(-4.0 + rng.normal(0, .3) + X @ beta_real)
        for k, v in enumerate(variables):
            analitica[f"{c}{SEP}{v}"] = X[:, k].astype(np.float32)
    analitica = pd.DataFrame(analitica)

    pesos = pd.DataFrame([(c, v, w) for c in campanas for v, w in zip(variables, [0.5, 0.3, 0.2])],
                         columns=["campana", "variable", "peso"])                  # criterio negocio

    minimos = elegible.sum(0) * rng.uniform(0.05, 0.4, n_campanas)
    minimos = (minimos * min(1.0, 0.6 * n_clientes / minimos.sum())).astype(int)
    estrategia = pd.DataFrame({"campana": campanas, "cantidad_minima": minimos})

    # Histórico: una ola previa contactó a una muestra aleatoria de elegibles
    filas = []
    for j, c in enumerate(campanas):
        idx = np.flatnonzero(elegible[:, j])
        idx = rng.choice(idx, int(len(idx) * frac_historico), replace=False)
        m = len(idx)
        recibio = rng.random(m) < 0.92
        abrio = recibio & (rng.random(m) < 0.35)
        compro = abrio & (rng.random(m) < np.clip(p_real[idx, j] * 2.5, 0, 1))
        llamada = ~compro & (rng.random(m) < 0.4)
        contesto = llamada & (rng.random(m) < 0.45)
        no_llamar = contesto & (rng.random(m) < 0.12)
        compro_ll = contesto & ~no_llamar & (rng.random(m) < np.clip(p_real[idx, j] * 3, 0, 1))
        filas.append(pd.DataFrame({
            ID: ids[idx], "campana": c, "recibio_mensaje": recibio, "abrio_comunicacion": abrio,
            "compro": compro, "gestion_llamada": llamada, "no_contesto_llamada": llamada & ~contesto,
            "contesto_no_llamar": no_llamar, "compro_llamada": compro_ll}))
    resultados = pd.concat(filas, ignore_index=True)
    resultados[FLAGS_RESULTADO] = resultados[FLAGS_RESULTADO].astype(np.int8)
    return {"clientes": clientes, "analitica": analitica, "pesos": pesos,
            "estrategia": estrategia, "resultados": resultados}

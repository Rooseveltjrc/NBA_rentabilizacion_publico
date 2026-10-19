"""Diseños muestrales para experimentación de campañas.

1. Tamaño muestral por potencia (test de dos proporciones, asignación desbalanceada).
2. Grupo control por campaña (holdout) con aleatorización estratificada
   (campaña x decil de score x estratos adicionales), reproducible por semilla.
3. Muestreo estratificado para pilotos / estimación (margen de error, corrección por
   población finita, asignación proporcional o de Neyman).
4. Chequeo de balance (SMD y chi-cuadrado) y medición del lift incremental con IC.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .data import ID


@dataclass
class ConfigExperimento:
    brazos: dict = field(default_factory=lambda: {"tratamiento": 0.9, "control": 0.1})
    modo: str = "potencia"           # "potencia": control mínimo para detectar mde; "fijo": % de brazos
    mde_relativo: float = 0.15       # lift relativo mínimo a detectar: p_trat = p_ctrl * (1 + mde)
    alpha: float = 0.05
    potencia: float = 0.80
    min_control: int = 500
    max_pct_control: float = 0.30    # tope para no sacrificar demasiada venta
    n_deciles: int = 10
    estratos_extra: list = field(default_factory=list)   # columnas adicionales para estratificar
    seed: int = 42


# ---------------------------------------------------------------- tamaños muestrales
def _z(alpha, potencia):
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(potencia)


def n_por_grupo(p_ctrl, p_trat, alpha=0.05, potencia=0.8, ratio=1.0):
    """n del control (y n_trat = ratio * n_ctrl) para detectar p_trat - p_ctrl."""
    d = abs(p_trat - p_ctrl)
    if d == 0:
        return np.inf
    return int(np.ceil(_z(alpha, potencia) ** 2 * (p_ctrl * (1 - p_ctrl) + p_trat * (1 - p_trat) / ratio) / d ** 2))


def control_minimo(n_trat, p_trat, mde_relativo, alpha=0.05, potencia=0.8):
    """Control mínimo dado un tratamiento fijo (el mínimo de negocio). inf si no alcanza."""
    p_ctrl = p_trat / (1 + mde_relativo)
    d2 = (p_trat - p_ctrl) ** 2 / _z(alpha, potencia) ** 2
    resto = d2 - p_trat * (1 - p_trat) / max(n_trat, 1)
    return int(np.ceil(p_ctrl * (1 - p_ctrl) / resto)) if resto > 0 else np.inf


def mde_alcanzable(n_trat, n_ctrl, p_trat, alpha=0.05, potencia=0.8):
    """Lift relativo mínimo detectable con los tamaños dados (búsqueda numérica)."""
    if n_trat <= 0 or n_ctrl <= 0 or p_trat <= 0:
        return np.nan
    z = _z(alpha, potencia)

    def holgura(mde):
        p_c = p_trat / (1 + mde)
        return (p_trat - p_c) - z * np.sqrt(p_trat * (1 - p_trat) / n_trat + p_c * (1 - p_c) / n_ctrl)

    if holgura(1e3) < 0:
        return np.inf
    return round(float(optimize.brentq(holgura, 1e-6, 1e3)), 4)


def tamano_muestra_proporcion(N, p=0.5, error=0.02, confianza=0.95):
    """n para estimar una proporción con margen de error dado, con corrección finita."""
    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    n0 = z ** 2 * p * (1 - p) / error ** 2
    return int(np.ceil(n0 / (1 + (n0 - 1) / N)))


# ---------------------------------------------------------------- estratificación
def agregar_estratos(df, score_col, cfg: ConfigExperimento, por="campana_nba"):
    """Decil de score dentro de cada campaña + estratos adicionales -> columna 'estrato'."""
    df = df.copy()
    df["decil_score"] = (df.groupby(por)[score_col]
                         .transform(lambda s: pd.qcut(s.rank(method="first"), min(cfg.n_deciles, len(s)),
                                                      labels=False)) + 1).astype("Int64")
    partes = [df[por].astype(str), "d" + df.decil_score.astype(str)] + \
             [df[c].astype(str) for c in cfg.estratos_extra]
    df["estrato"] = partes[0].str.cat(partes[1:], sep="|")
    return df


def _cuotas(tam: pd.Series, total: int) -> pd.Series:
    """Reparte 'total' entre estratos proporcional a su tamaño (mayor resto, total exacto)."""
    q = tam * total / tam.sum()
    base = np.floor(q).astype(int)
    resto = int(total - base.sum())
    base[(q - base).sort_values(ascending=False).index[:resto]] += 1
    return np.minimum(base, tam)


def aleatorizar(df, proporciones: dict, estrato_col="estrato", seed=42, col_salida="grupo",
                n_control: int | None = None):
    """Aleatorización estratificada: cada estrato reparte sus clientes según las proporciones.
    Con n_control, el control tiene exactamente ese tamaño total (cuotas por mayor resto)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_u"] = rng.random(len(df))
    df = df.sort_values([estrato_col, "_u"])
    pos = df.groupby(estrato_col).cumcount().to_numpy()
    tam = df.groupby(estrato_col)[estrato_col].transform("size").to_numpy()
    if n_control is not None:
        cuota = _cuotas(df.groupby(estrato_col).size(), n_control)
        es_ctrl = pos < df[estrato_col].map(cuota).to_numpy()
        trat = {k: v for k, v in proporciones.items() if k != "control"}
        resto = df[~es_ctrl].copy()
        resto = aleatorizar(resto.drop(columns="_u"), trat, estrato_col, seed + 1, col_salida) if trat \
            else resto.assign(**{col_salida: "tratamiento"})
        df[col_salida] = "control"
        df.loc[resto.index, col_salida] = resto[col_salida]
        return df.drop(columns="_u").sort_index()
    nombres = list(proporciones)
    acum = np.cumsum([proporciones[k] for k in nombres])
    acum = acum / acum[-1]
    frac = (pos + 0.5) / tam                       # posición relativa dentro del estrato
    df[col_salida] = np.array(nombres, dtype=object)[np.searchsorted(acum, frac, side="right").clip(max=len(nombres) - 1)]
    return df.drop(columns="_u").sort_index()


def proporciones_por_campana(asignados: pd.DataFrame, p_trat: dict, cfg: ConfigExperimento,
                             minimos: dict | None = None) -> pd.DataFrame:
    """Decide el % de control por campaña y reporta potencia/MDE del diseño."""
    filas = []
    for c, n in asignados.campana_nba.value_counts().items():
        p = float(p_trat.get(c, np.nan))
        tope = int(n * cfg.max_pct_control)
        if minimos is not None:                      # el tratamiento debe cubrir el mínimo de negocio
            tope = max(0, min(tope, n - int(minimos.get(c, 0))))
        if cfg.modo == "potencia":
            n_ctrl = 0
            for _ in range(5):                       # punto fijo: n_trat = n - n_ctrl
                req = control_minimo(n - n_ctrl, p, cfg.mde_relativo, cfg.alpha, cfg.potencia)
                if not np.isfinite(req):
                    n_ctrl = tope
                    break
                n_ctrl = req
            n_ctrl = int(min(max(cfg.min_control, n_ctrl), tope))
            pct = n_ctrl / n
        else:
            pct = cfg.brazos.get("control", 0.0) / sum(cfg.brazos.values())
            n_ctrl = min(int(round(n * pct)), tope)
            pct = n_ctrl / n
        n_trat = n - n_ctrl
        mde = mde_alcanzable(n_trat, n_ctrl, p, cfg.alpha, cfg.potencia)
        filas.append(dict(campana=c, asignados=n, p_esperada_tratamiento=round(p, 4),
                          pct_control=round(pct, 4), n_control=n_ctrl, n_tratamiento=n_trat,
                          mde_relativo_alcanzable=mde, cumple_mde_objetivo=bool(mde <= cfg.mde_relativo),
                          tratamiento_cubre_minimo=bool(minimos is None or n_trat >= minimos.get(c, 0))))
    return pd.DataFrame(filas)


def disenar_experimento(asignacion: pd.DataFrame, p_trat: dict, cfg: ConfigExperimento | None = None,
                        minimos: dict | None = None):
    """asignacion: id_cliente, campana_nba, score_nba [, estratos_extra].
    Devuelve (asignacion con 'estrato' y 'grupo', diseño por campaña, balance)."""
    cfg = cfg or ConfigExperimento()
    df = asignacion[asignacion.campana_nba.notna()].copy()
    diseno = proporciones_por_campana(df, p_trat, cfg, minimos)
    df = agregar_estratos(df, "score_nba", cfg)

    brazos_trat = {k: v for k, v in cfg.brazos.items() if k != "control"}
    partes = []
    for c, g in df.groupby("campana_nba"):
        fila = diseno.loc[diseno.campana == c].iloc[0]
        pc, tot = float(fila.pct_control), sum(brazos_trat.values())
        prop = {k: (1 - pc) * v / tot for k, v in brazos_trat.items()} | {"control": pc}
        partes.append(aleatorizar(g, prop, seed=cfg.seed + zlib.crc32(str(c).encode()) % 10_000,
                                  n_control=int(fila.n_control)))
    df = pd.concat(partes)
    out = asignacion.merge(df[[ID, "decil_score", "estrato", "grupo"]], on=ID, how="left")
    return out, diseno, balance(df, ["score_nba"])


# ---------------------------------------------------------------- muestreo para pilotos
def muestra_estratificada(df, estratos: list[str], n=None, error=0.02, confianza=0.95,
                          asignacion="proporcional", var_col=None, seed=42):
    """Selecciona una muestra estratificada. n=None -> se calcula por margen de error.
    asignacion: 'proporcional' o 'neyman' (requiere var_col, p.ej. score, para la varianza)."""
    N = len(df)
    n = n or tamano_muestra_proporcion(N, error=error, confianza=confianza)
    g = df.groupby(estratos)
    tam = g.size()
    if asignacion == "neyman":
        sd = g[var_col].std().fillna(0) + 1e-9
        peso = tam * sd
    else:
        peso = tam.astype(float)
    cuota = peso / peso.sum() * n
    base = np.floor(cuota).astype(int)
    faltan = n - base.sum()                       # mayor resto
    base[(cuota - base).sort_values(ascending=False).index[:faltan]] += 1
    base = np.minimum(base, tam)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(ix, base[k], replace=False) for k, ix in g.indices.items()
                          if base[k] > 0])
    plan = pd.DataFrame({"poblacion": tam, "muestra": base, "fraccion": (base / tam).round(4)})
    return df.iloc[np.sort(idx)], plan.reset_index()


# ---------------------------------------------------------------- balance y medición
def balance(df, covariables, grupo_col="grupo", por="campana_nba", ref="control"):
    """SMD de covariables y chi2 de la distribución por estrato entre cada brazo y el control."""
    filas = []
    for c, g in df.groupby(por):
        ctrl = g[g[grupo_col] == ref]
        for brazo, t in g[g[grupo_col] != ref].groupby(grupo_col):
            for v in covariables:
                sd = np.sqrt((t[v].var() + ctrl[v].var()) / 2) or 1e-9
                smd = (t[v].mean() - ctrl[v].mean()) / sd
                filas.append(dict(campana=c, brazo=brazo, covariable=v, smd=round(smd, 4),
                                  balanceado=abs(smd) < 0.1))
            tabla = pd.crosstab(g.estrato, g[grupo_col])[[brazo, ref]]
            pval = stats.chi2_contingency(tabla)[1] if tabla.shape[0] > 1 else 1.0
            filas.append(dict(campana=c, brazo=brazo, covariable="estrato(chi2 p-valor)",
                              smd=round(pval, 4), balanceado=pval > 0.05))
    return pd.DataFrame(filas)


def medir_lift(resultados: pd.DataFrame, alpha=0.05, ref="control"):
    """resultados con columnas campana, grupo y compra (o compro/compro_llamada).
    Lift absoluto y relativo de cada brazo vs control, IC y p-valor (z de dos proporciones)."""
    r = resultados.copy()
    if "compra" not in r:
        r["compra"] = ((r.compro == 1) | (r.compro_llamada == 1)).astype(int)
    z = stats.norm.ppf(1 - alpha / 2)
    filas = []
    for c, g in r.groupby("campana"):
        ctrl = g[g.grupo == ref].compra
        if ctrl.empty:                      # sin control no hay contra qué comparar
            continue
        for brazo, t in g[g.grupo != ref].groupby("grupo"):
            t = t.compra
            p1, p0, n1, n0 = t.mean(), ctrl.mean(), len(t), len(ctrl)
            se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
            pp = (t.sum() + ctrl.sum()) / (n1 + n0)
            se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0)) or 1e-12
            filas.append(dict(campana=c, brazo=brazo, n_trat=n1, n_ctrl=n0, conv_trat=round(p1, 4),
                              conv_ctrl=round(p0, 4), lift_abs=round(p1 - p0, 4),
                              lift_rel=round((p1 - p0) / p0, 4) if p0 > 0 else np.nan,
                              ic_inf=round(p1 - p0 - z * se, 4), ic_sup=round(p1 - p0 + z * se, 4),
                              p_valor=round(2 * (1 - stats.norm.cdf(abs(p1 - p0) / se0)), 4),
                              ventas_incrementales=round((p1 - p0) * n1, 1)))
    return pd.DataFrame(filas)

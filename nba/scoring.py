"""Score compuesto por campaña y ajuste de pesos con resultados históricos.

score_c(i) = sum_k w_ck * x_ck(i) / sum_k w_ck            (pesos de negocio)

Con resultados (insumo 4) por campaña:
  1. Población: clientes contactados (recibio_mensaje=1 o gestion_llamada=1).
     Objetivo: compra = compro OR compro_llamada.
  2. Regresión logística compra ~ variables -> importancia observada (coeficientes >= 0).
  3. Credibilidad z = compras / (compras + k): pocos datos -> se respeta el peso de negocio;
     muchos datos -> mandan los resultados.   w_final = z * w_datos + (1 - z) * w_negocio
  4. Calibración p = sigmoide(a + b * score) para que el score sea probabilidad de compra y
     sea comparable entre campañas (clave para que el optimizador reparta bien).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import ID, SEP


@dataclass
class ConfigScoring:
    k_credibilidad: float = 200.0    # compras necesarias para dar 50% de peso a los datos
    min_compras: int = 30            # por debajo no se ajusta ni calibra la campaña
    min_contactados: int = 300
    c_regularizacion: float = 1.0
    seed: int = 0


def mapa_variables(analitica: pd.DataFrame, campanas: list[str]) -> dict[str, list[str]]:
    """campana -> columnas de analítica que le pertenecen."""
    out = {}
    for c in campanas:
        cols = [x for x in analitica.columns if x.startswith(c + SEP)]
        out[c] = cols or ([c] if c in analitica.columns else [])
    return out


def _nombre_var(col, campana):
    return col[len(campana) + len(SEP):] if col.startswith(campana + SEP) else "score"


def pesos_negocio(pesos: pd.DataFrame | None, variables: dict[str, list[str]]) -> pd.DataFrame:
    """Pesos de negocio normalizados; variable sin peso definido -> peso igualitario."""
    filas = []
    for c, cols in variables.items():
        for col in cols:
            v = _nombre_var(col, c)
            w = np.nan
            if pesos is not None:
                m = pesos[(pesos.campana == c) & (pesos.variable == v)]
                w = float(m.peso.iloc[0]) if len(m) else np.nan
            filas.append((c, v, col, w))
    df = pd.DataFrame(filas, columns=["campana", "variable", "columna", "peso_negocio"])
    df["peso_negocio"] = df.peso_negocio.fillna(df.groupby("campana").peso_negocio.transform(
        lambda s: s.mean() if s.notna().any() else 1.0))
    df["peso_negocio"] /= df.groupby("campana").peso_negocio.transform("sum")
    return df


def _contactados(resultados: pd.DataFrame) -> pd.DataFrame:
    r = resultados.copy()
    r["contactado"] = (r.recibio_mensaje == 1) | (r.gestion_llamada == 1)
    r["compra"] = ((r.compro == 1) | (r.compro_llamada == 1)).astype(int)
    return r[r.contactado]


def ajustar_pesos(analitica, resultados, pesos_neg: pd.DataFrame, cfg: ConfigScoring | None = None):
    """Devuelve (pesos_df, calibracion_df). Sin resultados, pesos finales = pesos de negocio."""
    cfg = cfg or ConfigScoring()
    pesos = pesos_neg.copy()
    pesos["peso_datos"] = np.nan
    pesos["credibilidad"] = 0.0
    pesos["peso_final"] = pesos.peso_negocio
    calib = []
    r = _contactados(resultados) if resultados is not None else None
    ana = analitica.set_index(ID)

    for c, g in pesos.groupby("campana", sort=False):
        cols = g.columna.tolist()
        fila = dict(campana=c, contactados=0, compras=0, tasa_compra=np.nan, a=0.0, b=1.0,
                    calibrada=False, auc_negocio=np.nan, auc_ajustado=np.nan, motivo="")
        rc = r[r.campana == c] if r is not None else pd.DataFrame()
        rc = rc[rc[ID].isin(ana.index)] if len(rc) else rc
        fila["contactados"], fila["compras"] = len(rc), int(rc.compra.sum()) if len(rc) else 0
        if len(rc):
            fila["tasa_compra"] = round(rc.compra.mean(), 4)
        if fila["compras"] < cfg.min_compras or fila["contactados"] < cfg.min_contactados \
                or fila["compras"] == fila["contactados"]:
            fila["motivo"] = "sin resultados suficientes: se usan pesos de negocio sin calibrar"
            calib.append(fila)
            continue

        X = ana.loc[rc[ID], cols].to_numpy(np.float64)
        X = np.nan_to_num(X, nan=np.nanmean(X, 0))
        y = rc.compra.to_numpy()
        w_neg = g.peso_negocio.to_numpy()

        # Importancia observada: logística en train, AUC comparado en test
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=cfg.seed, stratify=y)
        lr = LogisticRegression(C=cfg.c_regularizacion, max_iter=1000).fit(Xtr, ytr)
        coef = np.clip(lr.coef_[0], 0, None)
        if coef.sum() <= 0:
            fila["motivo"] = "ninguna variable con relación positiva: se mantienen pesos de negocio"
            calib.append(fila)
            continue
        w_dat = coef / coef.sum()
        z = fila["compras"] / (fila["compras"] + cfg.k_credibilidad)
        w_fin = z * w_dat + (1 - z) * w_neg
        w_fin /= w_fin.sum()
        if len(np.unique(yte)) > 1:
            fila["auc_negocio"] = round(roc_auc_score(yte, Xte @ w_neg), 4)
            fila["auc_ajustado"] = round(roc_auc_score(yte, Xte @ w_fin), 4)

        # Calibración del score final a probabilidad (con todos los contactados)
        cal = LogisticRegression(C=1e6, max_iter=1000).fit((X @ w_fin)[:, None], y)
        fila.update(a=float(cal.intercept_[0]), b=float(cal.coef_[0, 0]), calibrada=True,
                    motivo=f"ajustada con credibilidad {z:.2f}")
        calib.append(fila)
        pesos.loc[g.index, "peso_datos"] = w_dat
        pesos.loc[g.index, "credibilidad"] = z
        pesos.loc[g.index, "peso_final"] = w_fin

    return pesos.round({"peso_negocio": 4, "peso_datos": 4, "credibilidad": 3, "peso_final": 4}), \
        pd.DataFrame(calib)


def calcular_scores(analitica: pd.DataFrame, pesos: pd.DataFrame, calibracion: pd.DataFrame) -> pd.DataFrame:
    """Una columna por campaña con el score final (probabilidad si la campaña está calibrada)."""
    out = {ID: analitica[ID].to_numpy()}
    cal = calibracion.set_index("campana")
    for c, g in pesos.groupby("campana", sort=False):
        X = analitica[g.columna].to_numpy(np.float32)
        s = np.nan_to_num(X, nan=0.0) @ g.peso_final.to_numpy(np.float32)
        if c in cal.index and cal.loc[c, "calibrada"]:
            s = 1 / (1 + np.exp(-(cal.loc[c, "a"] + cal.loc[c, "b"] * s)))
        out[c] = np.clip(s, 0, 1).astype(np.float32)
    return pd.DataFrame(out)


def embudo(resultados: pd.DataFrame) -> pd.DataFrame:
    """Métricas del embudo por campaña (insumo 4)."""
    r = resultados.copy()
    r["compra_total"] = (r.compro == 1) | (r.compro_llamada == 1)
    r["contesto"] = (r.gestion_llamada == 1) & (r.no_contesto_llamada == 0)
    g = r.groupby("campana")
    n = g.size()
    llam = g.gestion_llamada.sum()
    out = pd.DataFrame({
        "gestionados": n,
        "tasa_entrega": g.recibio_mensaje.mean(),
        "tasa_apertura": g.abrio_comunicacion.sum() / g.recibio_mensaje.sum().clip(lower=1),
        "conv_digital": g.compro.sum() / g.abrio_comunicacion.sum().clip(lower=1),
        "pct_gestion_llamada": llam / n,
        "contactabilidad_llamada": g.contesto.sum() / llam.clip(lower=1),
        "pct_pidio_no_llamar": g.contesto_no_llamar.sum() / g.contesto.sum().clip(lower=1),
        "conv_llamada": g.compro_llamada.sum() / g.contesto.sum().clip(lower=1),
        "conv_total": g.compra_total.mean(),
        "compras": g.compra_total.sum(),
    })
    return out.round(4).reset_index()


def restricciones_contacto(resultados: pd.DataFrame | None):
    """ids que pidieron no ser llamados (cualquier campaña) y compradores por campaña."""
    if resultados is None:
        return set(), {}
    no_llamar = set(resultados.loc[resultados.contesto_no_llamar == 1, ID])
    comp = resultados[(resultados.compro == 1) | (resultados.compro_llamada == 1)]
    compradores = comp.groupby("campana")[ID].apply(set).to_dict()
    return no_llamar, compradores

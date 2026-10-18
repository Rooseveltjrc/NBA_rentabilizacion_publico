"""Validación de la estrategia de negocio y disponibilidad de clientes por campaña."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EntradasNBA
from .optimizer import verificar_factibilidad


def evaluar_estrategia(e: EntradasNBA, holgura_riesgo: float = 1.2) -> dict:
    """Disponibilidad por campaña, solapamiento entre campañas y factibilidad global.

    estado por campaña:
      INFACTIBLE  elegibles < mínimo
      RIESGO      elegibles < holgura_riesgo * mínimo, o el mínimo exige más clientes que
                  los que la campaña puede tener sin quitárselos a mínimos de otras
      OK          resto
    """
    E, S = e.elegible, e.score
    n_eleg = E.sum(0)
    exclusivos = (E & (E.sum(1, keepdims=True) == 1)).sum(0)
    score_medio = np.divide((S * E).sum(0), n_eleg, out=np.zeros(len(n_eleg)), where=n_eleg > 0)

    # Conversiones esperadas si cada campaña tomara a sus mejores 'mínimo' clientes (cota superior)
    techo = np.zeros(len(e.campanas))
    for j in range(len(e.campanas)):
        s = S[E[:, j], j]
        m = min(int(e.minimos[j]), len(s))
        techo[j] = np.partition(s, len(s) - m)[len(s) - m:].sum() if m else 0.0

    cobertura = np.divide(n_eleg, e.minimos, out=np.full(len(n_eleg), np.inf), where=e.minimos > 0)
    estado = np.where(n_eleg < e.minimos, "INFACTIBLE",
                      np.where(cobertura < holgura_riesgo, "RIESGO", "OK"))

    por_campana = pd.DataFrame({
        "campana": e.campanas,
        "elegibles": n_eleg,
        "elegibles_exclusivos": exclusivos,
        "minimo": e.minimos,
        "maximo": e.maximos,
        "cobertura_elegibles_vs_minimo": np.round(cobertura, 2),
        "pct_base_requerida": np.round(e.minimos / len(e.ids), 4),
        "score_medio_elegibles": np.round(score_medio, 4),
        "conv_esperadas_techo_minimo": np.round(techo, 1),
        "estado": estado,
    })

    # Solapamiento: % de elegibles de la fila que también son elegibles de la columna
    inter = E.T.astype(np.int32) @ E.astype(np.int32)
    solape = pd.DataFrame(np.round(inter / np.maximum(n_eleg[:, None], 1), 3),
                          index=e.campanas, columns=e.campanas)

    factible, msg = verificar_factibilidad(e)
    con_alguna = int(E.any(1).sum())
    global_ = {
        "clientes_total": len(e.ids),
        "clientes_con_alguna_campana": con_alguna,
        "suma_minimos": int(e.minimos.sum()),
        "pct_base_comprometida_por_minimos": round(e.minimos.sum() / max(con_alguna, 1), 4),
        "factible": factible,
        "detalle": msg,
    }
    # Riesgo por competencia: aunque cada campaña tenga elegibles, los mínimos conjuntos pueden
    # no caber. Se marca en RIESGO a las campañas cuyo pool depende de clientes compartidos.
    if not factible:
        criticas = por_campana.elegibles_exclusivos < por_campana.minimo
        por_campana.loc[criticas & (por_campana.estado == "OK"), "estado"] = "RIESGO"

    return {"por_campana": por_campana, "solapamiento": solape, "global": global_}

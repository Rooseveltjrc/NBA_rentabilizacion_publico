import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from nba import asignar_greedy, asignar_nba, cargar_entradas, generar_datos_sinteticos
from nba.optimizer import verificar_factibilidad


def optimo_lp(e):
    """Óptimo exacto con HiGHS (sólo viable para instancias pequeñas)."""
    i, j = np.nonzero(e.elegible)
    k = len(i)
    w = e.score[i, j] * e.valor[j]
    r_cli = coo_matrix((np.ones(k), (i, np.arange(k))), shape=(len(e.ids), k))
    r_cmp = coo_matrix((np.ones(k), (j, np.arange(k))), shape=(len(e.campanas), k))
    A = vstack([r_cli, r_cmp, -r_cmp])
    b = np.concatenate([np.ones(len(e.ids)), e.maximos, -e.minimos])
    res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    assert res.status == 0
    return -res.fun


@pytest.mark.parametrize("seed", range(5))
def test_igual_al_optimo_lp(seed):
    e = cargar_entradas(*generar_datos_sinteticos(3_000, 8, seed=seed))
    r = asignar_nba(e)
    assert r.convergio
    a = r.asignacion
    assert (np.bincount(a[a >= 0], minlength=8) >= e.minimos).all()
    assert e.elegible[np.flatnonzero(a >= 0), a[a >= 0]].all()
    assert r.valor_total == pytest.approx(optimo_lp(e), rel=1e-4)
    assert r.valor_total >= asignar_greedy(e).valor_total - 1e-6


def test_con_maximos_y_valor():
    cli, ana, est = generar_datos_sinteticos(3_000, 6, seed=7)
    elig = cli.iloc[:, 1:].sum().to_numpy()
    est["cantidad_maxima"] = np.maximum((elig * 0.5).astype(int), est["cantidad_minima"].to_numpy())
    est["valor"] = np.linspace(1, 3, 6)
    e = cargar_entradas(cli, ana, est)
    r = asignar_nba(e)
    cnt = np.bincount(r.asignacion[r.asignacion >= 0], minlength=6)
    assert r.convergio and (cnt >= e.minimos).all() and (cnt <= e.maximos).all()
    assert r.valor_total == pytest.approx(optimo_lp(e), rel=1e-4)


def test_detecta_infactible():
    cli = pd.DataFrame({"id_cliente": [1, 2, 3], "A": [1, 1, 1], "B": [1, 1, 0]})
    ana = pd.DataFrame({"id_cliente": [1, 2, 3], "A": [.5, .5, .5], "B": [.5, .5, .5]})
    est = pd.DataFrame({"campana": ["A", "B"], "cantidad_minima": [2, 2]})
    ok, msg = verificar_factibilidad(cargar_entradas(cli, ana, est))
    assert not ok and "infactibles" in msg

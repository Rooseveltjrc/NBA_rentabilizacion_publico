"""Motor de asignación NBA.

Problema (asignación con cuotas):
    max   sum_ic  valor_c * score_ic * x_ic
    s.a.  sum_c x_ic <= 1                      (una campaña por cliente)
          minimo_c <= sum_i x_ic <= maximo_c   (estrategia de negocio)
          x_ic = 0 si el cliente no es elegible
          x_ic in {0, 1}

Es un problema de transporte (matriz totalmente unimodular), por lo que se resuelve
de forma exacta en su dual: cada campaña tiene un "precio" lambda_c y cada cliente
toma argmax_c (valor_c * score_ic + lambda_c). Los precios se ajustan con un esquema
tipo subasta (Bertsekas) hasta cumplir todas las cuotas. Coste por iteración O(n),
sin construir un modelo LP de n x c variables -> escala a millones de clientes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from .data import ID, EntradasNBA

SIN_CAMPANA = -1


@dataclass
class ResultadoNBA:
    asignacion: np.ndarray            # (n,) índice de campaña o -1
    precios: np.ndarray               # (c,) lambda_c: coste de oportunidad de cada cuota
    iteraciones: int
    convergio: bool
    entradas: EntradasNBA = field(repr=False)

    @property
    def valor_total(self) -> float:
        a = self.asignacion
        ok = a >= 0
        e = self.entradas
        return float((e.score[ok, a[ok]] * e.valor[a[ok]]).sum())

    def resumen(self) -> pd.DataFrame:
        e, a = self.entradas, self.asignacion
        conteo = np.bincount(a[a >= 0], minlength=len(e.campanas))
        score_sum = np.bincount(a[a >= 0], weights=e.score[a >= 0, a[a >= 0]], minlength=len(e.campanas))
        return pd.DataFrame({
            "campana": e.campanas,
            "elegibles": e.elegible.sum(0),
            "minimo": e.minimos,
            "asignados": conteo,
            "cumple_minimo": conteo >= e.minimos,
            "score_medio": np.divide(score_sum, conteo, out=np.zeros(len(conteo)), where=conteo > 0).round(4),
            "conversiones_esperadas": score_sum.round(1),
            "precio_sombra": self.precios.round(4),
        })

    def tabla_salida(self, top_alternativas: int = 2) -> pd.DataFrame:
        """Una fila por cliente: campaña NBA + siguientes mejores acciones (para fallback)."""
        e, a = self.entradas, self.asignacion
        camp = np.array(e.campanas + [None], dtype=object)
        ok = a >= 0
        out = pd.DataFrame({
            ID: e.ids,
            "campana_nba": camp[a],
            "score_nba": np.where(ok, e.score[np.arange(len(a)), np.maximum(a, 0)], np.nan),
        })
        if top_alternativas:
            util = _utilidad(e)
            util[ok, a[ok]] = -np.inf                       # excluir la ya asignada
            k = min(top_alternativas, util.shape[1])
            orden = np.argsort(-util, axis=1)[:, :k]
            for j in range(k):
                idx = orden[:, j]
                valido = np.isfinite(util[np.arange(len(a)), idx])
                out[f"alternativa_{j + 2}"] = np.where(valido, camp[idx], None)
        return out


def _utilidad(e: EntradasNBA) -> np.ndarray:
    u = e.score * e.valor[None, :]
    u[~e.elegible] = -np.inf
    return u


def verificar_factibilidad(e: EntradasNBA) -> tuple[bool, str]:
    """Comprueba con max-flow (sobre patrones de elegibilidad agregados) que los mínimos
    se pueden cumplir a la vez con una campaña por cliente."""
    patrones, conteo = np.unique(e.elegible, axis=0, return_counts=True)
    c, p = e.elegible.shape[1], len(patrones)
    # nodos: 0=fuente, 1..c=campañas, c+1..c+p=patrones, c+p+1=sumidero
    s, t = 0, c + p + 1
    inf = int(e.minimos.sum()) + 1
    filas, cols, caps = [], [], []
    for j in range(c):
        filas.append(s); cols.append(1 + j); caps.append(int(e.minimos[j]))
    pj, cj = np.nonzero(patrones)
    filas += (1 + cj).tolist(); cols += (1 + c + pj).tolist(); caps += [inf] * len(pj)
    filas += (1 + c + np.arange(p)).tolist(); cols += [t] * p; caps += conteo.tolist()
    g = csr_matrix((np.array(caps, dtype=np.int32), (filas, cols)), shape=(t + 1, t + 1))
    flujo = maximum_flow(g, s, t).flow_value
    requerido = int(e.minimos.sum())
    if flujo < requerido:
        cortas = [f"{e.campanas[j]} (elegibles={e.elegible[:, j].sum():,}, min={e.minimos[j]:,})"
                  for j in range(c) if e.elegible[:, j].sum() < e.minimos[j]]
        detalle = f" Campañas con menos elegibles que el mínimo: {cortas}" if cortas else \
            " Los mínimos compiten por los mismos clientes (revisar solapamiento de elegibilidad)."
        return False, f"Mínimos infactibles: se pueden cubrir {flujo:,} de {requerido:,}.{detalle}"
    return True, "ok"


def asignar_nba(e: EntradasNBA, umbral_no_contacto: float = 0.0, eps_final: float = 1e-6,
                max_iter: int = 5_000, verbose: bool = False) -> ResultadoNBA:
    """Asignación óptima con cuotas mínimas/máximas.

    umbral_no_contacto: utilidad de "no enviar nada". Un cliente sólo recibe campaña si
        valor*score + precio supera este umbral (los mínimos pueden forzarlo igualmente).
    eps_final: precisión de la subasta; la solución queda a <= n*eps_final del óptimo.
        Se usa epsilon-scaling (1e-2 -> eps_final) para evitar "guerras de precios" lentas
        cuando varias campañas compiten por los mismos pocos clientes.
    """
    factible, msg = verificar_factibilidad(e)
    if not factible:
        raise ValueError(msg)

    U = _utilidad(e)                                     # (n, c), -inf = no elegible
    UF, elegF = np.asfortranarray(U), np.asfortranarray(e.elegible)   # acceso rápido por columna
    n, c = U.shape
    lam = np.zeros(c, dtype=np.float64)
    nulo = umbral_no_contacto - 1e-9                     # empate -> preferir campaña

    def mejor_opcion(filas, excluir=None):
        A = U[filas] + lam[None, :]
        if excluir is not None:
            A[:, excluir] = -np.inf
        j = A.argmax(1)
        v = A[np.arange(len(filas)), j]
        sin = v <= nulo
        return np.where(sin, SIN_CAMPANA, j), np.where(sin, nulo, v)

    asig, mejor = mejor_opcion(np.arange(n))
    conteo = np.bincount(asig[asig >= 0], minlength=c)

    # Con tope (ajuste de holgura) se mueve sin sobrepaso: exactamente los d de menor valor
    # (incluidos empates), así nunca se cruza la cuota y no hay ciclos subir/bajar.
    def _seleccion(valores, d, eps, tope):
        if tope == np.inf:
            delta = np.partition(valores, d - 1)[d - 1] + eps
            return delta, valores < delta
        idx = np.argpartition(valores, d - 1)[:d]
        delta = min(valores[idx].max(), tope)
        sel = np.zeros(len(valores), bool)
        sel[idx[valores[idx] <= delta]] = True
        return delta, sel

    def subir(j, d, eps, tope=np.inf):
        """Sube lambda_j lo justo para atraer a los d clientes que menos pierden."""
        cand = np.flatnonzero(elegF[:, j] & (asig != j))
        if tope != np.inf:                       # no robar a campañas que están en su mínimo
            origen = asig[cand]
            cand = cand[(origen < 0) | (conteo[np.maximum(origen, 0)] > e.minimos[np.maximum(origen, 0)])]
        if len(cand) == 0:
            return False
        d = min(d, len(cand))
        arrepent = mejor[cand] - (UF[cand, j] + lam[j])
        delta, sel = _seleccion(arrepent, d, eps, tope)
        mueven = cand[sel]
        if len(mueven) == 0:
            return False
        lam[j] += delta
        np.subtract.at(conteo, asig[mueven][asig[mueven] >= 0], 1)
        asig[mueven] = j
        miembros = np.flatnonzero(asig == j)
        mejor[miembros] = UF[miembros, j] + lam[j]
        conteo[j] += len(mueven)
        return True

    def bajar(j, d, eps, tope=np.inf):
        """Baja lambda_j lo justo para liberar a los d clientes que menos pierden al salir."""
        miembros = np.flatnonzero(asig == j)
        alt_j, alt_v = mejor_opcion(miembros, excluir=j)
        if tope != np.inf:                       # no empujar hacia campañas en su máximo
            llena = (alt_j >= 0) & (conteo[np.maximum(alt_j, 0)] >= e.maximos[np.maximum(alt_j, 0)])
            miembros, alt_j, alt_v = miembros[~llena], alt_j[~llena], alt_v[~llena]
            if len(miembros) == 0:
                return False
            d = min(d, len(miembros))
        perdida = mejor[miembros] - alt_v
        delta, salen = _seleccion(perdida, d, eps, tope)
        if not salen.any():
            return False
        lam[j] -= delta
        mejor[miembros[~salen]] -= delta
        m_sal = miembros[salen]
        asig[m_sal], mejor[m_sal] = alt_j[salen], alt_v[salen]
        np.add.at(conteo, alt_j[salen][alt_j[salen] >= 0], 1)
        conteo[j] -= int(salen.sum())
        return True

    # Condiciones de optimalidad (holgura complementaria):
    #   minimo <= conteo <= maximo,  lambda > 0 => conteo == minimo,  lambda < 0 => conteo == maximo
    it, convergio = 0, False
    fases = [x for x in (1e-2, 1e-3, 1e-4, 1e-5) if x > eps_final] + [eps_final]
    for eps in fases:
        convergio, estado_previo = False, None
        while it < max_iter:
            cambios = False
            for j in range(c):
                if conteo[j] < e.minimos[j]:
                    cambios |= subir(j, int(e.minimos[j] - conteo[j]), eps)
                elif conteo[j] > e.maximos[j]:
                    cambios |= bajar(j, int(conteo[j] - e.maximos[j]), eps)
                elif lam[j] > eps and conteo[j] > e.minimos[j]:
                    cambios |= bajar(j, int(conteo[j] - e.minimos[j]), eps, tope=lam[j])
                elif lam[j] < -eps and conteo[j] < e.maximos[j]:
                    cambios |= subir(j, int(e.maximos[j] - conteo[j]), eps, tope=-lam[j])
            it += 1
            estado = (lam.round(9).tobytes(), conteo.tobytes())
            if not cambios or estado == estado_previo:   # estancado = sólo empates moviéndose
                convergio = True
                break
            estado_previo = estado
        if verbose:
            faltan = np.maximum(e.minimos - conteo, 0).sum()
            print(f"  eps={eps:.0e}: iter acumuladas {it}, faltan {faltan:,} para mínimos")

    asig = _refinar_lp(e, U, lam, asig, nulo, verbose=verbose)
    return ResultadoNBA(asig, lam, it, convergio, e)


def _refinar_lp(e, U, lam, asig, nulo, tau=1e-3, max_ambiguos=300_000, verbose=False):
    """Resuelve de forma exacta (LP de transporte, HiGHS) sólo a los clientes cuya mejor y
    segunda mejor opción a precios lambda están a menos de tau; el resto queda fijo."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack

    n, c = U.shape
    A = np.concatenate([U + lam[None, :], np.full((n, 1), nulo)], axis=1)
    top2 = -np.partition(-A, 1, axis=1)[:, :2]
    margen = top2[:, 0] - top2[:, 1]
    while True:
        amb = np.flatnonzero((margen < tau) & e.elegible.any(1))
        if len(amb) > max_ambiguos:
            amb = amb[np.argsort(margen[amb])[:max_ambiguos]]
        fijos = np.ones(n, bool); fijos[amb] = False
        af = asig[fijos]
        cnt_fijo = np.bincount(af[af >= 0], minlength=c)
        m_res = np.maximum(e.minimos - cnt_fijo, 0)
        M_res = e.maximos - cnt_fijo
        if (M_res >= 0).all() and len(amb):
            i, j = np.nonzero(e.elegible[amb])
            k = len(i)
            r_cli = coo_matrix((np.ones(k), (i, np.arange(k))), shape=(len(amb), k))
            r_cmp = coo_matrix((np.ones(k), (j, np.arange(k))), shape=(c, k))
            # objetivo: utilidad sobre la opción "no contactar" (nulo)
            res = linprog(-(U[amb[i], j] - nulo), A_ub=vstack([r_cli, r_cmp, -r_cmp]).tocsr(),
                          b_ub=np.concatenate([np.ones(len(amb)), M_res, -m_res]),
                          bounds=(0, 1), method="highs-ds")
            if res.status == 0:
                x = res.x > 0.5                  # matriz TU -> vértice entero
                nueva = asig.copy()
                nueva[amb] = SIN_CAMPANA
                nueva[amb[i[x]]] = j[x]
                if verbose:
                    print(f"  refinamiento LP: {len(amb):,} clientes ambiguos (tau={tau:g})")
                return nueva
        if len(amb) >= max_ambiguos or tau > 1:
            if verbose:
                print("  refinamiento LP omitido; se mantiene la solución de la subasta")
            return asig
        tau *= 10


def asignar_greedy(e: EntradasNBA) -> ResultadoNBA:
    """Línea base típica de negocio: llenar mínimos por campaña (de la más escasa a la
    más abundante) con los mejores scores, y el resto a su campaña de mayor score."""
    U = _utilidad(e)
    n, c = U.shape
    asig = np.full(n, SIN_CAMPANA)
    holgura = e.elegible.sum(0) - e.minimos
    for j in np.argsort(holgura):
        libres = np.flatnonzero(e.elegible[:, j] & (asig == SIN_CAMPANA))
        top = libres[np.argsort(-U[libres, j])[: e.minimos[j]]]
        asig[top] = j
    resto = np.flatnonzero((asig == SIN_CAMPANA) & e.elegible.any(1))
    asig[resto] = U[resto].argmax(1)
    return ResultadoNBA(asig, np.zeros(c), 1, True, e)

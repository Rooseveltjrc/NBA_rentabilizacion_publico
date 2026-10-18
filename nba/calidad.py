"""Validación de calidad de las entradas (completitud, duplicados, dominios, integridad).

Cada chequeo produce una fila: tabla, chequeo, columna, severidad, valor, umbral, detalle.
Severidad: ERROR (detiene el proceso), ADVERTENCIA (continúa, revisar), OK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FLAGS_RESULTADO, ID, SEP

ERROR, ADV, OK = "ERROR", "ADVERTENCIA", "OK"


@dataclass
class UmbralesCalidad:
    max_nulos_error: float = 0.05        # > 5% nulos en una columna -> ERROR
    max_nulos_advertencia: float = 0.0   # > 0% -> ADVERTENCIA
    min_cobertura_analitica: float = 0.95  # % de clientes con registro en analítica
    min_cobertura_resultados: float = 0.98  # % de ids de resultados que existen en clientes
    min_desviacion_score: float = 1e-3   # score casi constante -> no discrimina
    max_inconsistencia_embudo: float = 0.01  # % filas de resultados con embudo ilógico


@dataclass
class ReporteCalidad:
    filas: list = field(default_factory=list)

    def add(self, tabla, chequeo, severidad, columna="", valor=None, umbral=None, detalle=""):
        self.filas.append(dict(tabla=tabla, chequeo=chequeo, columna=columna, severidad=severidad,
                               valor=valor, umbral=umbral, detalle=detalle))

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)

    @property
    def errores(self) -> pd.DataFrame:
        d = self.df
        return d[d.severidad == ERROR] if len(d) else d

    @property
    def aprobado(self) -> bool:
        return len(self.errores) == 0

    def resumen(self) -> str:
        d = self.df
        cnt = d.severidad.value_counts().to_dict()
        lineas = [f"Calidad: {cnt.get(OK, 0)} OK, {cnt.get(ADV, 0)} advertencias, {cnt.get(ERROR, 0)} errores"]
        for _, r in d[d.severidad != OK].iterrows():
            lineas.append(f"  [{r.severidad}] {r.tabla}.{r.columna or '*'} {r.chequeo}: {r.detalle}")
        return "\n".join(lineas)


def _columnas(rep, tabla, df, requeridas):
    faltan = [c for c in requeridas if c not in df.columns]
    rep.add(tabla, "columnas_requeridas", ERROR if faltan else OK, valor=len(faltan),
            detalle=f"faltan {faltan}" if faltan else "")
    return not faltan


def _nulos(rep, tabla, df, cols, u: UmbralesCalidad):
    if not len(df):
        rep.add(tabla, "tabla_vacia", ERROR, detalle="0 filas")
        return
    tasa = df[cols].isna().mean()
    for c, t in tasa.items():
        if t > u.max_nulos_error:
            rep.add(tabla, "completitud", ERROR, c, round(t, 4), u.max_nulos_error, f"{t:.2%} nulos")
        elif t > u.max_nulos_advertencia:
            rep.add(tabla, "completitud", ADV, c, round(t, 4), u.max_nulos_advertencia, f"{t:.2%} nulos")
    if (tasa <= u.max_nulos_advertencia).all():
        rep.add(tabla, "completitud", OK, detalle=f"{len(cols)} columnas sin nulos")


def _duplicados(rep, tabla, df, llave) -> pd.DataFrame:
    """Duplicados exactos se eliminan (ADVERTENCIA); llave repetida con datos distintos = ERROR."""
    exactos = int(df.duplicated().sum())
    if exactos:
        rep.add(tabla, "duplicados_exactos", ADV, ",".join(llave), exactos,
                detalle=f"{exactos:,} filas idénticas eliminadas")
        df = df.drop_duplicates()
    llave_dup = int(df.duplicated(llave).sum())
    rep.add(tabla, "duplicados_llave", ERROR if llave_dup else OK, ",".join(llave), llave_dup,
            detalle=f"{llave_dup:,} registros con la misma llave y valores distintos" if llave_dup else "")
    if df[llave].isna().any().any():
        rep.add(tabla, "llave_nula", ERROR, ",".join(llave), int(df[llave].isna().any(axis=1).sum()))
    return df


def _flags(rep, tabla, df, cols):
    malos = {c: int((~df[c].dropna().isin([0, 1, True, False])).sum()) for c in cols}
    malos = {c: n for c, n in malos.items() if n}
    rep.add(tabla, "dominio_0_1", ERROR if malos else OK, valor=sum(malos.values()),
            detalle=f"valores fuera de {{0,1}}: {malos}" if malos else "")


def validar_entradas(clientes, analitica, estrategia, resultados=None, pesos=None,
                     u: UmbralesCalidad | None = None):
    """Devuelve (ReporteCalidad, tablas_limpias)."""
    u = u or UmbralesCalidad()
    rep = ReporteCalidad()

    # --- estrategia
    if _columnas(rep, "estrategia", estrategia, ["campana", "cantidad_minima"]):
        estrategia = _duplicados(rep, "estrategia", estrategia, ["campana"])
        _nulos(rep, "estrategia", estrategia, ["campana", "cantidad_minima"], u)
        neg = int((estrategia.cantidad_minima < 0).sum())
        if neg:
            rep.add("estrategia", "dominio", ERROR, "cantidad_minima", neg, detalle="mínimos negativos")
    campanas = estrategia["campana"].astype(str).tolist() if "campana" in estrategia else []

    # --- clientes
    if _columnas(rep, "clientes", clientes, [ID] + campanas):
        clientes = _duplicados(rep, "clientes", clientes, [ID])
        _nulos(rep, "clientes", clientes, campanas, u)
        _flags(rep, "clientes", clientes, campanas)
        sin_campana = int((clientes[campanas].fillna(0).sum(axis=1) == 0).sum())
        rep.add("clientes", "sin_elegibilidad", ADV if sin_campana else OK, valor=sin_campana,
                detalle=f"{sin_campana:,} clientes ({sin_campana / len(clientes):.1%}) no cumplen ninguna campaña")
        for c in campanas:
            if clientes[c].fillna(0).sum() == 0:
                rep.add("clientes", "campana_sin_elegibles", ERROR, c, 0, detalle="ningún cliente elegible")

    # --- analítica
    cols_ana = [c for c in analitica.columns if c != ID]
    if _columnas(rep, "analitica", analitica, [ID]):
        analitica = _duplicados(rep, "analitica", analitica, [ID])
        _nulos(rep, "analitica", analitica, cols_ana, u)
        sin_var = [c for c in campanas if not any(x == c or x.startswith(c + SEP) for x in cols_ana)]
        if sin_var:
            rep.add("analitica", "campana_sin_variables", ERROR, valor=len(sin_var), detalle=str(sin_var))
        num = analitica[cols_ana].apply(pd.to_numeric, errors="coerce")
        fuera = {c: int(n) for c, n in ((num < 0) | (num > 1)).sum().items() if n}
        rep.add("analitica", "rango_0_1", ERROR if fuera else OK, valor=sum(fuera.values()),
                detalle=f"valores fuera de [0,1]: {fuera}" if fuera else "")
        std = num.std()
        for c in std[std < u.min_desviacion_score].index:
            rep.add("analitica", "variable_constante", ADV, c, float(std[c]), u.min_desviacion_score,
                    "variable sin variación: no discrimina")
        if ID in clientes:
            cob = clientes[ID].isin(analitica[ID]).mean()
            rep.add("analitica", "cobertura_clientes", ERROR if cob < u.min_cobertura_analitica else OK,
                    ID, round(cob, 4), u.min_cobertura_analitica, f"{cob:.2%} de clientes con analítica")

    # --- pesos
    if pesos is not None and _columnas(rep, "pesos", pesos, ["campana", "variable", "peso"]):
        pesos = _duplicados(rep, "pesos", pesos, ["campana", "variable"])
        if (pesos.peso < 0).any():
            rep.add("pesos", "dominio", ERROR, "peso", int((pesos.peso < 0).sum()), detalle="pesos negativos")
        inexist = [f"{r.campana}{SEP}{r.variable}" for r in pesos.itertuples()
                   if f"{r.campana}{SEP}{r.variable}" not in analitica.columns]
        if inexist:
            rep.add("pesos", "variable_inexistente", ADV, valor=len(inexist), detalle=str(inexist[:10]))

    # --- resultados
    if resultados is not None and _columnas(rep, "resultados", resultados, [ID, "campana"] + FLAGS_RESULTADO):
        resultados = _duplicados(rep, "resultados", resultados, [ID, "campana"])
        _nulos(rep, "resultados", resultados, FLAGS_RESULTADO, u)
        _flags(rep, "resultados", resultados, FLAGS_RESULTADO)
        cob = resultados[ID].isin(clientes[ID]).mean() if ID in clientes else 1.0
        rep.add("resultados", "integridad_clientes", ERROR if cob < u.min_cobertura_resultados else OK,
                ID, round(cob, 4), u.min_cobertura_resultados, f"{cob:.2%} de ids existen en clientes")
        desconocidas = sorted(set(resultados.campana.astype(str)) - set(campanas))
        if desconocidas:
            rep.add("resultados", "campana_desconocida", ADV, "campana", len(desconocidas),
                    detalle=f"sin estrategia vigente (se usan sólo para calibrar si coinciden): {desconocidas[:10]}")
        r = resultados.fillna(0).astype({c: bool for c in FLAGS_RESULTADO})
        reglas = {
            "abrio_sin_recibir": r.abrio_comunicacion & ~r.recibio_mensaje,
            "no_contesto_sin_llamada": r.no_contesto_llamada & ~r.gestion_llamada,
            "no_llamar_sin_llamada": r.contesto_no_llamar & ~r.gestion_llamada,
            "compro_llamada_sin_llamada": r.compro_llamada & ~r.gestion_llamada,
            "no_contesto_y_contesto": r.no_contesto_llamada & r.contesto_no_llamar,
            "compro_llamada_sin_contestar": r.compro_llamada & r.no_contesto_llamada,
        }
        for nombre, mask in reglas.items():
            t = mask.mean()
            if t > 0:
                sev = ERROR if t > u.max_inconsistencia_embudo else ADV
                rep.add("resultados", "consistencia_embudo", sev, nombre, round(t, 4),
                        u.max_inconsistencia_embudo, f"{int(mask.sum()):,} filas ({t:.2%})")
        if not any(m.any() for m in reglas.values()):
            rep.add("resultados", "consistencia_embudo", OK, detalle="embudo coherente")

    return rep, dict(clientes=clientes, analitica=analitica, estrategia=estrategia,
                     resultados=resultados, pesos=pesos)

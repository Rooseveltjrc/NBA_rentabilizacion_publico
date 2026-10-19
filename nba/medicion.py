"""M6 · Medición de campañas ejecutadas.

Insumos:
  programacion: campana, canal, fecha_inicio, fecha_fin, clientes_programados [, costo_unitario]
  gestion:      id_cliente, campana, canal, fecha, evento
                eventos digitales: envio, entrega, apertura, click
                eventos de llamada: llamada, contacto, no_contesta, no_llamar
  ventas:       id_cliente, producto, fecha_venta [, monto, canal_venta]
  asignacion:   (opcional, salida del NBA) id_cliente, campana_nba, grupo  -> habilita el lift

Análisis:
  1. Cumplimiento de la programación (programado vs gestionado, gestiones fuera de ventana)
  2. Embudo por campaña y canal (entrega, apertura, click, contactabilidad)
  3. Atribución de ventas: una venta del producto se atribuye a la campaña si ocurre entre
     el primer toque y fecha_fin + ventana_dias. Canal = último toque de mayor intención
     antes de la venta (click/contacto > apertura/entrega > envío/llamada).
  4. Serie diaria de gestiones, clicks y ventas atribuidas
  5. Lift vs grupo control (conversión e ingresos incrementales) y ROI si hay costo
  6. Hallazgos automáticos en texto
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ID
from .experimentacion import medir_lift

EVENTOS_DIGITALES = ["envio", "entrega", "apertura", "click"]
EVENTOS_LLAMADA = ["llamada", "contacto", "no_contesta", "no_llamar"]
EVENTOS = EVENTOS_DIGITALES + EVENTOS_LLAMADA
TOQUE = {"envio", "llamada"}                     # el cliente fue gestionado
PRIORIDAD = {"click": 3, "contacto": 3, "apertura": 2, "entrega": 1, "envio": 0, "llamada": 0,
             "no_contesta": -1, "no_llamar": -1}


@dataclass
class ConfigMedicion:
    ventana_atribucion_dias: int = 30
    min_cumplimiento: float = 0.90     # por debajo -> alerta
    alpha: float = 0.05


def preparar(programacion, gestion, ventas, asignacion=None):
    """Tipos, vocabulario y chequeos mínimos. Devuelve (tablas, avisos)."""
    avisos = []
    p, g, v = programacion.copy(), gestion.copy(), ventas.copy()
    for df, cols, n in ((p, ["campana", "canal", "fecha_inicio", "fecha_fin", "clientes_programados"], "programacion"),
                        (g, [ID, "campana", "canal", "fecha", "evento"], "gestion"),
                        (v, [ID, "producto", "fecha_venta"], "ventas")):
        faltan = [c for c in cols if c not in df]
        if faltan:
            raise ValueError(f"{n}: faltan columnas {faltan}")
    p["fecha_inicio"], p["fecha_fin"] = pd.to_datetime(p.fecha_inicio), pd.to_datetime(p.fecha_fin)
    g["fecha"], v["fecha_venta"] = pd.to_datetime(g.fecha), pd.to_datetime(v.fecha_venta)
    g["evento"] = g.evento.str.lower().str.strip()
    desconocidos = sorted(set(g.evento) - set(EVENTOS))
    if desconocidos:
        avisos.append(f"Eventos desconocidos ignorados: {desconocidos}")
        g = g[g.evento.isin(EVENTOS)]
    if "monto" not in v:
        v["monto"] = np.nan
    sin_prog = sorted(set(g.campana) - set(p.campana))
    if sin_prog:
        avisos.append(f"Gestiones de campañas sin programación: {sin_prog}")
    dup = int(g.duplicated().sum())
    if dup:
        avisos.append(f"{dup:,} eventos de gestión duplicados eliminados")
        g = g.drop_duplicates()
    a = asignacion.copy() if asignacion is not None else None
    return {"programacion": p, "gestion": g, "ventas": v, "asignacion": a}, avisos


def _ventana(p, dias):
    w = p.groupby("campana").agg(inicio=("fecha_inicio", "min"), fin=("fecha_fin", "max"))
    w["fin_atribucion"] = w.fin + pd.Timedelta(days=dias)
    return w


def cumplimiento(p, g):
    prog = p.groupby(["campana", "canal"], as_index=False).agg(
        clientes_programados=("clientes_programados", "sum"),
        fecha_inicio=("fecha_inicio", "min"), fecha_fin=("fecha_fin", "max"))
    toques = g[g.evento.isin(TOQUE)].merge(prog[["campana", "canal", "fecha_inicio", "fecha_fin"]],
                                          on=["campana", "canal"], how="left")
    toques["en_ventana"] = toques.fecha.between(toques.fecha_inicio, toques.fecha_fin + pd.Timedelta(hours=23, minutes=59))
    ej = toques.groupby(["campana", "canal"]).agg(
        clientes_gestionados=(ID, "nunique"), gestiones=(ID, "size"),
        pct_en_ventana=("en_ventana", "mean"),
        primera_gestion=("fecha", "min"), ultima_gestion=("fecha", "max")).reset_index()
    out = prog.merge(ej, on=["campana", "canal"], how="left").fillna({"clientes_gestionados": 0, "gestiones": 0})
    out["pct_cumplimiento"] = out.clientes_gestionados / out.clientes_programados.replace(0, np.nan)
    return out


def embudo(g):
    cnt = (g.groupby(["campana", "evento"])[ID].nunique().unstack(fill_value=0)
           .reindex(columns=EVENTOS, fill_value=0))
    d = pd.DataFrame(index=cnt.index)
    for e in EVENTOS:
        d[e] = cnt[e]
    div = lambda a, b: np.where(b > 0, a / np.maximum(b, 1), np.nan)
    d["tasa_entrega"] = div(d.entrega, d.envio)
    d["tasa_apertura"] = div(d.apertura, d.entrega)
    d["ctr"] = div(d.click, d.entrega)                 # click / entregado
    d["ctor"] = div(d.click, d.apertura)               # click / abierto
    d["contactabilidad"] = div(d.contacto, d.llamada)
    d["pct_no_llamar"] = div(d.no_llamar, d.contacto)
    return d.reset_index()


def atribuir(p, g, v, dias):
    """Una fila por venta con su campaña atribuida (o 'organica') y canal del último toque."""
    w = _ventana(p, dias)
    primer = (g[g.evento.isin(TOQUE)].groupby([ID, "campana"]).fecha.min()
              .rename("primer_toque").reset_index())
    vv = v.reset_index(names="id_venta").merge(primer, left_on=[ID, "producto"], right_on=[ID, "campana"], how="left")
    vv = vv.merge(w[["fin_atribucion"]], left_on="producto", right_index=True, how="left")
    vv["atribuida"] = vv.primer_toque.notna() & (vv.fecha_venta >= vv.primer_toque) & \
        (vv.fecha_venta <= vv.fin_atribucion)
    vv["dias_a_conversion"] = np.where(vv.atribuida, (vv.fecha_venta - vv.primer_toque).dt.days, np.nan)

    # Canal: último toque de mayor intención anterior a la venta
    t = g.merge(vv.loc[vv.atribuida, ["id_venta", ID, "producto", "fecha_venta"]],
                left_on=[ID, "campana"], right_on=[ID, "producto"])
    t = t[t.fecha <= t.fecha_venta]
    t["prio"] = t.evento.map(PRIORIDAD)
    t = t[t.prio >= 0].sort_values(["id_venta", "prio", "fecha"])
    ultimo = t.groupby("id_venta").tail(1).set_index("id_venta")
    vv["canal_atribuido"] = vv.id_venta.map(ultimo.canal)
    vv["evento_atribuido"] = vv.id_venta.map(ultimo.evento)
    vv.loc[~vv.atribuida, ["canal_atribuido", "campana"]] = ["organica", None]
    vv.loc[vv.atribuida, "campana"] = vv.loc[vv.atribuida, "producto"]
    return vv.drop(columns=["fin_atribucion"])


def serie_diaria(g, ventas_atr):
    gest = g[g.evento.isin(TOQUE)].assign(dia=lambda d: d.fecha.dt.normalize()).groupby(["dia", "campana"]).size()
    clk = g[g.evento == "click"].assign(dia=lambda d: d.fecha.dt.normalize()).groupby(["dia", "campana"]).size()
    va = ventas_atr[ventas_atr.atribuida].assign(dia=lambda d: d.fecha_venta.dt.normalize()) \
        .groupby(["dia", "campana"]).size()
    s = pd.concat({"gestiones": gest, "clicks": clk, "ventas_atribuidas": va}, axis=1).fillna(0).astype(int)
    return s.reset_index()


def lift(p, v, asignacion, dias, alpha):
    """Conversión e ingresos de tratamiento vs control dentro de la ventana de cada campaña."""
    if asignacion is None or "grupo" not in asignacion:
        return None
    w = _ventana(p, dias)
    a = asignacion[asignacion.campana_nba.isin(w.index) & asignacion.grupo.notna()][[ID, "campana_nba", "grupo"]]
    vv = v.merge(w, left_on="producto", right_index=True)
    vv = vv[vv.fecha_venta.between(vv.inicio, vv.fin_atribucion)]
    compras = vv.groupby([ID, "producto"]).agg(n=("fecha_venta", "size"), monto=("monto", "sum")).reset_index()
    a = a.merge(compras, left_on=[ID, "campana_nba"], right_on=[ID, "producto"], how="left")
    a["compra"] = a.n.notna().astype(int)
    a["monto"] = a.monto.fillna(0)
    res = medir_lift(a.rename(columns={"campana_nba": "campana"}), alpha=alpha)
    ing = a.groupby(["campana_nba", "grupo"]).monto.mean().unstack()
    if "control" in ing:
        for _, r in res.iterrows():
            dif = ing.loc[r.campana, r.brazo] - ing.loc[r.campana, "control"]
            res.loc[_, "ingreso_medio_trat"] = round(ing.loc[r.campana, r.brazo], 2)
            res.loc[_, "ingreso_medio_ctrl"] = round(ing.loc[r.campana, "control"], 2)
            res.loc[_, "ingresos_incrementales"] = round(dif * r.n_trat, 0)
    return res


def roi(p, cumpl, lift_df):
    if "costo_unitario" not in p or lift_df is None or "ingresos_incrementales" not in lift_df:
        return None
    costo = cumpl.merge(p[["campana", "canal", "costo_unitario"]].drop_duplicates(["campana", "canal"]),
                        on=["campana", "canal"])
    costo = (costo.clientes_gestionados * costo.costo_unitario).groupby(costo.campana).sum().rename("costo")
    r = lift_df.groupby("campana").ingresos_incrementales.sum().to_frame().join(costo)
    r["roi"] = (r.ingresos_incrementales - r.costo) / r.costo.replace(0, np.nan)
    return r.reset_index()


def _n(x):
    return f"{x:,.0f}".replace(",", ".")


def _p(x, d=0):
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


def hallazgos(res, cfg: ConfigMedicion) -> list[str]:
    """Frases cortas con lo que hay que mirar primero (formato numérico es-CO)."""
    out = []
    c = res["cumplimiento"]
    bajas = c[c.pct_cumplimiento < cfg.min_cumplimiento]
    for r in bajas.itertuples():
        out.append(f"{r.campana} ({r.canal}): se gestionó el {_p(r.pct_cumplimiento)} de lo programado "
                   f"({_n(r.clientes_gestionados)} de {_n(r.clientes_programados)}).")
    fuera = c[c.pct_en_ventana < 0.98]
    for r in fuera.itertuples():
        out.append(f"{r.campana} ({r.canal}): {_p(1 - r.pct_en_ventana, 1)} de las gestiones cayó fuera de la ventana programada.")
    e = res["embudo"].dropna(subset=["ctr"])
    if len(e) > 1:
        lo, hi = e.loc[e.ctr.idxmin()], e.loc[e.ctr.idxmax()]
        out.append(f"CTR más bajo: {lo.campana} ({_p(lo.ctr, 1)}); más alto: {hi.campana} ({_p(hi.ctr, 1)}).")
    l = res.get("lift")
    if l is not None and len(l):
        sig = l[(l.ic_inf > 0)]
        nos = l[~(l.ic_inf > 0)]
        if len(sig):
            out.append("Con ventas incrementales demostradas (IC 95% sobre cero): "
                       + ", ".join(f"{r.campana} (+{_p(r.lift_rel)})" for r in sig.itertuples()) + ".")
        if len(nos):
            out.append("Sin efecto concluyente: " + ", ".join(nos.campana) +
                       ". Puede ser falta de efecto o un control pequeño; revisar el diseño experimental.")
    va = res["ventas"]
    tot = len(va)
    if tot:
        out.append(f"{_p(va.atribuida.mean())} de las {_n(tot)} ventas del periodo se atribuye a una campaña; "
                   f"el resto es orgánica o fuera de ventana.")
        med = va.dias_a_conversion.median()
        if pd.notna(med):
            out.append(f"La mitad de las ventas atribuidas ocurre en los primeros {med:.0f} días desde el primer contacto.")
    return out


def medir(programacion, gestion, ventas, asignacion=None, cfg: ConfigMedicion | None = None) -> dict:
    cfg = cfg or ConfigMedicion()
    t, avisos = preparar(programacion, gestion, ventas, asignacion)
    p, g, v = t["programacion"], t["gestion"], t["ventas"]
    res = {"avisos": avisos}
    res["cumplimiento"] = cumplimiento(p, g)
    res["embudo"] = embudo(g)
    res["ventas"] = atribuir(p, g, v, cfg.ventana_atribucion_dias)
    res["atribucion_canal"] = (res["ventas"][res["ventas"].atribuida]
                               .groupby(["campana", "canal_atribuido"]).size().unstack(fill_value=0).reset_index())
    res["serie_diaria"] = serie_diaria(g, res["ventas"])
    res["lift"] = lift(p, v, t["asignacion"], cfg.ventana_atribucion_dias, cfg.alpha)
    res["roi"] = roi(p, res["cumplimiento"], res["lift"])
    res["hallazgos"] = hallazgos(res, cfg)
    res["config"] = cfg
    return res

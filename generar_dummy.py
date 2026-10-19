"""Genera un set de prueba de punta a punta: entradas -> corrida del NBA -> resultados de la
ronda siguiente -> medición de lift.

  python generar_dummy.py [--clientes 50000] [--carpeta data_dummy]

Estructura generada:
  data_dummy/entradas/      las 5 tablas de entrada en CSV
  data_dummy/salidas/       todo lo que produce el pipeline (+ asignacion_nba.csv)
  data_dummy/ronda_2/       resultados simulados de la campaña ejecutada y su lift
  data_dummy/medicion/      insumos de M6 (programación, gestión, ventas) y reporte_medicion.html
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nba.data import FLAGS_RESULTADO, ID, generar_escenario_completo
from nba.experimentacion import medir_lift
from nba.medicion import medir
from nba.reporte_medicion import generar_html
from nba.pipeline import ConfigNBA, ejecutar

NOMBRES = ["tarjeta_credito", "credito_consumo", "seguro_vida", "cdt_inversion",
           "cuenta_nomina", "credito_vehiculo", "seguro_hogar", "credito_libranza"]


def renombrar(d: dict, n_campanas: int) -> dict:
    mapa = {f"camp_{i:02d}": NOMBRES[i] for i in range(n_campanas)}

    def cols(df):
        return df.rename(columns=lambda c: next((c.replace(k, v) for k, v in mapa.items() if c.startswith(k)), c))

    d["clientes"], d["analitica"] = cols(d["clientes"]), cols(d["analitica"])
    for t in ("pesos", "estrategia", "resultados"):
        d[t]["campana"] = d[t]["campana"].map(mapa)
    return d


def ensuciar(d: dict, rng) -> dict:
    """Problemas leves de calidad para ver al módulo de calidad en acción (sólo advertencias)."""
    d["clientes"] = pd.concat([d["clientes"], d["clientes"].sample(25, random_state=1)])   # duplicados exactos
    col = d["analitica"].columns[1]
    d["analitica"].loc[d["analitica"].sample(frac=0.005, random_state=2).index, col] = np.nan  # 0.5% nulos
    r = d["resultados"]
    idx = r[r.recibio_mensaje == 0].sample(frac=0.05, random_state=3).index                  # embudo ilógico
    r.loc[idx, "abrio_comunicacion"] = 1
    return d


def simular_ronda_2(asig: pd.DataFrame, rng) -> pd.DataFrame:
    """Resultados de ejecutar la asignación: el tratamiento compra con su probabilidad y el
    control compra orgánicamente (sin contacto) con un lift real distinto por campaña."""
    a = asig[asig.campana_nba.notna()].copy()
    camps = sorted(a.campana_nba.unique())
    lift_real = dict(zip(camps, np.linspace(0.05, 0.60, len(camps))))
    p_trat = a.score_nba.to_numpy()
    p_ctrl = p_trat / (1 + a.campana_nba.map(lift_real).to_numpy())
    trat = a.contactar.to_numpy()
    compra = rng.random(len(a)) < np.where(trat, p_trat, p_ctrl)
    m = len(a)
    recibio = trat & (rng.random(m) < 0.93)
    abrio = recibio & (rng.random(m) < 0.4)
    llamada = trat & ~a.no_llamar.to_numpy() & (rng.random(m) < 0.35)
    contesto = llamada & (rng.random(m) < 0.5)
    no_llamar = contesto & (rng.random(m) < 0.1)
    por_llamada = compra & contesto & ~no_llamar & (rng.random(m) < 0.4)
    r = pd.DataFrame({
        ID: a[ID].to_numpy(), "campana": a.campana_nba.to_numpy(), "grupo": a.grupo.to_numpy(),
        "recibio_mensaje": recibio, "abrio_comunicacion": abrio, "compro": compra & ~por_llamada,
        "gestion_llamada": llamada, "no_contesto_llamada": llamada & ~contesto,
        "contesto_no_llamar": no_llamar, "compro_llamada": por_llamada})
    r[FLAGS_RESULTADO] = r[FLAGS_RESULTADO].astype(np.int8)
    return r, pd.DataFrame({"campana": camps, "lift_real_simulado": [lift_real[c] for c in camps]})


TICKET = {"tarjeta_credito": 180_000, "credito_consumo": 450_000, "seguro_vida": 120_000,
          "cdt_inversion": 250_000, "cuenta_nomina": 90_000, "credito_vehiculo": 900_000,
          "seguro_hogar": 110_000, "credito_libranza": 380_000}
COSTO = {"email": 60, "sms": 140, "llamada": 2_800}


def simular_insumos_medicion(asig: pd.DataFrame, res2: pd.DataFrame, rng, inicio="2026-11-02", dur=14):
    """Programación, eventos de gestión con fecha y ventas a partir de la ronda 2 simulada.
    Incluye a propósito una campaña con ejecución incompleta y gestiones fuera de ventana."""
    H = pd.Timedelta(hours=1)
    camps = list(dict.fromkeys(asig.campana_nba.dropna()))
    camps.sort()
    prog, ev, ven = [], [], []
    for k, c in enumerate(camps):
        ini = pd.Timestamp(inicio) + pd.Timedelta(days=3 * k)
        fin = ini + pd.Timedelta(days=dur - 1)
        digital = "email" if k % 2 == 0 else "sms"
        r = res2[res2.campana == c].reset_index(drop=True)
        trat = (r.grupo != "control").to_numpy()
        m = len(r)
        u = lambda a, b: pd.Series(pd.to_timedelta(rng.uniform(a, b, m), unit="D"))
        td = lambda x, unit: pd.Series(pd.to_timedelta(x, unit=unit))
        ids = r[ID].to_numpy()
        envio = trat & (rng.random(m) < (0.84 if k == 1 else 0.995))      # una campaña se queda corta
        t_env = ini + u(0, dur * 0.5)
        fuera = envio & (rng.random(m) < (0.03 if k == 2 else 0.0))        # envíos antes de la ventana
        t_env = t_env.where(~fuera, ini - pd.Timedelta(days=2))
        entrega = envio & (r.recibio_mensaje == 1).to_numpy()
        apertura = entrega & (r.abrio_comunicacion == 1).to_numpy()
        t_ape = t_env + td(rng.exponential(18, m), "h")
        comp_d = (r.compro == 1).to_numpy()
        click = apertura & (comp_d | (rng.random(m) < 0.30))
        t_clk = t_ape + td(rng.exponential(2, m), "h")
        llam = (r.gestion_llamada == 1).to_numpy()
        t_ll = ini + u(2, dur)
        contacto = llam & (r.no_contesto_llamada == 0).to_numpy()
        for mask, t, e, ch in ((envio, t_env, "envio", digital), (entrega, t_env + 0.1 * H, "entrega", digital),
                               (apertura, t_ape, "apertura", digital), (click, t_clk, "click", digital),
                               (llam, t_ll, "llamada", "llamada"), (contacto, t_ll + 0.05 * H, "contacto", "llamada"),
                               (llam & ~contacto, t_ll + 0.05 * H, "no_contesta", "llamada"),
                               ((r.contesto_no_llamar == 1).to_numpy(), t_ll + 0.1 * H, "no_llamar", "llamada")):
            ev.append(pd.DataFrame({ID: ids[mask], "campana": c, "canal": ch, "fecha": t[mask].to_numpy(), "evento": e}))
        n_dig, n_ll = int(trat.sum()), int(llam.sum())
        prog += [dict(campana=c, canal=digital, fecha_inicio=ini.date(), fecha_fin=fin.date(),
                      clientes_programados=n_dig, costo_unitario=COSTO[digital]),
                 dict(campana=c, canal="llamada", fecha_inicio=ini.date(), fecha_fin=fin.date(),
                      clientes_programados=int(n_ll * 1.04), costo_unitario=COSTO["llamada"])]
        # Ventas: digitales tras el click/apertura/envío, por llamada el mismo día, control orgánico en la ventana
        base_dig = t_clk.where(click, t_ape.where(apertura, t_env))
        t_dig = base_dig + td(rng.exponential(3.5, m), "D")
        t_org = ini + u(0, dur + 25)
        comp_l = (r.compro_llamada == 1).to_numpy()
        t_v = np.where(comp_l, (t_ll + 0.2 * H).to_numpy(), np.where(trat & envio, t_dig.to_numpy(), t_org.to_numpy()))
        quien = comp_d | comp_l
        monto = rng.lognormal(np.log(TICKET.get(c, 200_000)), 0.35, m).round(-3)
        ven.append(pd.DataFrame({ID: ids[quien], "producto": c, "fecha_venta": pd.to_datetime(t_v[quien]),
                                 "monto": monto[quien], "canal_venta": np.where(comp_l, "llamada", "digital")[quien]}))
    gestion = pd.concat(ev, ignore_index=True).sort_values("fecha")
    gestion["fecha"] = gestion.fecha.dt.floor("min")
    ventas = pd.concat(ven, ignore_index=True).sort_values("fecha_venta")
    ventas["fecha_venta"] = ventas.fecha_venta.dt.floor("min")
    return pd.DataFrame(prog), gestion, ventas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clientes", type=int, default=50_000)
    ap.add_argument("--campanas", type=int, default=6)
    ap.add_argument("--carpeta", default="data_dummy")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = Path(a.carpeta)
    ent, sal, r2, med = base / "entradas", base / "salidas", base / "ronda_2", base / "medicion"
    for p in (ent, sal, r2, med):
        p.mkdir(parents=True, exist_ok=True)

    d = generar_escenario_completo(a.clientes, a.campanas, seed=a.seed)
    d = ensuciar(renombrar(d, a.campanas), rng)
    for t, df in d.items():
        df.to_csv(ent / f"{t}.csv", index=False, float_format="%.4f")
    print(f"Entradas en {ent}/: " + ", ".join(f"{t} {df.shape}" for t, df in d.items()))

    out = ejecutar(*(ent / f"{t}.csv" for t in ("clientes", "analitica", "estrategia", "resultados", "pesos")),
                   cfg=ConfigNBA.desde_json("config/parametros.json"), salida=sal)
    out["asignacion"].to_csv(sal / "asignacion_nba.csv", index=False, float_format="%.4f")

    res2, real = simular_ronda_2(out["asignacion"], rng)
    res2.to_csv(r2 / "resultados_ronda_2.csv", index=False)
    lift = medir_lift(res2).merge(real, on="campana")
    lift.to_csv(r2 / "lift_por_campana.csv", index=False)
    print("\n[Ronda 2] Lift medido vs. lift real simulado")
    print(lift[["campana", "n_trat", "n_ctrl", "conv_trat", "conv_ctrl", "lift_rel",
                "lift_real_simulado", "ic_inf", "ic_sup", "p_valor"]].to_string(index=False))

    # M6 · Medición: insumos con fecha y reporte visual
    prog, gest, vent = simular_insumos_medicion(out["asignacion"], res2, rng)
    prog.to_csv(med / "programacion.csv", index=False)
    gest.to_csv(med / "gestion.csv", index=False)
    vent.to_csv(med / "ventas.csv", index=False)
    res = medir(prog, gest, vent, out["asignacion"])
    for k in ("cumplimiento", "embudo", "atribucion_canal", "serie_diaria", "lift", "roi"):
        if res[k] is not None:
            res[k].to_csv(med / f"{k}.csv", index=False)
    (med / "reporte_medicion.html").write_text(generar_html(res), encoding="utf-8")
    print(f"\n[M6] Medición: {len(gest):,} eventos, {len(vent):,} ventas -> {med / 'reporte_medicion.html'}")
    for h in res["hallazgos"]:
        print("  -", h)


if __name__ == "__main__":
    main()

"""Genera un set de prueba de punta a punta: entradas -> corrida del NBA -> resultados de la
ronda siguiente -> medición de lift.

  python generar_dummy.py [--clientes 50000] [--carpeta data_dummy]

Estructura generada:
  data_dummy/entradas/      las 5 tablas de entrada en CSV
  data_dummy/salidas/       todo lo que produce el pipeline (+ asignacion_nba.csv)
  data_dummy/ronda_2/       resultados simulados de la campaña ejecutada y su lift
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nba.data import FLAGS_RESULTADO, ID, generar_escenario_completo
from nba.experimentacion import medir_lift
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clientes", type=int, default=50_000)
    ap.add_argument("--campanas", type=int, default=6)
    ap.add_argument("--carpeta", default="data_dummy")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = Path(a.carpeta)
    ent, sal, r2 = base / "entradas", base / "salidas", base / "ronda_2"
    for p in (ent, sal, r2):
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


if __name__ == "__main__":
    main()

"""Ejecuta el NBA de punta a punta.

Uso con datos reales:
  python run_nba.py --clientes data/clientes.parquet --analitica data/analitica.parquet \
      --estrategia data/estrategia.csv --resultados data/resultados.parquet \
      --pesos data/pesos.csv --salida salida/

Demo con datos sintéticos:
  python run_nba.py --demo 3000000 --campanas 15
"""
import argparse
from pathlib import Path

from nba.data import generar_escenario_completo
from nba.pipeline import ConfigNBA, ejecutar


def main():
    ap = argparse.ArgumentParser()
    for t in ("clientes", "analitica", "estrategia", "resultados", "pesos"):
        ap.add_argument(f"--{t}")
    ap.add_argument("--config", default="config/parametros.json", help="archivo de parámetros")
    ap.add_argument("--salida", default="salida")
    ap.add_argument("--demo", type=int, help="n clientes sintéticos")
    ap.add_argument("--campanas", type=int, default=8)
    a = ap.parse_args()

    cfg = ConfigNBA.desde_json(a.config) if Path(a.config).exists() else ConfigNBA()
    if a.demo:
        d = generar_escenario_completo(a.demo, a.campanas)
        ejecutar(d["clientes"], d["analitica"], d["estrategia"], d["resultados"], d["pesos"], cfg, a.salida)
    else:
        ejecutar(a.clientes, a.analitica, a.estrategia, a.resultados, a.pesos, cfg, a.salida)


if __name__ == "__main__":
    main()

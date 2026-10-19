"""M6 · Medición de campañas ejecutadas.

  python run_medicion.py --programacion programacion.csv --gestion gestion.csv --ventas ventas.csv \
      [--asignacion salida/asignacion_nba.parquet] [--ventana 30] [--salida medicion/]

Genera las tablas de análisis en CSV y reporte_medicion.html (ábrelo en el navegador).
"""
import argparse
from pathlib import Path

from nba.data import leer_tabla
from nba.medicion import ConfigMedicion, medir
from nba.reporte_medicion import generar_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programacion", required=True)
    ap.add_argument("--gestion", required=True)
    ap.add_argument("--ventas", required=True)
    ap.add_argument("--asignacion", help="salida del NBA con la columna grupo; habilita el lift")
    ap.add_argument("--ventana", type=int, default=30, help="días de atribución después de fecha_fin")
    ap.add_argument("--titulo", default="Medición de campañas")
    ap.add_argument("--salida", default="medicion")
    a = ap.parse_args()

    res = medir(leer_tabla(a.programacion), leer_tabla(a.gestion), leer_tabla(a.ventas),
                leer_tabla(a.asignacion) if a.asignacion else None, ConfigMedicion(ventana_atribucion_dias=a.ventana))
    d = Path(a.salida)
    d.mkdir(parents=True, exist_ok=True)
    for k in ("cumplimiento", "embudo", "atribucion_canal", "serie_diaria", "lift", "roi"):
        if res[k] is not None:
            res[k].to_csv(d / f"{k}.csv", index=False)
    res["ventas"].to_csv(d / "ventas_atribuidas.csv", index=False)
    (d / "reporte_medicion.html").write_text(generar_html(res, a.titulo), encoding="utf-8")
    print("Hallazgos:")
    for h in res["hallazgos"]:
        print("  -", h)
    print(f"Reporte: {d / 'reporte_medicion.html'}")


if __name__ == "__main__":
    main()

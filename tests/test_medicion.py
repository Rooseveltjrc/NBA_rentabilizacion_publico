import numpy as np
import pandas as pd
import pytest

from nba.data import ID
from nba.medicion import ConfigMedicion, medir
from nba.reporte_medicion import generar_html


def _caso():
    """2 campañas, 6 clientes: casos de atribución, canal, fuera de ventana y control."""
    prog = pd.DataFrame([
        dict(campana="A", canal="email", fecha_inicio="2026-11-01", fecha_fin="2026-11-10", clientes_programados=4, costo_unitario=100),
        dict(campana="A", canal="llamada", fecha_inicio="2026-11-01", fecha_fin="2026-11-10", clientes_programados=2, costo_unitario=1000),
        dict(campana="B", canal="sms", fecha_inicio="2026-11-05", fecha_fin="2026-11-15", clientes_programados=2, costo_unitario=200),
    ])
    g = pd.DataFrame([
        (1, "A", "email", "2026-11-02 09:00", "envio"), (1, "A", "email", "2026-11-02 09:01", "entrega"),
        (1, "A", "email", "2026-11-02 10:00", "apertura"), (1, "A", "email", "2026-11-02 10:05", "click"),
        (2, "A", "email", "2026-11-03 09:00", "envio"), (2, "A", "email", "2026-11-03 09:01", "entrega"),
        (2, "A", "llamada", "2026-11-04 15:00", "llamada"), (2, "A", "llamada", "2026-11-04 15:00", "contacto"),
        (3, "A", "email", "2026-10-25 09:00", "envio"),                       # fuera de ventana
        (4, "B", "sms", "2026-11-06 09:00", "envio"), (4, "B", "sms", "2026-11-06 09:00", "entrega"),
        (4, "B", "sms", "2026-11-06 09:00", "entrega"),                        # duplicado
    ], columns=[ID, "campana", "canal", "fecha", "evento"])
    v = pd.DataFrame([
        (1, "A", "2026-11-03", 100.0),     # tras click -> email
        (2, "A", "2026-11-05", 200.0),     # contacto en llamada (mayor intención que entrega) -> llamada
        (4, "B", "2026-11-01", 50.0),      # antes del primer toque -> orgánica
        (5, "A", "2026-11-04", 80.0),      # nunca contactado -> orgánica
        (1, "B", "2026-11-06", 30.0),      # compra B sin gestión de B -> orgánica
    ], columns=[ID, "producto", "fecha_venta", "monto"])
    asig = pd.DataFrame({ID: [1, 2, 3, 5, 6, 4], "campana_nba": ["A", "A", "A", "A", "A", "B"],
                         "grupo": ["tratamiento", "tratamiento", "tratamiento", "control", "control", "tratamiento"]})
    return prog, g, v, asig


def test_atribucion_y_canal():
    r = medir(*_caso())
    va = r["ventas"].set_index([ID, "producto"])
    assert va.loc[(1, "A"), "atribuida"] and va.loc[(1, "A"), "canal_atribuido"] == "email"
    assert va.loc[(2, "A"), "canal_atribuido"] == "llamada"
    assert not va.loc[(4, "B"), "atribuida"] and not va.loc[(5, "A"), "atribuida"]
    assert not va.loc[(1, "B"), "atribuida"]
    assert va.loc[(1, "A"), "dias_a_conversion"] == 0


def test_cumplimiento_embudo_y_avisos():
    r = medir(*_caso())
    c = r["cumplimiento"].set_index(["campana", "canal"])
    assert c.loc[("A", "email"), "clientes_gestionados"] == 3
    assert c.loc[("A", "email"), "pct_cumplimiento"] == pytest.approx(0.75)
    assert c.loc[("A", "email"), "pct_en_ventana"] == pytest.approx(2 / 3)
    e = r["embudo"].set_index("campana")
    assert e.loc["A", "click"] == 1 and e.loc["A", "ctor"] == pytest.approx(1.0)
    assert any("duplicados" in a for a in r["avisos"])
    assert any("fuera de la ventana" in h for h in r["hallazgos"])


def test_lift_y_roi():
    r = medir(*_caso())
    l = r["lift"].set_index("campana")
    assert l.loc["A", "conv_trat"] == pytest.approx(2 / 3, abs=1e-4) and l.loc["A", "conv_ctrl"] == pytest.approx(0.5)
    assert l.loc["A", "ingresos_incrementales"] == pytest.approx((300 / 3 - 80 / 2) * 3)
    assert set(r["roi"].columns) >= {"campana", "costo", "roi"}


def test_sin_asignacion_no_hay_lift_y_reporte_html():
    prog, g, v, _ = _caso()
    r = medir(prog, g, v, None, ConfigMedicion(ventana_atribucion_dias=10))
    assert r["lift"] is None and r["roi"] is None
    h = generar_html(r)
    assert h.startswith("<!doctype html>") and "¿Se ejecutó lo programado?" in h and "<svg" in h


def test_columnas_faltantes():
    prog, g, v, _ = _caso()
    with pytest.raises(ValueError, match="gestion"):
        medir(prog, g.drop(columns="evento"), v)

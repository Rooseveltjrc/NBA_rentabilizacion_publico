import numpy as np
import pandas as pd
import pytest

from nba import calidad, experimentacion as ex, scoring
from nba.data import ID, generar_escenario_completo
from nba.pipeline import ConfigNBA, ejecutar


@pytest.fixture(scope="module")
def esc():
    return generar_escenario_completo(30_000, 5, seed=1)


def test_calidad_ok(esc):
    rep, _ = calidad.validar_entradas(esc["clientes"], esc["analitica"], esc["estrategia"],
                                      esc["resultados"], esc["pesos"])
    assert rep.aprobado


def test_calidad_detecta_problemas(esc):
    cli = pd.concat([esc["clientes"], esc["clientes"].iloc[:10]])            # duplicados exactos
    cli2 = cli.copy(); cli2.iloc[0, 1] = 1 - cli2.iloc[0, 1]
    cli2 = pd.concat([cli2, cli2.iloc[[0]].assign(**{cli2.columns[1]: 1 - cli2.iloc[0, 1]})])  # llave en conflicto
    ana = esc["analitica"].copy(); ana.iloc[:3000, 1] = np.nan             # 10% nulos
    res = esc["resultados"].copy(); res.loc[:999, "abrio_comunicacion"] = 1; res.loc[:999, "recibio_mensaje"] = 0
    rep, limpio = calidad.validar_entradas(cli2, ana, esc["estrategia"], res)
    err = set(rep.errores.chequeo)
    assert {"duplicados_llave", "completitud", "consistencia_embudo"} <= err
    assert not rep.aprobado
    assert len(limpio["clientes"]) < len(cli2)                              # exactos eliminados


def test_ajuste_pesos_mejora_auc(esc):
    camps = esc["estrategia"].campana.tolist()
    pn = scoring.pesos_negocio(esc["pesos"], scoring.mapa_variables(esc["analitica"], camps))
    pesos, cal = scoring.ajustar_pesos(esc["analitica"], esc["resultados"], pn)
    assert cal.calibrada.all()
    assert np.allclose(pesos.groupby("campana").peso_final.sum(), 1, atol=1e-3)
    assert (cal.auc_ajustado >= cal.auc_negocio - 0.01).mean() >= 0.8
    s = scoring.calcular_scores(esc["analitica"], pesos, cal)
    assert s[camps].min().min() >= 0 and s[camps].max().max() <= 1


def test_sin_resultados_usa_pesos_negocio(esc):
    camps = esc["estrategia"].campana.tolist()
    pn = scoring.pesos_negocio(None, scoring.mapa_variables(esc["analitica"], camps))
    assert np.allclose(pn.peso_negocio, 1 / 3)
    pesos, cal = scoring.ajustar_pesos(esc["analitica"], None, pn)
    assert not cal.calibrada.any() and (pesos.peso_final == pesos.peso_negocio).all()


def test_tamano_muestral_conocido():
    assert abs(ex.n_por_grupo(0.10, 0.12) - 3839) <= 5          # valor de referencia clásico
    n_c = ex.control_minimo(50_000, 0.12, 0.2)
    assert ex.mde_alcanzable(50_000, n_c, 0.12) <= 0.2 + 1e-3
    assert ex.tamano_muestra_proporcion(10**6, error=0.02) == 2396


def test_aleatorizacion_exacta_y_balanceada():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({ID: range(20_000), "campana_nba": rng.choice(["A", "B"], 20_000),
                       "score_nba": rng.random(20_000)})
    cfg = ex.ConfigExperimento(modo="fijo", brazos={"msg_A": 0.45, "msg_B": 0.45, "control": 0.10})
    out, diseno, bal = ex.disenar_experimento(df, {"A": 0.1, "B": 0.1}, cfg)
    cnt = out.groupby(["campana_nba", "grupo"]).size().unstack()
    assert cnt.control.to_dict() == diseno.set_index("campana").n_control.to_dict()
    assert bal.balanceado.all()
    out2, *_ = ex.disenar_experimento(df, {"A": 0.1, "B": 0.1}, cfg)
    assert (out.grupo == out2.grupo).all()                      # reproducible


def test_muestra_estratificada():
    df = pd.DataFrame({"seg": np.repeat(["x", "y", "z"], [5000, 3000, 2000]), "v": np.arange(10_000)})
    m, plan = ex.muestra_estratificada(df, ["seg"], n=1000)
    assert len(m) == 1000 and plan.set_index("seg").muestra.to_dict() == {"x": 500, "y": 300, "z": 200}


def test_medir_lift():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"campana": "A", "grupo": np.r_[["tratamiento"] * 20000, ["control"] * 20000],
                      "compra": np.r_[rng.random(20000) < .12, rng.random(20000) < .10].astype(int)})
    l = ex.medir_lift(r).iloc[0]
    assert l.ic_inf < 0.02 < l.ic_sup and l.p_valor < 0.05


def test_pipeline_completo(esc, tmp_path):
    out = ejecutar(esc["clientes"], esc["analitica"], esc["estrategia"], esc["resultados"],
                   esc["pesos"], ConfigNBA(), tmp_path)
    ra = out["resumen_asignacion"]
    assert ra.cumple_minimo_negocio.all()
    assert out["comparativo"]["mejora_pct"] >= 0
    a = out["asignacion"]
    assert not a.loc[a.contactar, "campana_nba"].isna().any()
    # quien ya compró una campaña no vuelve a recibirla
    comp = esc["resultados"].query("compro == 1 or compro_llamada == 1")
    m = a.merge(comp[[ID, "campana"]], left_on=[ID, "campana_nba"], right_on=[ID, "campana"])
    assert m.empty
    assert (tmp_path / "asignacion_nba.parquet").exists()


def test_pipeline_detiene_por_calidad(esc, tmp_path):
    ana = esc["analitica"].copy(); ana.iloc[:, 1] = 5.0
    with pytest.raises(ValueError, match="calidad"):
        ejecutar(esc["clientes"], ana, esc["estrategia"], salida=tmp_path)
    assert (tmp_path / "reporte_calidad.csv").exists()


def test_config_json_coincide_con_defaults():
    from dataclasses import asdict
    assert asdict(ConfigNBA.desde_json("config/parametros.json")) == asdict(ConfigNBA())

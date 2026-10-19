"""Reporte HTML autocontenido del módulo de medición (SVG + JS mínimo, sin dependencias).

Se abre en cualquier navegador y también con displayHTML() en Databricks.
Cada gráfico trae tooltip al pasar el mouse y una tabla con los datos ("Ver datos").
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

# Paleta categórica validada (orden fijo) · rampa ordinal azul para el embudo
SERIE = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"]
ORDINAL = ["--o1", "--o2", "--o3", "--o4", "--o5"]


# ------------------------------------------------------------------ formato es-CO
def n0(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}".replace(",", ".")


def pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".") + "%"


def pts(x, d=1):
    return f"{x * 100:+.{d}f}".replace(".", ",") + " pts"


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return ("-$" if x < 0 else "$") + n0(abs(x))


def ticks(lo, hi, n=5):
    """Valores de eje redondos (1, 2, 2.5, 5 × 10^k) que cubren [lo, hi]."""
    span = (hi - lo) or 1
    raw = span / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return np.arange(np.floor(lo / step) * step, hi + step * 0.999, step)


def esc(s):
    return html.escape(str(s), quote=True)


def tabla(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    th = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    filas = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            txt = f(v) if f else (n0(v) if isinstance(v, (int, np.integer, float, np.floating)) and not isinstance(v, bool) else esc(v))
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            tds.append(f"<td class='{'num' if num else ''}'>{txt}</td>")
        filas.append("<tr>" + "".join(tds) + "</tr>")
    return f"<details><summary>Ver datos</summary><div class='tbl'><table><tr>{th}</tr>{''.join(filas)}</table></div></details>"


# ------------------------------------------------------------------ gráficos
def kpis(res) -> str:
    c, va, l, roi = res["cumplimiento"], res["ventas"], res["lift"], res["roi"]
    tiles = [
        ("Clientes gestionados", n0(c.clientes_gestionados.sum()),
         f"{pct(c.clientes_gestionados.sum() / c.clientes_programados.sum())} de lo programado"),
        ("Ventas atribuidas", n0(int(va.atribuida.sum())), f"{pct(va.atribuida.mean())} del total de ventas"),
    ]
    if l is not None and len(l):
        tiles.append(("Ventas incrementales", n0(l.ventas_incrementales.sum()),
                      "frente al grupo control"))
        if "ingresos_incrementales" in l:
            tiles.append(("Ingresos incrementales", money(l.ingresos_incrementales.sum()), "frente al grupo control"))
    if roi is not None:
        r = (roi.ingresos_incrementales.sum() - roi.costo.sum()) / roi.costo.sum()
        tiles.append(("ROI de la ola", pct(r, 0), f"costo {money(roi.costo.sum())}"))
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><span class='k'>{k}</span><b>{v}</b><span class='s'>{s}</span></div>" for k, v, s in tiles) + "</div>"


def chart_cumplimiento(c: pd.DataFrame) -> str:
    c = c.sort_values(["campana", "canal"]).reset_index(drop=True)
    W, L, R, rh = 720, 210, 120, 30
    H = 30 + rh * len(c) + 30
    tk = ticks(0, max(c.clientes_programados.max(), c.clientes_gestionados.max()))
    mx = tk[-1]
    X = lambda v: L + (W - L - R) * v / mx
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Clientes gestionados frente a programados'>"]
    for t in tk:
        out.append(f"<line class='grid' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='20' y2='{H - 26}'/>"
                   f"<text class='ax' x='{X(t):.1f}' y='{H - 8}' text-anchor='middle'>{n0(t)}</text>")
    for i, r in c.iterrows():
        y = 26 + i * rh
        low = r.pct_cumplimiento < 0.9
        tip = (f"{r.campana} · {r.canal}|Programados: {n0(r.clientes_programados)}|Gestionados: {n0(r.clientes_gestionados)}"
               f"|Cumplimiento: {pct(r.pct_cumplimiento)}|En ventana: {pct(r.pct_en_ventana)}")
        out.append(f"<g class='hit' data-tip='{esc(tip)}'><rect x='0' y='{y - 2}' width='{W}' height='{rh}' fill='transparent'/>"
                   f"<text class='lbl' x='{L - 10}' y='{y + 15}' text-anchor='end'>{esc(r.campana)} · {esc(r.canal)}</text>"
                   f"<rect x='{L}' y='{y + 4}' width='{max(X(r.clientes_gestionados) - L, 1):.1f}' height='16' rx='4' style='fill:var(--s1)'/>"
                   f"<line x1='{X(r.clientes_programados):.1f}' x2='{X(r.clientes_programados):.1f}' y1='{y}' y2='{y + 24}' class='target'/>"
                   f"<text class='val{' warn' if low else ''}' x='{W - R + 10}' y='{y + 16}'>{'⚠ ' if low else ''}{pct(r.pct_cumplimiento)}</text></g>")
    out.append("</svg>")
    leyenda = ("<div class='legend'><span><i style='background:var(--s1)'></i>Gestionados</span>"
               "<span><i class='tick'></i>Programados</span><span>⚠ menos de 90% de cumplimiento</span></div>")
    return leyenda + "".join(out)


def chart_embudo(e: pd.DataFrame, va: pd.DataFrame) -> str:
    vd = int(va.loc[va.atribuida & (va.evento_atribuido == "click"), "id_cliente"].nunique())
    etapas = [("Envíos", e.envio.sum()), ("Entregas", e.entrega.sum()), ("Aperturas", e.apertura.sum()),
              ("Clicks", e.click.sum()), ("Compraron tras el click", vd)]
    W, L, rh = 720, 180, 44
    H = rh * len(etapas) + 16
    mx = etapas[0][1] or 1
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Embudo digital de todas las campañas'>"]
    for i, (n, v) in enumerate(etapas):
        y = 8 + i * rh
        w = max((W - L - 150) * v / mx, 2)
        paso = f"{pct(v / etapas[i - 1][1])} de la etapa anterior" if i and etapas[i - 1][1] else ""
        tip = f"{n}|{n0(v)} clientes" + (f"|{paso}" if paso else "")
        out.append(f"<g class='hit' data-tip='{esc(tip)}'><text class='lbl' x='{L - 10}' y='{y + 22}' text-anchor='end'>{n}</text>"
                   f"<rect x='{L}' y='{y + 6}' width='{w:.1f}' height='24' rx='4' style='fill:var({ORDINAL[i]})'/>"
                   f"<text class='val' x='{L + w + 8:.1f}' y='{y + 22}'>{n0(v)}"
                   f"<tspan class='muted'>{('  ·  ' + pct(v / etapas[i - 1][1])) if i and etapas[i - 1][1] else ''}</tspan></text></g>")
    out.append("</svg>")
    return "".join(out)


def chart_lineas(s: pd.DataFrame, campanas: list[str]) -> str:
    s = s.copy()
    s["dia"] = pd.to_datetime(s.dia)
    dias = pd.date_range(s.dia.min(), s.dia.max(), freq="D")
    piv = s.pivot_table(index="dia", columns="campana", values="ventas_atribuidas", aggfunc="sum").reindex(dias, fill_value=0)
    piv = piv.reindex(columns=[c for c in campanas if c in piv.columns], fill_value=0).fillna(0)
    W, L, R, T, B = 720, 44, 24, 14, 30
    H = 280
    tk = ticks(0, max(piv.values.max(), 1) * 1.12)
    mx = tk[-1]
    X = lambda i: L + (W - L - R) * i / max(len(dias) - 1, 1)
    Y = lambda v: T + (H - T - B) * (1 - v / mx)
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Ventas atribuidas por día y campaña'>"]
    for t in tk:
        out.append(f"<line class='grid' x1='{L}' x2='{W - R}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
                   f"<text class='ax' x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{n0(t)}</text>")
    paso = max(len(dias) // 6, 1)
    for i in range(0, len(dias), paso):
        anc = "end" if i > len(dias) - paso else ("start" if i == 0 else "middle")
        out.append(f"<text class='ax' x='{X(i):.1f}' y='{H - 10}' text-anchor='{anc}'>{dias[i]:%d/%m}</text>")
    out.append(f"<line class='base' x1='{L}' x2='{W - R}' y1='{Y(0):.1f}' y2='{Y(0):.1f}'/>")
    picos = []
    for k, c in enumerate(piv.columns):
        col = SERIE[campanas.index(c) % 8]
        pts_ = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(piv[c]))
        out.append(f"<polyline points='{pts_}' style='stroke:var({col})' class='line'/>")
        i_max = int(np.argmax(piv[c].to_numpy()))
        picos.append([X(i_max), Y(piv[c].iloc[i_max]), c, col])
    # etiqueta directa sobre el pico de cada curva; se sube si choca con otra cercana
    picos.sort(key=lambda p: p[1])
    puestas = []
    for x, y, c, col in picos:
        ty = y - 10
        while any(abs(ty - py) < 14 and abs(x - px) < 110 for px, py in puestas):
            ty -= 14
        ty = max(ty, T + 10)
        puestas.append((x, ty))
        anc = "end" if x > W - R - 80 else ("start" if x < L + 40 else "middle")
        out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' style='fill:var({col});stroke:var(--surface);stroke-width:2'/>"
                   f"<text class='lbl halo' x='{x:.1f}' y='{ty:.1f}' text-anchor='{anc}'>{esc(c)}</text>")
    # capa de hover por día (crosshair)
    cw = (W - L - R) / max(len(dias) - 1, 1)
    for i, d in enumerate(dias):
        filas = "|".join(f"{c}: {n0(piv.loc[d, c])}" for c in piv.columns)
        out.append(f"<g class='hit col' data-tip='{esc(f'{d:%d/%m/%Y}|{filas}|Total: {n0(piv.loc[d].sum())}')}'>"
                   f"<rect x='{X(i) - cw / 2:.1f}' y='{T}' width='{cw:.1f}' height='{H - T - B}' fill='transparent'/>"
                   f"<line class='cross' x1='{X(i):.1f}' x2='{X(i):.1f}' y1='{T}' y2='{H - B}'/></g>")
    out.append("</svg>")
    ley = "<div class='legend'>" + "".join(
        f"<span><i style='background:var({SERIE[campanas.index(c) % 8]})'></i>{esc(c)}</span>" for c in piv.columns) + "</div>"
    return ley + "".join(out), piv.reset_index(names="dia")


def chart_canal(a: pd.DataFrame) -> str:
    """Pequeños múltiplos: una columna por canal, % de ventas atribuidas de cada campaña."""
    canales = [c for c in ["email", "sms", "llamada"] if c in a.columns] + \
              [c for c in a.columns if c not in ("campana", "email", "sms", "llamada")]
    a = a.set_index("campana")[canales]
    share = a.div(a.sum(axis=1), axis=0).fillna(0)
    W, L, rh = 720, 150, 30
    cw = (W - L) / len(canales)
    H = 34 + rh * len(share) + 6
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Participación de cada canal en las ventas atribuidas'>"]
    for j, ch in enumerate(canales):
        x0 = L + j * cw
        out.append(f"<text class='lbl b' x='{x0 + 4:.1f}' y='18'>{esc(ch)}</text>"
                   f"<line class='base' x1='{x0 + 4:.1f}' x2='{x0 + 4:.1f}' y1='26' y2='{H - 4}'/>")
        for i, (camp, r) in enumerate(share.iterrows()):
            y = 30 + i * rh
            w = max((cw - 70) * r[ch], 0)
            tip = f"{camp} · {ch}|{n0(a.loc[camp, ch])} ventas|{pct(r[ch])} de las ventas atribuidas de la campaña"
            out.append(f"<g class='hit' data-tip='{esc(tip)}'><rect x='{x0 + 4:.1f}' y='{y + 5}' width='{cw - 10:.1f}' height='18' fill='transparent'/>"
                       f"<rect x='{x0 + 4:.1f}' y='{y + 6}' width='{w:.1f}' height='16' rx='4' style='fill:var(--s1)'/>"
                       f"<text class='val' x='{x0 + 10 + w:.1f}' y='{y + 19}'>{pct(r[ch], 0)}</text></g>")
    for i, camp in enumerate(share.index):
        out.append(f"<text class='lbl' x='{L - 10}' y='{30 + i * rh + 19}' text-anchor='end'>{esc(camp)}</text>")
    out.append("</svg>")
    return "".join(out)


def chart_lift(l: pd.DataFrame) -> str:
    l = l.sort_values("lift_abs", ascending=False).reset_index(drop=True)
    W, L, R, rh = 720, 150, 150, 34
    H = 20 + rh * len(l) + 34
    tk = ticks(min(l.ic_inf.min(), 0), max(l.ic_sup.max(), 0))
    lo, hi = tk[0], tk[-1]
    X = lambda v: L + (W - L - R) * (v - lo) / (hi - lo)
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Lift de conversión frente al control con intervalo de 95%'>"]
    for t in tk:
        out.append(f"<line class='grid' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='12' y2='{H - 26}'/>"
                   f"<text class='ax' x='{X(t):.1f}' y='{H - 8}' text-anchor='middle'>{pts(t, 0) if abs(t) > 1e-12 else '0'}</text>")
    out.append(f"<line class='zero' x1='{X(0):.1f}' x2='{X(0):.1f}' y1='8' y2='{H - 26}'/>")
    for i, r in l.iterrows():
        y = 26 + i * rh
        sig = r.ic_inf > 0
        tip = (f"{r.campana}|Conversión tratamiento: {pct(r.conv_trat)} (n={n0(r.n_trat)})|Conversión control: {pct(r.conv_ctrl)} (n={n0(r.n_ctrl)})"
               f"|Lift: {pts(r.lift_abs)} ({pct(r.lift_rel, 0)})|IC 95%: {pts(r.ic_inf)} a {pts(r.ic_sup)}|p-valor: {str(r.p_valor).replace('.', ',')}")
        out.append(f"<g class='hit' data-tip='{esc(tip)}'><rect x='0' y='{y - 12}' width='{W}' height='{rh}' fill='transparent'/>"
                   f"<text class='lbl' x='{L - 10}' y='{y + 4}' text-anchor='end'>{esc(r.campana)}</text>"
                   f"<line class='whisk' x1='{X(r.ic_inf):.1f}' x2='{X(r.ic_sup):.1f}' y1='{y}' y2='{y}'/>"
                   f"<line class='whisk' x1='{X(r.ic_inf):.1f}' x2='{X(r.ic_inf):.1f}' y1='{y - 5}' y2='{y + 5}'/>"
                   f"<line class='whisk' x1='{X(r.ic_sup):.1f}' x2='{X(r.ic_sup):.1f}' y1='{y - 5}' y2='{y + 5}'/>"
                   f"<circle cx='{X(r.lift_abs):.1f}' cy='{y}' r='5' class='dot{'' if sig else ' hollow'}'/>"
                   f"<text class='val' x='{W - R + 12}' y='{y + 4}'>{pct(r.lift_rel, 0)} "
                   f"<tspan class='{'good' if sig else 'muted'}'>{'✓ significativo' if sig else '○ no concluyente'}</tspan></text></g>")
    out.append("</svg>")
    return ("<div class='legend'><span><i class='dotk'></i>Lift (punto) e intervalo de 95% (línea)</span>"
            "<span>Relleno = el intervalo no toca cero</span></div>" + "".join(out))


def chart_histograma(va: pd.DataFrame, dias_max: int) -> str:
    d = va.loc[va.atribuida, "dias_a_conversion"].dropna().clip(upper=dias_max).astype(int)
    cnt = d.value_counts().reindex(range(dias_max + 1), fill_value=0)
    W, L, B, T = 720, 44, 30, 12
    H = 200
    tk = ticks(0, max(cnt.max(), 1), 4)
    mx = tk[-1]
    bw = (W - L - 10) / len(cnt)
    Y = lambda v: T + (H - T - B) * (1 - v / mx)
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Días entre el primer contacto y la venta'>"]
    for t in tk:
        out.append(f"<line class='grid' x1='{L}' x2='{W - 10}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
                   f"<text class='ax' x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{n0(t)}</text>")
    med = d.median() if len(d) else 0
    for i, v in cnt.items():
        x = L + i * bw
        out.append(f"<g class='hit' data-tip='{esc(f'Día {i}' + ('+' if i == dias_max else '') + f'|{n0(v)} ventas')}'>"
                   f"<rect x='{x:.1f}' y='{T}' width='{bw:.1f}' height='{H - T - B}' fill='transparent'/>"
                   f"<rect x='{x + 1:.1f}' y='{Y(v):.1f}' width='{max(bw - 2, 1):.1f}' height='{Y(0) - Y(v):.1f}' rx='2' style='fill:var(--s1)'/></g>")
        if i % 5 == 0:
            out.append(f"<text class='ax' x='{x + bw / 2:.1f}' y='{H - 10}' text-anchor='middle'>{i}</text>")
    xm = L + (med + 0.5) * bw
    out.append(f"<line class='zero' x1='{xm:.1f}' x2='{xm:.1f}' y1='{T}' y2='{H - B}'/>"
               f"<text class='lbl' x='{xm + 6:.1f}' y='{T + 12}'>mediana: {med:.0f} días</text>")
    out.append(f"<line class='base' x1='{L}' x2='{W - 10}' y1='{Y(0):.1f}' y2='{Y(0):.1f}'/></svg>")
    return "".join(out), cnt.rename_axis("dias").reset_index(name="ventas")


# ------------------------------------------------------------------ documento
CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--good:#006300;--warn:#a35f00;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--o1:#86b6ef;--o2:#5598e7;--o3:#2a78d6;--o4:#1c5cab;--o5:#104281;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--good:#0ca30c;--warn:#fab219;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--o1:#184f95;--o2:#1c5cab;--o3:#2a78d6;--o4:#5598e7;--o5:#86b6ef;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--good:#0ca30c;--warn:#fab219;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--o1:#184f95;--o2:#1c5cab;--o3:#2a78d6;--o4:#5598e7;--o5:#86b6ef;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:980px;margin:0 auto;padding-block:32px 64px;display:flex;flex-direction:column;gap:28px}
h1{font-size:1.8rem;margin:0;line-height:1.2;text-wrap:balance}h2{font-size:1.15rem;margin:0}
.sub{color:var(--ink2);margin:4px 0 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.kpi .k{font-size:.78rem;color:var(--ink2);text-transform:uppercase;letter-spacing:.05em}.kpi b{font-size:1.7rem;font-weight:600}
.kpi .s{font-size:.85rem;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:18px;display:flex;flex-direction:column;gap:10px;min-width:0}
.card p{margin:0;color:var(--ink2);max-width:70ch}
.card svg{width:100%;height:auto;display:block;min-width:560px}
.scroll{overflow-x:auto}
ul.find{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend i.tick{width:2px;height:14px;background:var(--ink);border-radius:0}
.legend i.dotk{width:10px;height:10px;border-radius:50%;background:var(--s1)}
text{font-family:inherit}
.ax{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.lbl{fill:var(--ink);font-size:12.5px}.lbl.b{font-weight:600}
.halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.val{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}.val.warn{fill:var(--warn);font-weight:600}
.muted{fill:var(--muted)}.good{fill:var(--good);font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}.base{stroke:var(--axis);stroke-width:1}.zero{stroke:var(--ink2);stroke-width:1}
.target{stroke:var(--ink);stroke-width:2}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.whisk{stroke:var(--s1);stroke-width:2}.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}.dot.hollow{fill:var(--surface);stroke:var(--s1)}
.cross{stroke:var(--ink2);stroke-width:1;opacity:0}.col:hover .cross{opacity:.5}
.hit{cursor:default}.hit:hover rect[style]{opacity:.85}
details summary{cursor:pointer;color:var(--ink2);font-size:.88rem}
.tbl{overflow-x:auto;margin-top:8px}
table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}th{color:var(--ink2);font-weight:600}
td.num{text-align:right}
.avisos{font-size:.88rem;color:var(--ink2)}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:.82rem;box-shadow:0 4px 16px rgba(0,0,0,.12);max-width:300px;display:none;z-index:9;line-height:1.45}
#tip b{display:block;margin-bottom:2px}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.hit').forEach(g=>{
 g.addEventListener('pointermove',e=>{const p=g.dataset.tip.split('|');
  tip.innerHTML='<b>'+p[0].replace(/</g,'&lt;')+'</b>'+p.slice(1).map(s=>s.replace(/</g,'&lt;')).join('<br>');
  tip.style.display='block';const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);
  tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';});
 g.addEventListener('pointerleave',()=>tip.style.display='none');});
"""


def generar_html(res: dict, titulo="Medición de campañas") -> str:
    c, e, va, l = res["cumplimiento"], res["embudo"], res["ventas"], res["lift"]
    campanas = list(dict.fromkeys(c.sort_values("fecha_inicio").campana))
    ini, fin = c.fecha_inicio.min(), c.fecha_fin.max()
    dias = res["config"].ventana_atribucion_dias
    lineas, tabla_lineas = chart_lineas(res["serie_diaria"], campanas)
    hist, tabla_hist = chart_histograma(va, dias)

    secciones = [
        f"<header><h1>{esc(titulo)}</h1><p class='sub'>Ola del {ini:%d/%m/%Y} al {fin:%d/%m/%Y} · "
        f"{len(campanas)} campañas · ventana de atribución {dias} días · generado {datetime.now():%d/%m/%Y %H:%M}</p></header>",
        kpis(res),
        "<section class='card'><h2>Lo que hay que mirar primero</h2><ul class='find'>"
        + "".join(f"<li>{esc(h)}</li>" for h in res["hallazgos"]) + "</ul></section>",
        "<section class='card'><h2>¿Se ejecutó lo programado?</h2><p>Clientes gestionados (barra) frente a los programados (marca), por campaña y canal.</p>"
        f"<div class='scroll'>{chart_cumplimiento(c)}</div>"
        + tabla(c[["campana", "canal", "clientes_programados", "clientes_gestionados", "pct_cumplimiento", "pct_en_ventana"]],
                {"pct_cumplimiento": pct, "pct_en_ventana": pct}) + "</section>",
        "<section class='card'><h2>Embudo digital</h2><p>Clientes únicos en cada etapa, sumando todas las campañas. El porcentaje compara con la etapa anterior.</p>"
        f"<div class='scroll'>{chart_embudo(e, va)}</div>"
        + tabla(e[["campana", "envio", "entrega", "apertura", "click", "tasa_entrega", "tasa_apertura", "ctr", "ctor",
                   "llamada", "contacto", "contactabilidad", "pct_no_llamar"]],
                {k: pct for k in ["tasa_entrega", "tasa_apertura", "ctr", "ctor", "contactabilidad", "pct_no_llamar"]}) + "</section>",
        "<section class='card'><h2>Ventas atribuidas por día</h2><p>Ventas del producto ocurridas después del primer contacto y dentro de la ventana de atribución.</p>"
        f"<div class='scroll'>{lineas}</div>" + tabla(tabla_lineas.assign(dia=tabla_lineas.dia.dt.strftime("%d/%m/%Y"))) + "</section>",
        "<section class='card'><h2>¿Por qué canal llegan las ventas?</h2><p>Participación del último toque de mayor intención (click o contacto en llamada) en las ventas atribuidas de cada campaña.</p>"
        f"<div class='scroll'>{chart_canal(res['atribucion_canal'])}</div>" + tabla(res["atribucion_canal"]) + "</section>",
    ]
    if l is not None and len(l):
        cols = ["campana", "n_trat", "n_ctrl", "conv_trat", "conv_ctrl", "lift_abs", "lift_rel", "ic_inf", "ic_sup", "p_valor",
                "ventas_incrementales"] + [x for x in ["ingresos_incrementales"] if x in l]
        secciones.append(
            "<section class='card'><h2>¿La campaña generó ventas que no habrían ocurrido?</h2>"
            "<p>Diferencia de conversión entre contactados y grupo control. Si el intervalo no toca cero, el efecto es real con 95% de confianza.</p>"
            f"<div class='scroll'>{chart_lift(l)}</div>"
            + tabla(l[cols], {k: pct for k in ["conv_trat", "conv_ctrl", "lift_rel"]} |
                    {k: pts for k in ["lift_abs", "ic_inf", "ic_sup"]} | {"p_valor": lambda x: str(x).replace(".", ","),
                                                                        "ingresos_incrementales": money}) + "</section>")
    if res["roi"] is not None:
        secciones.append("<section class='card'><h2>Retorno por campaña</h2><p>Ingresos incrementales frente al costo de gestión.</p>"
                         + tabla(res["roi"], {"ingresos_incrementales": money, "costo": money, "roi": lambda x: pct(x, 0)})
                         .replace("<details>", "<details open>") + "</section>")
    secciones.append("<section class='card'><h2>¿Cuánto tarda en llegar la venta?</h2><p>Días entre el primer contacto y la venta atribuida.</p>"
                     f"<div class='scroll'>{hist}</div>" + tabla(tabla_hist) + "</section>")
    if res["avisos"]:
        secciones.append("<section class='avisos'><b>Avisos de calidad de los insumos:</b><ul>"
                         + "".join(f"<li>{esc(a)}</li>" for a in res["avisos"]) + "</ul></section>")
    return (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(titulo)}</title><style>{CSS}</style></head><body><div class='wrap'>{''.join(secciones)}</div>"
            f"<div id='tip' role='tooltip'></div><script>{JS}</script></body></html>")

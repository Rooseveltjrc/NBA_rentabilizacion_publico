# NBA de campañas comerciales

Motor *Next Best Action* que asigna **una campaña por cliente** maximizando las conversiones
esperadas, cumpliendo los mínimos de negocio, con pesos de score ajustados por resultados
y grupos de control para medir el impacto incremental.

```
[1] Calidad ─► [2] Scores ponderados + ajuste con resultados ─► [3] Estrategia
    ─► [4] Optimización ─► [5] Diseño experimental ─► Salidas
                                                        │ campaña ejecutada
[6] Medición (ventas, clicks, gestión, programación) ◄──┘ ─► reporte + vuelve a [2]
```

## Entradas

| # | Tabla | Columnas | Notas |
|---|-------|----------|-------|
| 1 | `clientes` | `id_cliente`, `<campana>`… | 1/0 = cumple condiciones |
| 2 | `analitica` | `id_cliente`, `<campana>__<variable>`… | scores/variables en [0,1]. Una columna llamada `<campana>` se toma como variable `score` |
| – | `pesos` (opcional) | `campana`, `variable`, `peso` | importancia de negocio por variable; sin tabla = pesos iguales |
| 3 | `estrategia` | `campana`, `cantidad_minima` [, `cantidad_maxima`, `valor`] | `valor` = peso económico de la campaña (1 por defecto) |
| 4 | `resultados` (opcional) | `id_cliente`, `campana`, `recibio_mensaje`, `abrio_comunicacion`, `compro`, `gestion_llamada`, `no_contesto_llamada`, `contesto_no_llamar`, `compro_llamada` [, `grupo`] | flags 0/1 |

```bash
pip install -r requirements.txt
python run_nba.py --config config/parametros.json --clientes clientes.parquet --analitica analitica.parquet \
    --estrategia estrategia.csv --resultados resultados.parquet --pesos pesos.csv --salida salida/
python run_nba.py --demo 3000000 --campanas 15      # escenario sintético
python generar_dummy.py                              # set de prueba en CSV: data_dummy/
python run_medicion.py --programacion programacion.csv --gestion gestion.csv \
    --ventas ventas.csv --asignacion salida/asignacion_nba.parquet --salida medicion/
python -m pytest -q tests
```

## Metodología

**1. Calidad (`nba/calidad.py`)** — columnas requeridas, completitud (nulos > 5% = ERROR, > 0% =
advertencia), duplicados exactos (se eliminan) y de llave con valores distintos (ERROR), dominios
(flags ∈ {0,1}, scores ∈ [0,1]), variables constantes, cobertura de analítica sobre clientes (≥ 95%),
integridad referencial de resultados, y coherencia del embudo (p. ej. abrió sin recibir, compró en
llamada sin llamada). Con algún ERROR el proceso se detiene y deja `reporte_calidad.csv`. Umbrales
configurables en `UmbralesCalidad`.

**2. Scores y pesos (`nba/scoring.py`)**
- `score_c = Σ w_k · x_k` con pesos de negocio normalizados.
- Con resultados: sobre los **contactados** (recibió mensaje o se gestionó por llamada) y con
  objetivo `compra = compro OR compro_llamada`, una regresión logística por campaña da la
  importancia observada de cada variable.
- **Credibilidad** `z = compras / (compras + 200)`: `w_final = z·w_datos + (1−z)·w_negocio`.
  Con pocos datos se respeta a negocio y con muchos mandan los resultados.
- **Calibración** `p = σ(a + b·score)`: el score pasa a ser una probabilidad de compra comparable
  entre campañas, condición necesaria para que el optimizador reparta bien.
- Se reporta el AUC en holdout (pesos de negocio frente a ajustados) y el embudo por campaña.
- Reglas derivadas: quien ya compró una campaña deja de ser elegible a ella; quien pidió no ser
  llamado queda marcado con `no_llamar` en la salida.

**3. Estrategia (`nba/estrategia.py`)** — por campaña: elegibles, elegibles exclusivos, cobertura
frente al mínimo, conversión techo y estado (OK / RIESGO / INFACTIBLE), además de la matriz de solapamiento.
La **factibilidad global** se comprueba con *max-flow* sobre patrones de elegibilidad: detecta
cuándo los mínimos no caben juntos aunque cada campaña tenga elegibles suficientes por separado.

**4. Optimización (`nba/optimizer.py`)** — un problema de transporte:
`max Σ valor_c·p_ic·x_ic` con una campaña por cliente y `mínimo_c ≤ Σ_i x_ic ≤ máximo_c`.
Se resuelve en el dual (subasta con *ε-scaling*: un precio sombra por campaña) y después un
refinamiento LP exacto (HiGHS) sobre los clientes casi empatados. Coincide con el óptimo LP exacto
en los tests y escala a 3M × 15 en ~35 s. El `precio_sombra` mide cuánta conversión cuesta
imponer el mínimo de esa campaña: es un insumo directo para negociar la estrategia.

**5. Experimentación (`nba/experimentacion.py`)**
- Grupo control por campaña con **aleatorización estratificada** (campaña × decil de score ×
  estratos extra), con totales exactos y una semilla reproducible. Admite varios brazos (A/B/n de mensajes).
- Modo `potencia`: control mínimo para detectar un lift relativo `mde` (por defecto 15%,
  α = 0.05, potencia = 0.8), con tope del 30% y respetando que el **tratamiento cubra el mínimo**.
  Por defecto el control se suma al mínimo antes de optimizar (`control_adicional_al_minimo`).
- Balance: SMD < 0.1 y chi² por estrato.
- `muestra_estratificada`: muestras para pilotos con margen de error, corrección finita y
  asignación proporcional o de Neyman.
- `medir_lift`: con resultados que traen `grupo`, calcula el lift absoluto y relativo, IC 95%, p-valor y
  ventas incrementales. Cierra el ciclo con el paso 2.

**6. Medición (`nba/medicion.py`, `nba/reporte_medicion.py`)**: mide la ola ya ejecutada.

| Insumo | Columnas |
|---|---|
| `programacion` | `campana`, `canal`, `fecha_inicio`, `fecha_fin`, `clientes_programados` [, `costo_unitario`] |
| `gestion` | `id_cliente`, `campana`, `canal`, `fecha`, `evento`. Eventos: `envio`, `entrega`, `apertura`, `click`, `llamada`, `contacto`, `no_contesta`, `no_llamar` |
| `ventas` | `id_cliente`, `producto`, `fecha_venta` [, `monto`, `canal_venta`] |
| `asignacion` (opcional) | salida del NBA con `grupo`: habilita el lift frente al control y el ROI |

- **Cumplimiento:** clientes programados frente a gestionados por campaña y canal, y % de gestiones dentro de la ventana.
- **Embudo:** entrega, apertura, CTR (click/entrega), CTOR (click/apertura), contactabilidad y % que pidió no llamar.
- **Atribución:** una venta del producto cuenta para la campaña si ocurre entre el primer toque y
  `fecha_fin + ventana` (30 días por defecto). El canal es el último toque de mayor intención antes
  de la venta: click o contacto, luego apertura o entrega, luego envío o llamada. El resto se clasifica como orgánica.
- **Impacto:** lift de conversión e ingresos incrementales frente al control (IC 95%) y ROI si hay `costo_unitario`.
- **Reporte:** `reporte_medicion.html` es autocontenido (se abre en el navegador o con `displayHTML` en Databricks).
  Trae hallazgos automáticos y gráficos con tooltip y tabla de datos: cumplimiento, embudo, ventas
  por día, canal, lift y tiempo a conversión.

## Salidas (`salida/`)

| Archivo | Contenido |
|---------|-----------|
| `asignacion_nba.parquet` | por cliente: `campana_nba`, `score_nba`, `alternativa_2/3`, `no_llamar`, `decil_score`, `estrato`, `grupo`, `contactar` |
| `reporte_calidad.csv` | todos los chequeos con su severidad |
| `pesos_ajustados.csv`, `calibracion_scores.csv`, `embudo_resultados.csv` | scoring |
| `evaluacion_estrategia.csv`, `solapamiento_campanas.csv` | disponibilidad |
| `resumen_asignacion.csv` | mínimo, asignados, contactados, conversiones esperadas y precio sombra |
| `diseno_experimental.csv`, `balance_experimental.csv` | experimento |
| `resumen.json` | factibilidad, comparativo frente al greedy y configuración usada |

## Supuestos a validar
- El histórico se usa como si el contacto hubiera sido aleatorio entre elegibles. Si las olas previas
  se seleccionaron por score, hay sesgo de selección: los grupos control de este diseño lo corrigen
  en las siguientes iteraciones.
- La conversión digital y la de llamada se tratan como una sola compra. Si el canal es una decisión,
  el modelo se extiende a pares (campaña, canal) con capacidad de call center.

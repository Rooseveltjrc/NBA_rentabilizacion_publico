# Data dummy: prueba de punta a punta

Se genera con `python generar_dummy.py` (50,000 clientes, 6 campañas, semilla 7). La misma
semilla produce siempre los mismos archivos. Opciones: `--clientes`, `--campanas`, `--seed`.

## entradas/ (lo que entregan negocio y analítica)

| Archivo | Filas | Columnas | Contenido |
|---|---|---|---|
| `clientes.csv` | 50,025 | `id_cliente` + 1 por campaña | 1 = cumple condiciones. Trae 25 duplicados exactos a propósito |
| `analitica.csv` | 50,000 | `id_cliente` + `<campaña>__<variable>` | 3 variables por campaña en [0,1]: `modelo_propension`, `var_recencia`, `var_afinidad`. 0.5% de nulos a propósito en una columna |
| `pesos.csv` | 18 | `campana`, `variable`, `peso` | Criterio de negocio: 0.5 / 0.3 / 0.2 |
| `estrategia.csv` | 6 | `campana`, `cantidad_minima` | Mínimos por campaña |
| `resultados.csv` | 12,389 | `id_cliente`, `campana` + 7 flags del embudo | Ola anterior (insumo 4). 50 filas con "abrió sin recibir" a propósito |

Campañas: `tarjeta_credito`, `credito_consumo`, `seguro_vida`, `cdt_inversion`, `cuenta_nomina`, `credito_vehiculo`.

Las variables tienen un poder predictivo real distinto del que supone negocio, para que el
ajuste de pesos tenga algo que descubrir. Los errores sembrados son leves: deben salir como
ADVERTENCIA, no como ERROR.

## salidas/ (lo que produce el NBA)

| Archivo | Para qué sirve |
|---|---|
| `asignacion_nba.csv` / `.parquet` | Una fila por cliente: `campana_nba`, `score_nba`, `alternativa_2/3`, `no_llamar`, `decil_score`, `estrato`, `grupo`, `contactar`. Es lo que se entrega a operación |
| `reporte_calidad.csv` | Cada chequeo de calidad con su severidad |
| `pesos_ajustados.csv` | Peso de negocio, peso observado en los datos, credibilidad y peso final |
| `calibracion_scores.csv` | Contactados, compras y AUC con pesos de negocio frente a pesos ajustados |
| `embudo_resultados.csv` | Entrega, apertura, contactabilidad y conversión por campaña |
| `evaluacion_estrategia.csv` | Elegibles, exclusivos, cobertura y estado por campaña |
| `solapamiento_campanas.csv` | % de elegibles compartidos entre campañas |
| `resumen_asignacion.csv` | Mínimo frente a contactados, conversiones esperadas y precio sombra |
| `diseno_experimental.csv` | Tamaño del control y lift mínimo detectable por campaña |
| `balance_experimental.csv` | Comparabilidad entre control y tratamiento |
| `resumen.json` | Factibilidad, mejora frente a la regla simple y parámetros usados |

## ronda_2/ (simulación del ciclo siguiente)

| Archivo | Contenido |
|---|---|
| `resultados_ronda_2.csv` | Resultados de ejecutar la asignación, con la columna `grupo`. El control no se contacta y compra de forma orgánica |
| `lift_por_campana.csv` | Lift medido (IC 95%, p-valor, ventas incrementales) junto al `lift_real_simulado` |

## medicion/ (M6: la ola ejecutada, con fechas)

| Archivo | Contenido |
|---|---|
| `programacion.csv` | 12 filas: cada campaña con su canal digital (email o sms) y llamada, ventanas de 14 días escalonadas desde el 02/11/2026 |
| `gestion.csv` | ~111 mil eventos (envío, entrega, apertura, click, llamada, contacto…) con fecha y hora |
| `ventas.csv` | ~9 mil ventas con fecha, monto y canal de venta |
| `cumplimiento.csv`, `embudo.csv`, `atribucion_canal.csv`, `serie_diaria.csv`, `lift.csv`, `roi.csv` | tablas del análisis |
| `reporte_medicion.html` | reporte visual: ábrelo en el navegador |

Sembrado a propósito: `credito_consumo` (sms) se ejecuta al 86% y `credito_vehiculo` (email) tiene 3%
de envíos fuera de ventana. Ambos deben aparecer en "Lo que hay que mirar primero".

En una operación real, `resultados_ronda_2.csv` vuelve como `resultados.csv` en la siguiente
corrida y cierra el ciclo.

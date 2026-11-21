# Inventario técnico de seis modelos de navegación mapless


## 1. Resumen y correspondencia exacta de los seis modelos

**No todos se definen en `models.py`.** Comparten el **punto de entrada de inferencia ROS 2 `RNN_mapless.py`**, siempre que se utilice la versión instalada que reconoce todas las arquitecturas, y las clases PyTorch se implementan en módulos diferentes. La ejecución Webots se coordina mediante `evaluate_navigation.py`, mientras que `benchmark_30.py` organiza suites, checkpoints y resultados; estos dos últimos **no son** la definición de las redes.

Directorio base de código: `~/GIT/mapless/src/Navigation/mapless_navigation/mapless_nav/scripts/`.

| ID del benchmark | Checkpoint en `modelos.7z` | Arquitectura real | Módulo/clase que define el modelo | Entrada de ocupación | Inferencia ROS 2 |
|---|---|---|---|---|---|
| `RNN_T1_factorial` | `Trail-01_mejor_factorial_lom_0-1_seed449153931/best_gap.pth` | `ros1_legacy` | `TorchModels/models.py` → `RNN` | 2 LOM temporales | `RNN_mapless.py` |
| `RNN_TN` | `Trail-02_mapless_ganador_TN_LOM_seed73746319/best_model.pth` | `ros1_legacy` | `TorchModels/models.py` → `RNN` | 2 LOM temporales | `RNN_mapless.py` |
| `RNN_T2_champion` | `champion_lstm_5x4/best_model_champion.pth` | `ros1_legacy` | `TorchModels/models.py` → `RNN` | 2 T-LOM temporales | `RNN_mapless.py` |
| `ConvLSTM_M1_T4` | `Trail-03_M1LSTMT4/M1_T4_rep02_seed3217914112_35ep_best_model.pth` | `m123_m1` | `TorchModels/m123_models.py` → `M123Policy` (variante M1) | LOM(t) + T-LOM(t), C=2 | `RNN_mapless.py` |
| `M1B_C8` | `Trail-04_M1B_c08_final/best_model.pth` | `m1b_mega` | `TorchModels/m1b_mega_models.py` → `M1BMegaPolicy` | Cuatro pares LOM/T-LOM, C=8 | `RNN_mapless.py` |
| `M1B_Depth_T4` | `M1B_Depth_T4D/best_model.pth` | `m1b_depth_direct_v2` | `TorchModels/m1b_depth_verified.py` → implementación Depth específica | Dos pares LOM/T-LOM, C=4 **+ Depth** | `RNN_mapless.py` + preparación `m1b_pair_live.py` |

**Precisiones sobre código:**

- **Sí**, el controlador ROS 2 de entrada se llama `RNN_mapless.py` para los seis; **no** significa que los seis ejecuten la misma clase ni el mismo preprocesamiento. El archivo del usuario contiene marcadores para `ros1_legacy`, `m123`, `m1b_mega` y `m1b_depth_direct_v2`.
- En experimentos anteriores se utilizó también `RNN_mapless_mega.py` como fuente de una actualización del controlador. Para reproducir resultados hay que congelar **la revisión efectiva de `RNN_mapless.py`**, no depender solamente del nombre del archivo.
- `m123_models.py` aporta también la celda `SpatialLSTMCell` reutilizada por M1_B. `m1b_pair_live.py` se ocupa del emparejamiento/preparación de observaciones para la variante Depth.
- `evaluate_navigation.py` gestiona Webots, RViz, rutas y mediciones, **no** contiene los pesos ni define las capas de las seis políticas. `benchmark_30.py` es un orquestador.
- Esta tabla se fundamenta en los `architecture`/`config` incorporados en los checkpoints, las formas de sus tensores y los módulos de código de las campañas, **no** solamente en los nombres elegidos para las carpetas.

## 2. Parámetros y estructura verificada de cada arquitectura

Los conteos de parámetros corresponden a la suma de elementos de los **tensores de pesos del checkpoint real**, inspeccionados con PyTorch; no son estimaciones a partir del nombre del modelo.

| Modelo | Parámetros en checkpoint | CNN / encoder | Recurrente | Cabezal de velocidades | Dropout | LR inicial | Época guardada |
|---|---:|---|---|---|---|---:|---:|
| RNN T1 factorial | **1,229,834** | Conv2d 2→2, kernel 3; MaxPool 7, stride 3 | RNN `tanh`, 4 capas, 350 unidades | 350→120→80→3 | denso 0.20 / 0.15 | 0.00015 | 7 |
| RNN TN | **1,229,834** | Igual | Igual | Igual | denso 0.20 / 0.05 | 0.00015 | 12 |
| RNN T2 champion | **1,229,834** | Igual | Igual | Igual | denso 0.30 / 0.15 | 0.00010 | 8 |
| ConvLSTM M1 T4 | **1,002,987** | Conv1 2→16, kernel 3, stride 2; fusión objetivo; Conv2 17→32; MaxPool 2 | 1 ConvLSTM espacial, 16 canales ocultos, BPTT=16 | Flatten(8,000)→120→80→3 | conv 0.20 / 0.15; head 0.20 / 0.15 | 0.00030 | 12 |
| M1_B C=8 | **1,003,851** | Conv1 8→16; fusión objetivo; Conv2 17→32; MaxPool 2 | 1 ConvLSTM espacial, 16 canales ocultos, BPTT=16 | Flatten(8,000)→120→80→3 | conv 0.20 / 0.15; head 0.20 / 0.15 | 0.00030 | 11 |
| M1_B + Depth T4 | **1,026,651** | Rama ocupación 4→16→32; rama Depth 2→16→32; fusión espacial | 1 ConvLSTM espacial, 16 canales ocultos, BPTT=16 | Flatten(8,000)→120→80→3 | conv 0.20 / 0.15; head 0.20 / 0.15 | 0.00030 | 7 |

**Detalles observables en los tensores:** las tres RNN tienen `conv1.weight=(2,2,3,3)` y cuatro bloques `rnn_cell_list`; la primera capa recurrente recibe **904 características**, coherente con el recorte 100×80 de estos checkpoints. M1 presenta `conv1=(16,2,3,3)`, `conv2=(32,17,3,3)` y puertas ConvLSTM `(64,48,3,3)`. M1_B cambia solamente la primera convolución a `(16,8,3,3)` respecto a ese perfil. Depth tiene `conv1=(16,4,3,3)`, además de `depth_conv1=(16,2,3,3)` y `depth_conv2=(32,16,3,3)`; sus puertas ConvLSTM son `(64,80,3,3)`, compatible con la fusión de características espaciales de dos ramas.

La salida de todas estas políticas es **tres velocidades normalizadas** `(v_x, v_y, ω)` mediante una capa final de tamaño 3 y activación `tanh`. Los metadatos de los seis checkpoints indican `vmax=[0.5,0.5,1.0]` cuando está presente o bajo el preprocesamiento correspondiente. Se requiere la **decodificación propia de cada checkpoint** antes de publicar comandos; los pesos de una familia no se pueden cargar en otra.

### 2.1. RNN heredadas: tres checkpoints, misma arquitectura

Son redes recurrentes simples `tanh` —**no** una LSTM aunque una carpeta se llame `champion_lstm_5x4`— con cuatro capas de 350 unidades. La red aplica Conv2d a un bloque de **dos observaciones** y entrega a las celdas recurrentes las características resultantes junto con el objetivo. Los parámetros entrenados del MLP son 350→120→80→3. El campo `fc` del checkpoint se conserva por compatibilidad, aunque no se utiliza en el `forward` vigente.

Sus diferencias reales son **representación (LOM/T-LOM), datos de entrenamiento, semillas, dropout y learning rate**, no tamaño del estado oculto. El contenido de `champion_lstm_5x4/README_champion.txt` declara «LSTM, 3 capas, 256 unidades», pero **contradice tanto `config.json` como la forma de las matrices de pesos**. Para reproducir inferencia se deben usar `ros1_legacy`, cuatro capas y 350 unidades; el texto del README es una etiqueta histórica no autorizada para reconstruir la red.

### 2.2. ConvLSTM M1 T4

Por cada paso recurrente recibe **LOM(t) y T-LOM(t)**, un par de imágenes alineadas, no dos observaciones de instantes distintos. Conv1 crea 16 características, se incorpora la meta mediante un canal mixto y Conv2 produce 32 características. La ConvLSTM conserva mapas espaciales `h` y `c` con 16 canales. La salida proviene del cabezal 120→80→3. No utiliza un mapa global ni memoria externa de obstáculos.

### 2.3. M1_B C=8

El encoder recibe **cuatro pares cronológicos**, ocho canales por paso recurrente; la ConvLSTM sigue teniendo 16 canales ocultos y BPTT 16. La cantidad de canales C describe la **ventana explícita de ocupación**, mientras que BPTT describe el alcance del gradiente a través de los pasos de entrenamiento. Son mecanismos diferentes. Al reiniciar una trayectoria, la configuración indica `oldest_to_current_replicate_earliest_on_reset`: se replica la primera observación disponible para completar el historial causal, sin usar frames futuros.

### 2.4. M1_B + Depth T4

La configuración declara **dos pares de LOM/T-LOM (C=4)** y una **rama adicional Depth de dos canales**; no es equivalente a la versión `c08` de ocho canales. Los pesos incluyen una rama convolucional específica de profundidad y una ConvLSTM que fusiona las características. Los pesos indican dos canales Depth de entrada, pero no documentan por sí solos si ambos representan dos instantes diferentes; eso exige comprobar el preprocesador. El objetivo se fusiona en filas mixtas después de Conv1, no como cuatro canales adicionales de ocupación.

La representación Depth del entrenamiento figura como `16UC1_PNG_normalized_near_high`, con rango máximo **3.5 m**. En inferencia es esencial verificar la conversión desde la imagen ROS, su sincronización temporal con las cuadrículas y la orientación. Este modelo no resuelve automáticamente puntos ciegos cercanos que tampoco aparezcan en la cámara.

## 3. Entradas LOM/T-LOM, temporalidad y objetivo

Todas las configuraciones señalan la variante **5×4 m**, con área de ocupación **100×80 celdas**. En los códigos auditados se agregan **dos filas de 80 columnas** para codificar la meta relativa `(d,θ)`, por lo que la matriz preparada suele medir **102×80** por canal/observación. Es importante no confundir las dos filas de objetivo con la extensión física del mapa.

| Modelo | Canales / secuencia de entrada | LOM utilizada | T-LOM utilizada | Información adicional |
|---|---|---|---|---|
| RNN T1 | `[LOM(t−1), LOM(t)]`, 2 observaciones | Sí | No | Objetivo `(d,θ)` |
| RNN TN | `[LOM(t−1), LOM(t)]`, 2 observaciones | Sí | No | Objetivo `(d,θ)` |
| RNN T2 | `[T-LOM(t−1), T-LOM(t)]`, 2 observaciones | No | Sí | Objetivo `(d,θ)` |
| M1 T4 | `[LOM(t), T-LOM(t)]`, C=2 | Sí | Sí | Objetivo como canal mixto espacial |
| M1_B C08 | `[LOM(t−3),TLOM(t−3),...,LOM(t),TLOM(t)]`, C=8 | Sí | Sí | Objetivo actual; historial de 4 pares |
| M1_B Depth | `[LOM(t−1),TLOM(t−1),LOM(t),TLOM(t)]`, C=4; **Depth** en otra rama | Sí | Sí | Rama Depth de 2 canales según pesos; correspondencia temporal exacta por verificar, objetivo actual |

**Definiciones operativas:**

- **LOM:** cuadrícula de ocupación local en el marco del robot; no es un mapa global almacenado. Las RNN T1/TN emplean LOM, mientras que M1 y las variantes M1_B mezclan LOM con T-LOM.
- **T-LOM:** representación local transformada/umbralizada derivada del flujo de ocupación para entrenamiento. El controlador consultado muestra una conversión de la señal publicada de ocupación a una codificación binaria antes de normalizar, de acuerdo con la configuración del modelo. **No** deben intercambiarse LOM y T-LOM simplemente porque tengan la misma forma.
- **Escala:** los seis checkpoints normalizan ocupación (`normalize_occupancy=true`); la entrada de entrenamiento de ocupación queda en **0–1**. Para T-LOM el controlador auditado convierte valores publicados `0/50/100` al binario `0/100` mediante umbral ≥50 y luego normaliza a 0–1. El origen y la rotación de las celdas deben conservarse igual que durante entrenamiento.
- **Objetivo:** distancia y ángulo **relativos** al robot. RNN histórica conserva filas de objetivo dentro de cada observación; M1/M1_B extraen el objetivo actual y lo incorporan como **mapa mixto en dos mitades horizontales** de características antes de Conv2. No entra pose global, mapa de planificación ni ruta de referencia a la política.
- **Estado temporal:** para M1/M1_B, la ConvLSTM conserva `(h,c)` internamente durante la trayectoria; debe reiniciarse al comenzar un nuevo episodio. Para las RNN se requiere respetar el manejo de ventanas de dos observaciones y cualquier estado que mantenga la versión efectiva del controlador. La generación de las cuadrículas LOM/T-LOM no debe modificarse para el benchmark.
- **Control:** las tres salidas son comandos de velocidad objetivo; no deben describirse como mediciones físicas de odometría.

## 4. Datasets de entrenamiento y escenarios

### 4.1. Composición de conjuntos

| Conjunto identificado | Escenarios documentados | Cantidad | Evidencia disponible y limitación |
|---|---|---:|---|
| **T1 histórico** | **E01, E02, E03, E12, E17, E22, E27, E30** | 8 | Selección histórica T1 documentada en la conversación del proyecto; el checkpoint `RNN_T1_factorial` guarda `environment_set=T1`, **no la lista explícita**. Asociar esos ocho escenarios al checkpoint requiere conservar el manifiesto `bad1c548…` para verificación final. |
| **TN** | **No recuperables de los archivos archivados** | No documentado | `RNN_TN` solo identifica `environment_set=TN`; la campaña exploró selecciones de escenarios candidatas, pero **no hay listado final en el checkpoint**. No sustituirlo por T1 ni por T2. |
| **T2 histórico** | **E01, E02, E03, E07, E09, E11, E12, E13, E14, E15, E17, E18, E21, E22, E25, E26, E28, E30** | 18 | Reconstruido a partir de la columna «T2 histórico: Sí» del reporte histórico de 30 escenarios. La configuración `RNN_T2_champion` guarda solo `environment_set=T2`, sin lista; **la correspondencia exacta con este checkpoint debe contrastarse con su manifiesto de pares** `3061f22b…`. |
| **T4 / D1_T4_equal** | **E02, E03, E04, E08, E13, E17, E18, E21, E25, E30** | 10 | Listados explícitos en `config` de M1 y M1_B; plan con **600 episodios de entrenamiento**, 60 por escenario, y **461 de validación** en total. |
| **T4-D Depth** | **E02, E03, E04, E08, E13, E17, E18, E21, E25, E30** | 10 | Lista explícita en configuración Depth; **150 episodios de entrenamiento**, **30 de validación**, plan de pares `paired_T4_150ep…`. No confundir con los 600 episodios de M1/M1_B. |

**Nota de procedencia:** las composiciones de T1 y T2 recuperan definiciones **históricas** disponibles en conversaciones y reportes anteriores, no datos incorporados en los respectivos `config.json`. La lista exacta de **TN está ausente**. Sin los manifiestos originales de episodios/pares y sus hashes no es científicamente válido certificar la pertenencia de cada archivo NPZ a esos tres entrenamientos.

### 4.2. Cantidad de datos por checkpoint

| Modelo | Conjunto declarado | Presupuesto de entrenamiento informado | Validación informada | Semilla |
|---|---|---:|---:|---:|
| RNN T1 factorial | T1 | 10,000 **secuencias/pares** | 4,000 secuencias | `449153931` |
| RNN TN | TN | 3,000 **secuencias/pares** | 4,000 secuencias | `73746319` |
| RNN T2 champion | T2 | 10,000 **secuencias/pares** | 4,000 secuencias | `325487` |
| ConvLSTM M1 T4 | D1/T4 | 600 episodios | 461 episodios | `3217914112` |
| M1_B C08 | D1/T4 | 600 episodios | 461 episodios | `3655314999` |
| M1_B + Depth T4 | T4-D | 150 episodios | 30 episodios | `504327592` |

**No comparar directamente las magnitudes:** en las RNN se contabilizan secuencias supervisadas/pares de observaciones; en las ConvLSTM se informan episodios cronológicos y fragmentos de BPTT. En M1 y M1_B los metadatos también indican **1,818 chunks de entrenamiento** y **1,403 de validación**. La pertenencia a T4 no implica que Depth usó los mismos ejemplos que M1: su plan de 150/30 es diferente.

### 4.3. Proporciones `straight` / `semi_complex` / `complex` / `v_complex`

**Punto crítico de trazabilidad:** ninguno de los seis `config.json` incorporados en `modelos.7z` registra los **conteos finales de clases utilizados por ese checkpoint**. Los campos `train_sequences`, `train_episodes` y `train_scenarios` no equivalen a una distribución por clase. Por eso la proporción **real de entrenamiento por checkpoint no se puede verificar a partir del archivo entregado**.

| Diseño / fuente histórica | Straight | Semi-complex | Complex | V-complex | Qué representa |
|---|---:|---:|---:|---:|---|
| Esquema histórico `moderate_hard` de búsqueda RNN | 20 % | 20 % | 30 % | 30 % | **Cuota de selección propuesta/utilizada en determinadas pruebas**, pero **sin asignación demostrada** a los tres checkpoints RNN archivados. |
| Esquema histórico alternativo RNN | 10 % | 20 % | 30 % | 40 % | Otra proporción ensayada históricamente; **no se puede afirmar que sea la del ganador concreto** sin su manifiesto. |
| Pools de rutas balanceadas T4/T5 | 25 % | 25 % | 25 % | 25 % | Cuota del **pool generador** de rutas, frecuentemente 50/50/50/50 al generar 200 metas; **no demuestra** que los 600 episodios seleccionados de M1/M1_B conservaran exactamente esa mezcla. |
| Plan de entrenamiento T4-D con Depth | No documentado | No documentado | No documentado | No documentado | El plan `paired_T4_150ep…` no está incluido en `modelos.7z`; se necesita auditarlo para conocer la mezcla real. |
| **Benchmark nuevo E01–E30** | **25 %** | **25 %** | **25 %** | **25 %** | **Diseño de evaluación**, no entrenamiento: 3 goals de cada clase por escenario, 90 metas por clase y 360 por modelo. |

Para reportar porcentajes **efectivamente observados** en entrenamiento deben recuperarse y contar sus etiquetas del manifiesto original (`pair_manifest_sha256` para T1/TN/T2, `episode_plan_sha256` para M1/M1_B, `training_plan_sha256` para Depth). Solo entonces podremos escribir una fila por checkpoint con numerador y denominador auditables.

## 5. Resultados conocidos de las pruebas de los seis checkpoints

**Advertencia metodológica:** los resultados provienen de **campañas diferentes**, con tamaños de suites distintos, distintas semillas y mapas. **No son un ranking comparativo válido de los seis**. El benchmark de 30 mapas se propuso precisamente para eliminar esa diferencia de condiciones.

| Modelo archivado | Resultado documentado | MSE de validación del checkpoint | Contexto y límites |
|---|---|---:|---|
| **RNN T1 factorial** | Comparación histórica denominada «factorial T1»: **11/50 llegadas aceptables** y **32/50 colisiones** | **0.0135808** | Prueba pareada frente a T3 refinado en 50 rutas. Es un resultado asociado al diseño histórico T1; **no hay en el archivo aportado un manifiesto que vincule inequívocamente las 50 pruebas a este mismo SHA del `best_gap.pth`**. |
| **RNN TN** | Comparación histórica en E20/E08: valor de puntuación comunicado **0.7519** frente a **0.6562** para T1+LOM | **0.0146076** | No se especifica el denominador ni se demuestra que la puntuación sea «tasa de éxito». **No se dispone de conteos de llegadas/colisiones trazables a este SHA** en el archivo. |
| **RNN T2 champion** | README histórico: **8 llegadas y 0 colisiones**, puntuación global **0.881963** | **0.0140757** | El README **no precisa denominador ni tareas**; no se convierte «8 llegadas» en «8/8» sin verificar ensayos. Además su descripción de arquitectura es inconsistente con los pesos. |
| **ConvLSTM M1 T4** | **9/12 éxitos (75.0 %)** y **3/12 colisiones (25.0 %)** | **0.0094653** | Desarrollo E02, E08 y E21, cuatro goals por mapa; corresponde a M1/D1/T4 réplica 2, semilla 3217914112. |
| **M1_B C=8** | **78/144 éxitos (54.2 %)** y **54/144 colisiones (37.5 %)** para el checkpoint elegido; **18/72 éxitos difíciles** | **0.0093676** | Confirmación de desarrollo en E02, E08 y E21, 48 rutas por mapa, semilla 3655314999. En conjunto, **las dos semillas C08** sumaron **157/288 éxitos**, **111/288 colisiones** y **34/144 éxitos difíciles**; no confundir agregado con un único checkpoint. |
| **M1_B + Depth T4** | **137/288 éxitos (47.57 %)**, **109/288 colisiones**, **41 falsas llegadas** y **1 timeout** | **0.01538075** | Resultado del archivo `M1B_Depth_T4D/results.md`, comparación de seis mapas E02,E18,E25 (vistos) y E12,E20,E28 (no vistos), 48 metas por mapa; el `config.json` aún contiene `navigation_evaluated=false`, posiblemente metadato anterior a la prueba. Se reporta como evidencia documental que debe ligarse al log de evaluación. |

**La MSE no debe compararse entre familias como métrica de navegación**, dado que los conjuntos, las distribuciones de entrenamiento/validación, las etiquetas y los objetivos experimentales son distintos.

### 5.1. Comparación anterior específica de M1_B: base frente a C08

Para contextualizar por qué se archivó la variante C=8, la campaña de refinamiento informó:

| Fase | Base (C=4) | C08 (C=8) | Lectura |
|---|---|---|---|
| `promote` (4 semillas, 288 pruebas/modelo) | **179/288 éxitos**, 102 colisiones, 46/144 difíciles | 172/288 éxitos, 107 colisiones, 41/144 difíciles | Base fue mejor en esa etapa. |
| `final` (2 semillas reservadas, 288 pruebas/modelo) | **157/288 éxitos**, 118 colisiones, 28/144 difíciles | **157/288 éxitos**, 111 colisiones, 34/144 difíciles | Éxito global empatado; C08 tuvo 7 colisiones menos y 6 éxitos difíciles más, **pero más fallos sin colisión**. |

El checkpoint archivado `M1B_C8` es **una réplica individual elegida** de aquella campaña; no representa por sí solo las dos semillas agregadas.

### 5.2. Advertencias sobre resultados históricos

- El benchmark histórico de 30 escenarios que comparó un **Teacher** con una **«LSTM histórica»** (639/1,200 éxitos LSTM informados) **no identifica automáticamente ninguno de estos seis checkpoints**. No se asigna ese total al RNN T2 `champion`.
- Las cifras de «llegada aceptable», «puntuación global», «éxito» y «colisión» se conservaron con su denominación original; sin los logs no son intercambiables.
- Las 50 rutas del factorial T1 y las 288 rutas de Depth pertenecen a campañas distintas; no pueden compararse como porcentajes de una única prueba común.
- Los resultados de navegación de 30 escenarios × 12 goals × seis modelos **siguen pendientes de ejecución**; ningún número anterior corresponde a ese benchmark nuevo.

## 6. Benchmark común propuesto: E01–E30

El experimento solicitado consiste en **30 escenarios × cuatro categorías × tres goals/categoría = 360 rutas diferentes por modelo**. Se ejecutan las mismas 360 rutas en cada uno de los seis checkpoints: **2,160 ensayos Webots/RViz**.

| Parámetro común | Valor |
|---|---|
| Escenarios | E01–E30 |
| Metas por escenario | 12 |
| Distribución | 3 straight + 3 semi_complex + 3 complex + 3 v_complex |
| Metas distintas por modelo | 360 |
| Sesiones totales de navegación | 2,160 |
| Variables comunes | Mundo, posición inicial, orientación inicial, objetivo, criterios de éxito y contacto, tiempo límite y semilla de simulación |
| Variables específicas | Representación y cantidad de LOM/T-LOM, histórico temporal, sensor Depth, arquitectura y preprocesamiento de cada checkpoint |
| Controlador | `RNN_mapless.py` con despacho de arquitectura |
| Evaluador | `evaluate_navigation.py`, RViz obligatorio |
| Orquestador | `benchmark_30.py` |

La prueba incluirá escenarios que fueron usados para entrenar **alguno** de los modelos y otros que no. Para T4 sabemos exactamente cuáles se utilizaron; para T1/TN/T2 la pertenencia precisa debe quedar «no verificada» hasta que se recuperen sus manifiestos. El conjunto previamente denominado `EVAL_HOLDOUT` se abre al ejecutar esta comparación; ello se debe documentar y **no** reutilizarlo posteriormente para seguir ajustando estos mismos modelos como si continuara reservado.

## 7. Fuentes auditadas y verificaciones pendientes

**Archivos del usuario inspeccionados:**

1. `modelos.7z`: seis checkpoints `.pth`, los `config.json`, `Trail-03.../reporte_M123_desarrollo_35ep.md`, `Trail-04.../reporte.md`, `M1B_Depth_T4D/results.md` y `champion_lstm_5x4/README_champion.txt`.
2. Código aportado: `models(7).py` (clase `RNN` y `RNNCell`); `RNN_mapless(5).py`; implementaciones `m123_models.py` y `m1b_mega_models.py` incluidas en el paquete previamente preparado para la mega prueba.
3. Reportes del proyecto: `01_reporte_final_benchmark(1).md` (columna histórica T2 y resultados antiguos Teacher/LSTM), `01_T4_T5_acuerdos.md`, `REPORTE_MEGA_PRUEBA_M1B_2026-11-21.md`, `PROTOCOLO_BENCHMARK_30_ESCENARIOS.md`.
4. La salida que compartió el investigador confirma que, en su laptop, están presentes `RNN_mapless.py`, `evaluate_navigation.py`, `TorchModels/models.py`, `TorchModels/m123_models.py`, `TorchModels/m1b_mega_models.py`, `TorchModels/m1b_depth_verified.py` y `m1b_pair_live.py`.

**Pendientes para cerrar la trazabilidad experimental:** obtener los manifiestos de pares T1/TN/T2 (listas exactas y clases); auditar la distribución real de clases de los 600 episodios T4 y los 150 de T4-D; congelar hashes de código de inferencia y preprocesamiento Depth de la laptop; comprobar cada checkpoint con carga `strict=True`; y ejecutar el benchmark común. No se reemplaza una ausencia de esos archivos por una suposición.

---

**Conclusión:** existen **tres checkpoints de RNN clásica con una misma definición de red** y **tres checkpoints ConvLSTM de arquitecturas distintas** (M1, M1_B C08 y M1_B+Depth). Pueden compartir **el mismo nodo de inferencia `RNN_mapless.py`**, pero **no** la misma implementación PyTorch, entrada ni distribución de datos. La información verificable de entrenamiento y navegación se ha consignado por checkpoint, dejando explícitas las incertidumbres de T1/TN/T2 y de las proporciones de clase no incluidas en los archivos originales.
# Recolección de datasets con rutas precomputadas

Este directorio contiene dos orquestadores de recolección de datasets basados en pools de rutas precomputadas:

- `collect_dataset_batch_classic_nav.sh` — recolección **automatizada** con el sistema Classic Navigation (el robot navega solo).
- `collect_dataset_teleop.sh` — recolección **teleoperada** (un operador humano conduce el robot con joystick por cada ruta del pool).

Ambos scripts llaman a `path_generator.py`, validan cuotas y archivos de poses, publican el pool activo en `data/<escenario>/path_pool*.json` y lanzan ROS 2/Webots. No se necesitan scripts auxiliares de orquestación.

`data_analysis/generate_path_pool.py` se usa únicamente en modo `--incremental` de `collect_dataset_teleop.sh`; el script automatizado invoca siempre `path_generator.py` directamente.

## Prerrequisitos

- ROS 2 (Humble o superior) y Webots instalados y configurados.
- Entorno de trabajo compilado: `colcon build` y `source install/setup.bash`.
- Dependencias de Python: `numpy`, `Pillow`, `PyYAML`, `scipy` (obligatorias para `path_generator.py` y `teleop_dataset_recorder.py`); `matplotlib` (opcional, para generar los PNGs de rutas); `pot` (opcional, solo para el análisis de similitud de la sección final).
- El paquete `debugger` debe estar en el workspace. `teleop_dataset_recorder.py` importa `from debugger.utils import DebugClient`; si falta, el nodo falla al arrancar con `ModuleNotFoundError: No module named 'debugger'`.
- Los mapas de los escenarios deben existir en `webots_env/environments/maps/<escenario>_*/` con el YAML de prohibición (`map_<folder>_prohibition.yaml`) y su PNG asociado. El YAML también debe contener `world_offset_x`, `world_offset_y`, `world_offset_yaw` cuando `auto_world_offset=true` (default del launch).

---

## `collect_dataset_batch_classic_nav.sh` — recolección automatizada

### Prueba rápida

```bash
source install/setup.bash
bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
  --scenarios "E03" \
  --n-goals 4 \
  --seed-base 2026101400 \
  --timeout-seconds 900 \
  --max-attempts 25000 \
  --stall-timeout 300 \
  --goal-position-tolerance-m 0.40 \
  --log-dir /tmp/mapless_path_smoke/logs
```

El total de rutas se distribuye de forma equilibrada entre las clases `straight`, `semi_complex`, `complex` y `v_complex`. Para fijar cuotas por clase se usa `--st_sem_cmp_vc 10_10_15_20` en lugar de `--n-goals`.

### Parámetros de generación

Las cotas geométricas y el ajuste del planificador también se controlan desde argumentos. Consulta `--help` para la lista completa. Algunos ejemplos:

```bash
bash collect_dataset_batch_classic_nav.sh --help
bash collect_dataset_batch_classic_nav.sh --scenarios "E03" --n-goals 4 \
  --hard-clearance-m 0.35 --min-goal-distance-m 2.0 \
  --max-goal-distance-m 3.0 --max-path-length-m 3.0 \
  --v-max-goal-distance-m 3.6 --v-max-path-length-m 4.2 \
  --v-goal-candidates 10 --v-plan-candidates 3
```

Los argumentos después de `--` se pasan a `ros2 launch`. Por ejemplo, para sobrescribir un argumento de launch: `-- world_offset_x:=0.0`.

`--timeout-seconds` y `--max-attempts` limitan la generación, no la duración de la grabación. `--stall-timeout` se aplica a la fase de grabación.

### Estabilización y control en pasajes estrechos (`v_complex`)

Para evitar colisiones contra paredes en giros cerrados y pasillos angostos, el nodo de control `simple_move` admite parámetros de ajuste fino que pueden modificarse en `navigation_controller/launch/controller.launch.xml` o inyectarse al final del script de recolección usando el separador `--`:

```bash
bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
  --scenarios "E03" \
  --n-goals 8 \
  --seed-base 2026101499 \
  --goal-position-tolerance-m 0.40 \
  --log-dir /tmp/mapless_path_tolerance_fix \
  -- -p simple_move.path_lookahead_m:=0.15 \
     -p simple_move.control_beta:=0.15 \
     -p simple_move.max_linear_speed:=0.20 \
     -p simple_move.turn_speed_factor_near:=0.25
```

---

## `collect_dataset_teleop.sh` — recolección teleoperada

Graba episodios donde **un operador humano** conduce el robot por cada ruta del pool. Útil para generar datasets con la etiqueta `teacher=human` y comparar contra los generados por `classic_nav`.

### Flujo

1. **Fase 1 — Preparación del pool**. El script filtra las rutas cuyo `start`/`goal` ya aparecen en sesiones previas (`episode_metadata.jsonl`, `failed_paths.json`, y los `data.csv` de cada `run_*`). El conteo en CSV se hace por **transición de `episode_id`**, no por `is_episode_start`, para no verse afectado por la race del recorder C++. Si tras filtrar faltan rutas para cubrir las cuotas, invoca `path_generator.py` para regenerar un pool con cuotas mayores.
2. **Fase 2 — Recolección**. Abre Webots + RViz y espera. Cada pulsación de **B** asigna la siguiente ruta del pool al operador. El operador conduce hasta el `start` de la ruta y luego hasta el `goal` para completar el episodio.
3. **Cierre**. Cuando se cumple el número de goals pedido (o se agotan los paths del pool), el recorder C++ cierra el launch solo y el script imprime `teleop_complete`.

### Controles

- **B** — Asignar ruta / pausar grabación. Si se pulsa durante `WAITING_FOR_START` (antes de llegar al start), la ruta se devuelve al pool.
- **X** — Teleport al `start` de la ruta actual con yaw aleatorio y reinicio del intento. Si se pulsa durante `RECORDING`, el episodio actual se descarta (el buffer no se guarda en `data.csv`), pero la ruta se mantiene activa: al llegar de nuevo al start se reintenta el mismo path. Si no hay ruta asignada, teleport a sala inexplorada.
- **Y** — Saltar la ruta actual y asignar la siguiente. La ruta saltada se añade a `failed_paths.json` con `last_status: skipped_by_operator`. **Este es el único mecanismo de descarte manual.**

### Comportamiento ante colisión

En teleop, la colisión **no descarta la ruta**. Al detectarse `/navigation/collision=true` durante un episodio (`STATE_RECORDING`):

1. El episodio se interrumpe inmediatamente (el buffer se descarta, no se guarda en `data.csv`).
2. El robot se teleporta automáticamente al `start` de la ruta actual con yaw aleatorio.
3. El path permanece activo (`current_start`, `current_goal` sin cambios).
4. Cuando el robot llegue de nuevo al start, comienza el intento N+1 del mismo path.

Durante 1 segundo tras el teleport se ignoran pulsos residuales de colisión (`_collision_grace_until`) para evitar abortar el intento siguiente antes de empezar. Si el robot sigue en colisión al llegar al start, el episodio no inicia y el operador debe pulsar X o Y.

### Reintentos

Por defecto `path_max_attempts=0`, lo que significa **reintentos ilimitados**. El path solo se abandona al llegar al goal o al pulsar Y. Para habilitar el descarte automático tras N intentos, ajusta `path_max_attempts` a un valor positivo en el launch; en ese caso, tras N fallos consecutivos, la ruta pasa a `failed_paths.json` con `last_status: teleport_after_collision`.

### Bloqueo del joystick izquierdo (cabeza)

El launch `record_single_world_teleop.launch.xml` tiene `lock_head:=true` por defecto. Con esa configuración, los comandos del joystick izquierdo son ignorados: la cabeza permanece fija en `head_pan_offset` / `head_tilt_offset`. Esto evita que movimientos accidentales del joystick cambien la cámara y contaminen los datos. El joystick derecho (base) sigue funcionando con normalidad. Para desactivar el bloqueo:

```bash
ros2 launch mapless_dataset record_single_world_teleop.launch.xml \
    scenario:=E01 world:=E01 lock_head:=false
```

O desde el orquestador: `bash collect_dataset_teleop.sh ... --unlock-head`.

### Prueba rápida

```bash
source install/setup.bash
bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_teleop.sh \
  --scenarios "E01" \
  --n-goals 4
```

Al pulsar **B**, el script asigna la primera ruta; el operador conduce al start y al goal. Al completar los 4 goals el launch cierra solo.

### Cuotas por clase

Igual que el modo automatizado:

```bash
--n-goals 10             # balanceado: 3, 3, 2, 2
--st_sem_cmp_vc 3_3_2_2  # cuotas exactas: straight=3, semi=3, complex=2, v_complex=2
```

### Preparar pools sin lanzar la simulación (`--pool-only`)

`--pool-only` ejecuta solo la Fase 1 y sale sin abrir Webots. Útil para pre-generar pools de varios escenarios antes de empezar a grabar:

```bash
bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_teleop.sh \
  --scenarios "E01 E02 E03 E07 E09 E11 E12 E13 E14 E15 E17 E18 E21 E22 E25 E26 E27 E28 E30" \
  --n-goals 100 \
  --max-attempts 50000 \
  --timeout-seconds 2400 \
  --pool-only \
  --log-dir src/Navigation/mapless_navigation/dataset/data/logs/pools_all
```

Cada escenario produce un pool de ~204 rutas (2× `--n-goals` por el buffer `quota + deficit + 1`). Después se graban con `--incremental` (o `--record-only` si ya hay pool). El script retorna `exit 0` si todos los escenarios completaron, `exit 1` si alguno quedó corto o falló. No requiere `ros2 pkg prefix` — el chequeo del paquete `mapless_dataset` se omite en este modo.

### Modo `--incremental`

Reanuda desde un pool existente sin regenerar lo que ya está grabado:

```bash
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 100 --incremental
```

Internamente llama a `data_analysis/generate_path_pool.py --incremental --run-dir-glob 'run_*'`, que:

1. Carga `path_pool_history.json` (o `path_pool.json` si no hay history).
2. Detecta `committed_ids` vía `path_id` exacto desde `episode_metadata.jsonl` + fallback CSV con tolerancia 0.30 m.
3. Detecta `failed_ids` desde `failed_paths.json`.
4. Calcula `live_ids = pool_ids - committed_ids - failed_ids`.
5. Si `live_ids < target`, pide al generador `need_margin = int(need * 1.2) + 5` rutas nuevas.
6. Deduplica contra el history por pares `start`/`goal` en ambas direcciones.
7. Filtra y escribe `path_pool.json` (vista live) + `incremental_state.json` (contadores).

Requiere que `data/<escenario>/path_pool.json` exista. Si no existe, el shell cae automáticamente a generación fresca. Compatible con `--timeout-seconds` y `--max-attempts` (se pasan al generador). Incompatible con `--purge-data` y `--st_sem_cmp_vc` (el wrapper rechaza la combinación porque `_build_filtered_quota` no soporta cuotas explícitas).

### Solo grabar con un pool existente (`--record-only`)

```bash
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 10 --record-only
```

`--record-only` **omite la Fase 1** del shell (no filtra ni regenera el pool en disco). Sin embargo, `teleop_dataset_recorder.py` **sí sigue filtrando en tiempo de carga**: descarta los paths cuyos `start`/`goal` coincidan con `episode_metadata.jsonl` o `failed_paths.json` (tolerancia 0.30 m). Esto es intencional y es lo que permite grabar en chunks sin repetir rutas.

#### `--n-goals N` con `--record-only` significa "N nuevas"

En `--record-only`, el sampler graba `N` episodios **nuevos en esta invocación**, no `N` acumulados. Ejemplo para llegar a 100 totales:

```bash
# Chunk 1 — 25 nuevos (acumulado: 25)
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 25 --record-only

# Chunk 2 — 25 nuevos (acumulado: 50)
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 25 --record-only

# Chunk 3 — 25 nuevos (acumulado: 75)
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 25 --record-only

# Chunk 4 — 25 nuevos (acumulado: 100)
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 25 --record-only
```

O en dos pasos: `--n-goals 50 --record-only` dos veces. Ambos producen 100 totales, con el reparto por clase que decidas.

**Importante**: si usas `--n-goals 100 --record-only` dos veces, **acabarás con ~200 episodios**, no 100. Porque cada invocación graba 100 adicionales (hasta agotar el pool de 204). El debugger muestra `total_saved=N` (acumulado) y `CLASSES straight=0/25...` (de esta sesión), lo que puede confundir. Usa siempre la suma de N = total objetivo.

El pool tiene ~204 rutas (2× el `--n-goals` con el que se generó, por el buffer `quota + deficit + 1`). Con 100 totales de 204 disponibles, sobra margen para reintentos y descartes.

### Filtrado de rutas ya usadas

Por defecto `collect_dataset_teleop.sh` **no repite** rutas entre sesiones. Hay dos niveles de filtro:

**Nivel 1 — Fase 1 del shell** (`filter_pool()` en el heredoc Python de `collect_dataset_teleop.sh`):

- `episode_metadata.jsonl` — un registro por goal exitoso (con `path_id`, `class`, `start`, `goal`, `attempt`).
- `failed_paths.json` — rutas descartadas (reintentos agotados o saltadas).
- `data.csv` de cada `run_*` — recoge también endpoints de sesiones no-teleop (classic_nav, nav2). Cuenta por **transición de `episode_id`**, no por `is_episode_start`.

Las rutas cuyo par `start`/`goal` coincide (dentro de `--endpoint-tolerance-m`, default 0.30 m) con alguna entrada previa se descartan. Si tras el filtro hay déficits, se regenera un pool nuevo con cuotas mayores (`quota + deficit + 1` por clase).

**Nivel 2 — Carga del pool en el recorder** (`_load_used_endpoints` + `_endpoint_matches` en `teleop_dataset_recorder.py`):

Este filtro corre **siempre**, incluso con `--record-only`. Es el que garantiza que grabar en chunks no repite start/goal. Combina `episode_metadata.jsonl` + `failed_paths.json` con tolerancia 0.30 m en ambas direcciones (mismo sentido e invertido start↔goal).

**En modo `--incremental`** se usa además `_match_committed_ids` de `data_analysis/generate_path_pool.py`, que prefiere el match exacto por `path_id` desde `episode_metadata.jsonl`, con fallback a CSV por endpoint a 0.30 m.

No existe un flag para desactivar el filtrado de nivel 2. Para forzar la grabación de un path previamente usado, hay que borrar manualmente la línea correspondiente en `episode_metadata.jsonl` o en `failed_paths.json`.

### Parámetros exclusivos del modo teleop

| Flag | Descripción |
|---|---|
| `--record-only` | Omite la Fase 1 (no filtra ni regenera el pool en disco). El recorder sigue excluyendo paths ya usados al cargar |
| `--incremental` | Carga el pool existente, filtra rutas ya grabadas/descartadas vía `path_id`, genera solo el déficit. Requiere `data/<escenario>/path_pool.json` existente |
| `--pool-only` | Ejecuta solo la Fase 1 (preparación del pool) y sale sin lanzar la simulación |
| `--endpoint-tolerance-m M` | Tolerancia de coincidencia de endpoints (default 0.30) |
| `--stall-timeout SEC` | Matar el launch si el log no avanza en SEC segundos (default 600) |
| `--goal-position-tolerance-m M` | Radio de tolerancia para considerar un goal alcanzado (default 0.40) |
| `--unlock-head` | Permite que el joystick izquierdo mueva la cabeza (default: bloqueado) |

El resto de parámetros (`--scenarios`, `--n-goals`, `--st_sem_cmp_vc`, `--seed`, `--purge-data`, `--yes`, `--max-attempts`, `--timeout-seconds`, geometría de rutas) son idénticos a los del modo automatizado.

### Datos generados por sesión teleop

Cada ejecución crea una carpeta `data/<escenario>/run_<timestamp>_human/` con:

- `data.csv` — un sample por fila (pose, comando, goal, `is_episode_start`, flags de colisión/stuck).
- `descriptor.yaml` — metadatos del run (scenario, teacher=human, cuotas, `num_samples`, `num_episodes`).
- `episode_tries.json` — un registro por episodio con `episode_id`, `class`, `tries`. Lo escribe el recorder C++.
- `lom/` — imágenes `.png` con los Local Occupancy Maps por sample.

Además, en `data/<escenario>/`:

- `episode_metadata.jsonl` — un JSON por línea con `timestamp`, `episode_id`, `path_id`, `class`, `start`, `goal`, `attempt`. **Solo se escribe cuando el episodio se graba con éxito**; intentos fallidos no aparecen. Es la fuente autoritativa para `total_saved` y para el filtro anti-repetición entre sesiones.
- `failed_paths.json` — rutas descartadas (reintentos agotados o saltadas con Y). Se acumula entre sesiones.

### Nota sobre `is_episode_start` y `episode_id`

`teleop_dataset_recorder.py` publica `episode_id` (Int32) y `goal_active` (Bool) en topics separados, y existe una race conocida en el recorder C++ por la cual el primer sample de un episodio puede quedar con `is_episode_start=0`. Mitigación aplicada: `time.sleep(0.05)` entre ambos publishes en `begin_episode_at_start`. Aun así, en datasets antiguos algunos episodios pueden no tener el flag. **Para contar episodios, usa siempre la transición de `episode_id`** en `data.csv`, no `is_episode_start`. `count_goals_in_csv` ya implementa este criterio y `load_goal_counter` usa `max(count_csv, count_metadata)` para blindar el conteo.

---

## Estructura de datos del pool

Cada ejecución genera el pool con `path_generator.py` y lo instala en `data/<escenario>/`:

- `path_pool.json` — pool activo: `paths` con `id`, `class`, `task_type`, `start`, `goal`, `metrics`, `search`.
- `path_pool_poses.json` — waypoints suavizados por `id`, en formato `{id: [[x,y], ...]}`.

Los pools generados se guardan también en carpetas `teleop_pool_seed_<seed>/` por trazabilidad, junto con:

- `path_pool.csv` — tabla plana de métricas por ruta.
- `report.json` / `report.md` — resumen de la generación (intentos, tiempo, rechazos por causa, diagnósticos de búsqueda v_complex).
- `routes_<class>.png` — inspección visual de las rutas aceptadas por clase.

Estas carpetas son autocontenidas y pueden borrarse sin afectar el pool activo. Para ver las rutas, abre directamente las `routes_<class>.png`; `path_generator.py` no tiene flag `--visualize`.

### Modo `--incremental` — archivos de estado

Cuando se usa `--incremental`, `data_analysis/generate_path_pool.py` mantiene:

- `path_pool_history.json` / `path_pool_history_poses.json` — pool acumulado histórico (todas las rutas generadas, incluidas las ya grabadas).
- `incremental_state.json` — `{committed_before, add, total_after, target_add}` con los contadores de la última ejecución.

`path_pool.json` es siempre la vista filtrada (solo paths live), mientras que `path_pool_history.json` conserva todo. Esto permite volver a `--incremental` varias veces sin perder rutas.

---

## Seguridad y purga de datos

Por defecto ambos scripts conservan datos y ejecuciones anteriores. Cada intento de generación usa una carpeta única; nunca sobrescribe una carpeta previa.

`--purge-data` elimina todo `dataset/data/` antes de empezar. En terminal interactivo (TTY) pide escribir `PURGE` para confirmar. En ejecución no interactiva (redirección, pipe, CI) exige `--yes` explícito y aborta si no se pasa. **Usar con cuidado**: también borra episodios previamente grabados. Es incompatible con `--incremental` (el script lo rechaza).

Los logs de generación y grabación, junto con un archivo de estado por escenario, se guardan en el directorio definido con `--log-dir`.

---

## Verificación del dataset

Para comprobar que un dataset se generó correctamente:

1. Revisa que la última carpeta `run_*` tenga `descriptor.yaml` con `num_samples > 0`.
2. En modo teleop, verifica que `episode_metadata.jsonl` tenga una línea por goal exitoso y que ningún par `start`/`goal` se repita entre sesiones:

```bash
python3 - <<'PY'
import json, math
path = "src/Navigation/mapless_navigation/dataset/data/E01/episode_metadata.jsonl"
entries = [json.loads(l) for l in open(path)]
seen, dups = [], 0
for e in entries:
    s, g = e["start"], e["goal"]
    for ps, pg in seen:
        if math.hypot(s[0]-ps[0], s[1]-ps[1]) < 0.30 and \
           math.hypot(g[0]-pg[0], g[1]-pg[1]) < 0.30:
            dups += 1
            print(f"DUP: {e['path_id']} {e['class']} {s}->{g}")
    seen.append((s, g))
print(f"total={len(entries)} duplicates={dups}")
PY
```

3. Opcionalmente, revisa visualmente las rutas abriendo `data/<escenario>/teleop_pool_seed_*/routes_<class>.png`. Cada panel muestra una ruta sobre el mapa con su clase, longitud, conteo de giros y clearance mínimo.

---

## Análisis de similitud entre escenarios (métrica de Wasserstein)

El script `compute_similarity.py` (ubicado en `webots_env/scripts/maps/`) permite calcular la distancia de Wasserstein entre los mapas de ocupación de distintos escenarios. Esta métrica es útil para:

- Cuantificar la diversidad del conjunto de escenarios utilizados para entrenamiento.
- Predecir la dificultad de generalización del sistema de navegación mapless a nuevos entornos.
- Seleccionar escenarios de prueba que cubran un rango de similitud respecto a los de entrenamiento.

### Uso básico

```bash
cd src/Hardware/justina_description/webots_env/scripts/maps
python3 compute_similarity.py --method sinkhorn --max_points 4000
```

El script detecta automáticamente todos los escenarios disponibles en `environments/maps/` y genera una matriz de distancias en formato CSV (`wasserstein_distances_sinkhorn.csv`). Los métodos disponibles son:

- `sliced`: Sliced-Wasserstein (rápido, aproximado).
- `sinkhorn`: Wasserstein regularizado por entropía (equilibrio velocidad/precisión, recomendado).
- `emd`: Wasserstein-1 exacto (lento pero exacto).

### Integración con el dataset

Para analizar la similitud de los escenarios utilizados en un dataset:

1. Asegúrate de que los mapas de todos los escenarios estén en `environments/maps/<escenario>/`.
2. Ejecuta `compute_similarity.py` para obtener la matriz de distancias.
3. Compara la matriz con las tasas de éxito de navegación obtenidas en cada escenario. La hipótesis es que a mayor distancia (menor similitud), mayor degradación del rendimiento.

Esta información puede guiar la selección de escenarios para entrenamiento (buscando diversidad) y para pruebas (cubriendo un espectro de dificultad).

---

## Solución de problemas

- **`ROS 2 package 'mapless_dataset' not found`**: `source install/setup.bash` no está activo o el daemon de ROS tiene caché stale. Ejecuta `source install/setup.bash && ros2 daemon stop`.
- **El script no encuentra los mapas**: verifica que `map_generator.py` se haya ejecutado y que las rutas en `MAPS_ROOT` sean correctas. Debe existir `map_<folder>_prohibition.yaml` y su PNG asociado.
- **Fallo en la validación de cuotas**: revisa `data/<escenario>/teleop_pool_seed_*/report.json` (campos `class_counts`, `class_deficits`, `stop_reason`, `failure_counts`) para ver qué clase de ruta no alcanzó la cuota. Ajusta `--max-attempts` o los parámetros geométricos. Nota: el archivo `generation_manifest.json` mencionado en versiones antiguas del script no se produce.
- **Colisiones durante la grabación (automatizado)**: considera ajustar los parámetros de `simple_move` (ver sección de estabilización) o reducir la velocidad máxima.
- **El robot se queda atascado**: aumenta `--stall-timeout` o revisa la presencia de obstáculos no mapeados.
- **`Session complete` repetido en el log del teleop**: el launch cerró y el shell no lo detectó a tiempo. `collect_dataset_teleop.sh` hace re-clasificación post-loop (`launch_exit → teleop_complete`) cuando detecta `Stopping:` o `Session complete:` en el log. Si el contador de goals queda por debajo del objetivo, el estado se marca como `teleop_partial` y el script retorna `exit 1`.
- **`total_saved` muestra un valor menor que `episode_metadata.jsonl`**: el recorder usa `max(count_csv, count_metadata)`. Si `csv=0` y `meta=N`, la causa típica es la race del recorder C++ que pierde `is_episode_start=1` en el primer sample. El conteo por `episode_metadata.jsonl` es autoritativo. Si ambos coinciden pero `total_saved=0` con el archivo poblado, verifica con `ros2 param get /teleop_dataset_recorder output_dir` y `ros2 param get /teleop_dataset_recorder scenario` que apunten al `data/EXX/` real.
- **`Cannot record: pool not ready (deficits={...})`**: el filtro de endpoints descartó rutas del pool que ya estaban en `episode_metadata.jsonl` de sesiones previas, dejando déficit para alguna clase. Solución: `--incremental` (regenera el déficit) o `--record-only` (acepta el déficit y graba solo lo disponible, aunque el sampler se negará a iniciar si alguna cuota positiva queda en cero).
- **El pool se agota antes de cumplir cuotas**: aumenta `--n-goals` (el script regenera con `quota + deficit + 1` por clase), o usa `--endpoint-tolerance-m` más alto si hay falsos negativos en el filtro.
- **`--incremental` reporta `WARNING: post-dedup coverage below target`**: el generador produjo nuevas rutas pero todas colisionaron con el history. Re-ejecuta `--incremental` para dar otra oportunidad al generador (cada corrida pide `need_margin = int(need*1.2)+5` rutas nuevas).
- **`--incremental` reporta `committed=0` aunque haya episodios grabados**: probablemente `episode_metadata.jsonl` no existe o no tiene `path_id`. En datasets antiguos sin metadata, el fallback por CSV usa tolerancia 0.30 m sobre `pose_x, pose_y` al primer sample; si el robot empezó a grabar con `start_reached_radius_m=0.5`, la distancia al `start` del pool puede exceder 0.30 m. Sube `--endpoint-tolerance-m` a 0.60 en ese caso.
- **Error de importación de `ot`**: instala la librería POT con `pip install POT`. Solo es necesaria para `compute_similarity.py`, no para los scripts de recolección.
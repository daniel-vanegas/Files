# Recolección de datasets con rutas precomputadas

Este directorio contiene dos orquestadores de recolección de datasets basados en pools de rutas precomputadas:

- `collect_dataset_batch_classic_nav.sh` — recolección **automatizada** con el sistema Classic Navigation (el robot navega solo).
- `collect_dataset_teleop.sh` — recolección **teleoperada** (un operador humano conduce el robot con joystick por cada ruta del pool).

Ambos scripts llaman a `path_generator.py`, validan cuotas y archivos de poses, publican el pool activo en `data/<escenario>/path_pool*.json` y lanzan ROS 2/Webots. No se necesitan scripts auxiliares de orquestación.

## Prerrequisitos

- ROS 2 (Humble o superior) y Webots instalados y configurados.
- Entorno de trabajo compilado: `colcon build` y `source install/setup.bash`.
- Dependencias de Python: `numpy`, `Pillow`, `PyYAML`, `pot` (para el análisis de similitud, ver sección final).
- Los mapas de los escenarios deben existir en `webots_env/scripts/maps/<escenario>/map_<escenario>.png` (generados previamente con `map_generator.py`).

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

1. **Fase 1 — Preparación del pool**. El script filtra las rutas cuyo `start`/`goal` ya aparecen en sesiones previas (`episode_metadata.jsonl`, `failed_paths.json`, y los `data.csv` de cada `run_*`). Si tras filtrar faltan rutas para cubrir las cuotas, invoca `path_generator.py` para regenerar un pool con cuotas mayores.
2. **Fase 2 — Recolección**. Abre Webots + RViz y espera. Cada pulsación de **B** asigna la siguiente ruta del pool al operador. El operador conduce hasta el `start` de la ruta y luego hasta el `goal` para completar el episodio.
3. **Cierre**. Cuando se cumple el número de goals pedido (o se agotan los paths del pool), el recorder C++ cierra el launch solo y el script imprime `teleop_complete`.

### Controles

- **B** — Asignar ruta / pausar grabación. Si se pulsa durante `WAITING_FOR_START` (antes de llegar al start), la ruta se devuelve al pool.
- **X** — Teleport al `start` de la ruta actual (o a una sala aleatoria si no hay ruta asignada).
- **Y** — Saltar la ruta actual y asignar la siguiente. La ruta saltada se añade a `failed_paths.json` con `last_status: skipped_by_operator`.

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

### Solo grabar con un pool existente

```bash
bash collect_dataset_teleop.sh --scenarios "E01" --n-goals 10 --record-only
```

`--record-only` desactiva el filtrado y la regeneración. Usa el pool tal cual está en `data/<escenario>/path_pool*.json`. Útil si ya preparaste el pool y solo quieres grabar.

### Filtrado de rutas ya usadas

Por defecto `collect_dataset_teleop.sh` **no repite** rutas entre sesiones. Cada sesión lee:

- `episode_metadata.jsonl` — un registro por goal exitoso (con `path_id`, `class`, `start`, `goal`, `attempt`).
- `failed_paths.json` — rutas que agotaron reintentos o fueron saltadas.
- `data.csv` de cada `run_*` — recoge también endpoints de sesiones no-teleop (classic_nav, nav2).

Las rutas cuyo par `start`/`goal` coincide (dentro de `--endpoint-tolerance-m`, default 0.30 m) con alguna entrada previa se descartan. Si tras el filtro hay déficits, se regenera un pool nuevo con cuotas mayores (`quota + deficit + 1` por clase).

Para desactivar el filtrado: `--no-exclude-used-endpoints` (o `--record-only`). Para aumentar la tolerancia: `--endpoint-tolerance-m 0.40`.

### Parámetros exclusivos del modo teleop

| Flag | Descripción |
|---|---|
| `--record-only` | No filtra ni regenera; usa el pool existente tal cual |
| `--endpoint-tolerance-m M` | Tolerancia de coincidencia de endpoints (default 0.30) |
| `--stall-timeout SEC` | Matar el launch si el log no avanza en SEC segundos (default 600) |
| `--goal-position-tolerance-m M` | Radio de tolerancia para considerar un goal alcanzado (default 0.40) |

El resto de parámetros (`--scenarios`, `--n-goals`, `--st_sem_cmp_vc`, `--seed`, `--purge-data`, `--yes`, geometría de rutas) son idénticos a los del modo automatizado.

### Datos generados por sesión teleop

Cada ejecución crea una carpeta `data/<escenario>/run_<timestamp>_human/` con:

- `data.csv` — un sample por fila (pose, comando, goal, `is_episode_start`, flags de colisión/stuck).
- `descriptor.yaml` — metadatos del run (scenario, teacher=human, cuotas, `num_samples`, `num_episodes`).
- `episode_tries.json` — un registro por episodio con `episode_id`, `class`, `tries`.
- `lom/` — imágenes `.png` con los Local Occupancy Maps por sample.

Además, en `data/<escenario>/`:

- `episode_metadata.jsonl` — un JSON por línea con `timestamp`, `episode_id`, `path_id`, `class`, `start`, `goal`, `attempt`. **Solo se escribe cuando el episodio se graba con éxito**; intentos fallidos no aparecen.
- `failed_paths.json` — rutas descartadas (colisión persistente, timeout agotado, salto manual con Y). Se acumula entre sesiones.

---

## Estructura de datos del pool

Cada ejecución genera el pool con `path_generator.py` y lo instala en `data/<escenario>/`:

- `path_pool.json` — pool activo: `paths` con `id`, `class`, `task_type`, `start`, `goal`, `metrics`, `search`.
- `path_pool_poses.json` — waypoints suavizados por `id`, en formato `{id: [[x,y], ...]}`.

Los pools generados se guardan también en carpetas `generation_seed_<seed>/` (automatizado) o `teleop_pool_seed_<seed>/` (teleop) por trazabilidad. Ambas carpetas son autocontenidas y pueden borrarse sin afectar el pool activo.

## Seguridad y purga de datos

Por defecto ambos scripts conservan datos y ejecuciones anteriores. Cada intento de generación usa una carpeta única; nunca sobrescribe una carpeta previa.

`--purge-data` elimina todo `dataset/data/` antes de empezar y exige confirmar escribiendo `PURGE`; `--yes` permite omitir la pregunta en una ejecución no interactiva. **Usar con cuidado**: también borra episodios previamente grabados.

Los logs de generación y grabación, junto con un archivo de estado por escenario, se guardan en el directorio definido con `--log-dir`.

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

3. Opcionalmente, visualiza las rutas del pool con `path_generator.py --visualize`.

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

## Solución de problemas

- **`ROS 2 package 'mapless_dataset' not found`**: `source install/setup.bash` no está activo o el daemon de ROS tiene caché stale. Ejecuta `source install/setup.bash && ros2 daemon stop`.
- **El script no encuentra los mapas**: verifica que `map_generator.py` se haya ejecutado y que las rutas en `MAPS_ROOT` sean correctas.
- **Fallo en la validación de cuotas**: revisa el log o `generation_manifest.json` para ver qué clase de ruta no alcanzó la cuota. Ajusta `--max-attempts` o los parámetros geométricos.
- **Colisiones durante la grabación (automatizado)**: considera ajustar los parámetros de `simple_move` (ver sección de estabilización) o reducir la velocidad máxima.
- **El robot se queda atascado**: aumenta `--stall-timeout` o revisa la presencia de obstáculos no mapeados.
- **`Session complete` repetido en el log del teleop**: el launch cerró y el shell no lo detectó a tiempo. Asegúrate de que `collect_dataset_teleop.sh` tiene la re-clasificación post-loop de `launch_exit → teleop_complete`.
- **El pool se agota antes de cumplir cuotas**: aumenta `--n-goals` para la regen (el script regenera con `quota + deficit + 1` por clase), o usa `--endpoint-tolerance-m` más alto si hay falsos negativos en el filtro.
- **Error de importación de `ot`**: instala la librería POT con `pip install POT`.

---

**Nota**: Los detalles específicos de los argumentos y la estructura de archivos pueden variar según la versión actual del script. Se recomienda ejecutar `bash collect_dataset_batch_classic_nav.sh --help` o `bash collect_dataset_teleop.sh --help` y revisar el código para confirmar.
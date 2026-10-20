# Recolección de datasets con rutas precomputadas (Classic Navigation)

Este directorio contiene el orquestador principal para generar un pool de rutas y grabar episodios de navegación utilizando el sistema Classic Navigation. El script `collect_dataset_batch_classic_nav.sh` llama directamente a `path_generator.py`, valida cuotas y archivos de poses, publica el pool activo en `data/<escenario>/path_pool*.json` y lanza ROS 2/Webots. No se necesitan scripts auxiliares de orquestación.

## Prerrequisitos

- ROS 2 (Humble o superior) y Webots instalados y configurados.
- Entorno de trabajo compilado: `colcon build` y `source install/setup.bash`.
- Dependencias de Python: `numpy`, `Pillow`, `PyYAML`, `pot` (para el análisis de similitud, ver sección final).
- Los mapas de los escenarios deben existir en `webots_env/scripts/maps/<escenario>/map_<escenario>.png` (generados previamente con `map_generator.py`).

## Prueba rápida

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

## Parámetros de generación

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

## Estabilización y control en pasajes estrechos (`v_complex`)

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

## Estructura de datos generados

Cada ejecución crea una carpeta única dentro de `dataset/data/<escenario>/` con el formato `generation_seed_<semilla>_<timestamp>/`. Dentro se encuentran:

- `path_pool_<escenario>.json`: pool de rutas generadas (coordenadas de waypoints, clase de complejidad, etc.).
- `poses_<escenario>.json`: poses asociadas a cada ruta.
- `episodes/`: directorio con los episodios grabados. Cada episodio contiene los archivos de sensores (LiDAR, RGB-D, odometría) y las acciones de control.
- `generation_manifest.json`: metadatos de la generación (estado, cuotas, semilla, argumentos utilizados).

Solo se publica el pool plano (`data/<escenario>/path_pool.json`) cuando el pool y las poses pasan la validación completa de cuotas. `generation_manifest.json` registra el estado de la ejecución actual y evita usar pools antiguos si la generación falla.

## Seguridad y purga de datos

Por defecto, el script conserva datos y ejecuciones anteriores. Cada intento de generación usa una carpeta única; nunca sobrescribe una carpeta previa.

`--purge-data` elimina todo `dataset/data/` antes de empezar y exige confirmar escribiendo `PURGE`; `--yes` permite omitir la pregunta en una ejecución no interactiva. **Usar con cuidado**: también borra episodios previamente grabados.

Los logs de generación y grabación, junto con un archivo de estado por escenario, se guardan en el directorio definido con `--log-dir`.

## Verificación del dataset

Para comprobar que un dataset se generó correctamente:

1. Revisa que `generation_manifest.json` tenga `"status": "success"`.
2. Verifica que el número de rutas por clase coincida con las cuotas solicitadas.
3. Inspecciona que cada episodio en `episodes/` contenga los archivos de sensores y control esperados.
4. Opcionalmente, ejecuta una visualización rápida de las rutas con `path_generator.py --visualize`.

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

- **El script no encuentra los mapas**: verifica que `map_generator.py` se haya ejecutado y que las rutas en `MAPS_BASE` sean correctas.
- **Fallo en la validación de cuotas**: revisa `generation_manifest.json` para ver qué clase de ruta no alcanzó la cuota. Ajusta `--max-attempts` o los parámetros geométricos.
- **Colisiones durante la grabación**: considera ajustar los parámetros de `simple_move` (ver sección de estabilización) o reducir la velocidad máxima.
- **El robot se queda atascado**: aumenta `--stall-timeout` o revisa la presencia de obstáculos no mapeados.
- **Error de importación de `ot`**: instala la librería POT con `pip install POT`.

---

**Nota**: Los detalles específicos de los argumentos y la estructura de archivos pueden variar según la versión actual del script. Se recomienda ejecutar `bash collect_dataset_batch_classic_nav.sh --help` y revisar el código para confirmar.

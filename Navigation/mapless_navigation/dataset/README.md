# Dataset de navegación mapless (Justina)

Este paquete reúne la generación de pools de rutas, la simulación Webots y la grabación ROS 2. Los scripts se ejecutan desde el árbol de fuentes del workspace `mapless`; el paquete ROS 2 que proporciona el *launch* es `mapless_dataset`.

## Flujo recomendado: generar y grabar varios escenarios

Desde la raíz del repositorio:

```bash
cd ~/GIT/mapless
source install/setup.bash
```

Después de modificar los scripts o `random_goal_sampler.py`, recompila y vuelve a cargar el workspace:

```bash
colcon build --packages-select mapless_dataset
source install/setup.bash
```

El recolector unificado es:

```bash
src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh
```

Procesa los escenarios secuencialmente. Para cada escenario:

1. Genera un pool directamente con `path_generator.py`, desde el mismo orquestador.
2. Comprueba que el pool y sus poses estén completos y sean compatibles.
3. Publica `path_pool.json` y `path_pool_poses.json` en `dataset/data/<escenario>/`, que es la ubicación que consume `random_goal_sampler.py`.
4. Ejecuta `record_single_world_classic_nav.launch.xml`.
5. Espera a que el sampler termine, detecta errores o inactividad en el log y limpia el grupo de procesos de esa simulación antes de continuar.

Los pools originales y sus reportes permanecen bajo `data/<escenario>/generation_seed_<semilla>/`. Si esa carpeta ya existe, la nueva ejecución usa un nombre único como `generation_seed_<semilla>_run_<fecha>_<pid>`; no sobrescribe ejecuciones anteriores. El pool plano bajo `data/<escenario>/` es la copia activa que leerá el sampler. Los logs de la tanda se guardan en `dataset/logs/`, fuera de `data/`, para que `--purge-data` no los elimine.

El recolector consulta el `generation_manifest.json` recién generado antes de iniciar cada grabación y exige que los pools activos coincidan con el pool validado de esa tanda. Si la generación falla o queda incompleta, ese escenario se omite en la grabación aunque existan pools activos antiguos; así se evita grabar accidentalmente una ejecución anterior con la misma semilla.

### Argumentos principales

| Argumento | Comportamiento |
|---|---|
| `--scenarios "E01 E02 ..."` | Lista de escenarios; por defecto `E01 E02 E03`. |
| `--n-goals N` | Total de rutas/objetivos por escenario con distribución balanceada; por defecto **100**. |
| `--st_sem_cmp_vc S_S_C_V` | Cuotas explícitas en orden `straight_semi_complex_complex_v_complex`. La suma determina cuántos objetivos grabará el sampler. |
| `--seed-base N` | Semilla base. Para `E03`, la semilla usada es `N + 3`. Por defecto `2026101400`. |
| `--timeout-seconds SEC` | Límite de generación de rutas **por escenario**. No limita el tiempo de simulación ni la grabación. Si se omite, no se aplica límite de tiempo. |
| `--max-attempts N` | Límite opcional de intentos de generación por escenario. |
| `--progress-interval-seconds SEC` | Intervalo de avance del generador; por defecto `5`. |
| `--maps-root DIR` | Raíz de mapas; por defecto `webots_env/environments/maps`. |
| `--hard-clearance-m M` | Margen mínimo a obstáculos; por defecto `0.35`. |
| `--min-goal-distance-m M`, `--max-goal-distance-m M` | Distancia euclidiana mínima y máxima ordinaria; por defecto `2.0` y `3.0`. |
| `--max-path-length-m M`, `--max-length-ratio R` | Límites de longitud de ruta; por defecto `3.0` y `1.18`. |
| `--v-max-goal-distance-m M`, `--v-max-path-length-m M` | Límites de distancia y longitud para rutas `v_complex`; por defecto `3.6` y `4.2`. |
| `--v-goal-candidates N`, `--v-plan-candidates N` | Candidatos por inicio y planes por inicio `v_complex`; por defecto `10` y `3`. |
| `--endpoint-obstacle-band-m M`, `--obstacle-influence-m M` | Banda del extremo e influencia de obstáculo; por defecto `0.65` y `0.80`. |
| `--v-hotspot-probability P`, `--v-hotspot-radius-m M`, `--plan-cell-factor N` | Ajustes del muestreo cerca de curvas y resolución de planificación. |
| `--stall-timeout SEC` | Termina una simulación si el log no cambia durante ese intervalo; `0` desactiva la detección de inactividad. No es el timeout del generador. |
| `--purge-data` | Elimina el directorio completo `dataset/data/` antes de generar los pools. |
| `--yes` | Omite la confirmación interactiva; solo se acepta junto con `--purge-data`. |
| `--log-dir DIR` | Cambia el directorio de logs de la tanda. |
| `-- ARGUMENTOS` | Pasa argumentos adicionales a `ros2 launch`. |

No uses `--purge-data` si necesitas conservar datos anteriores. `--yes` no borra datos por sí solo.

## Ejemplo 1: limpiar los datos y generar 100 objetivos por escenario

El siguiente comando elimina primero `dataset/data/` y procesa los diez escenarios indicados. El valor predeterminado del recolector es 100 objetivos por escenario:

```bash
src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
  --purge-data \
  --yes \
  --scenarios "E01 E02 E03 E04 E05 E06 E07 E08 E09 E10"
```

## Ejemplo 2: conservar los datos, 200 objetivos por escenario y 15 minutos para generar rutas

Las cuotas `20_70_70_40` suman 200 objetivos por escenario. El límite de 900 segundos se aplica a la generación de rutas de cada escenario, no a la grabación:

```bash
src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
  --scenarios "E01 E02 E03 E04 E05 E06 E07 E08 E09 E10" \
  --st_sem_cmp_vc 20_70_70_40 \
  --timeout-seconds 900
```

Si un escenario no obtiene todas sus cuotas dentro del límite, su pool no se publica como pool activo y ese escenario se omite en la fase de grabación. La tanda informa el fallo y conserva los reportes/logs para diagnóstico.

## Flujo único de prueba

Para no desincronizar el pool, el manifiesto y los datos grabados, usa el orquestador clásico para la prueba de extremo a extremo. Los parámetros del generador se especifican mediante sus argumentos; `--help` muestra la lista completa. No se mantienen scripts shell auxiliares para generación/orquestación.

Ejemplo breve para validar el flujo en `E03`:

```bash
src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
  --scenarios "E03" \
  --n-goals 4 \
  --seed-base 2026101400 \
  --timeout-seconds 900 \
  --max-attempts 25000 \
  --stall-timeout 300 \
  --log-dir /tmp/mapless_path_smoke/logs
```

## Salidas importantes

```text
dataset/data/
├── generation_manifest.json
├── E01/
│   ├── generation_seed_<seed>[_run_<fecha>_<pid>]/
│   │   ├── path_pool.json
│   │   ├── path_pool_poses.json
│   │   ├── path_pool.csv
│   │   └── report.json
│   ├── path_pool.json
│   ├── path_pool_poses.json
│   └── <directorios de grabación>
└── ...
dataset/logs/collection_<fecha>/
├── generation_batch.log
├── E01_recording.log
├── E01.status
└── ...
```

El sampler requiere que las cuotas del pool sean positivas, sumen `num_goals` y que haya suficientes rutas utilizables por clase. Las cuotas del pool son las cuotas efectivas; por eso el modo explícito admite distribuciones no uniformes.

## Conversión posterior a `.npz`

El script `generate_npz_data.py` convierte las grabaciones crudas a archivos `.npz` junto a cada directorio de ejecución.

Todos los escenarios:

```bash
python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py
```

Un escenario:

```bash
python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py E01
```

Una ejecución:

```bash
python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py E01/<directorio_de_grabacion>
```

Notas:

- Los episodios no se filtran en esta etapa; las políticas de colisión y atasco se aplican posteriormente en `dataset_utils.filter_by_episode`.
- El `.npz` almacena un array de objetos de diccionarios. Para cargarlo se necesita `np.load(path, allow_pickle=True)`.
- Volver a ejecutar la conversión sobrescribe el `.npz` correspondiente.

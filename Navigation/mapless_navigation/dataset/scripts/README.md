# Recolección con rutas precomputadas (Classic Navigation)

El único orquestador `.sh` para generar el pool y grabar episodios con Classic Navigation es `collect_dataset_batch_classic_nav.sh`. Llama directamente a `path_generator.py`, valida las cuotas y los archivos de poses, publica el pool activo en `data/<escenario>/path_pool*.json` y luego lanza ROS 2/Webots. No necesita scripts auxiliares de orquestación.

## Prueba pequeña

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

El total de rutas se distribuye de forma equilibrada entre `straight`, `semi_complex`, `complex` y `v_complex`. Para fijar cuotas por clase se usa `--st_sem_cmp_vc 10_10_15_20` en lugar de `--n-goals`.

## Parámetros de generación

Las cotas geométricas y el ajuste del planificador también se controlan desde argumentos; consulta `--help`. Algunos ejemplos:

```bash
bash collect_dataset_batch_classic_nav.sh --help
bash collect_dataset_batch_classic_nav.sh --scenarios "E03" --n-goals 4 \
  --hard-clearance-m 0.35 --min-goal-distance-m 2.0 \
  --max-goal-distance-m 3.0 --max-path-length-m 3.0 \
  --v-max-goal-distance-m 3.6 --v-max-path-length-m 4.2 \
  --v-goal-candidates 10 --v-plan-candidates 3
```

Los argumentos después de `--` pasan a `ros2 launch`, por ejemplo para sobrescribir un argumento de launch: `-- world_offset_x:=0.0`. `--timeout-seconds` y `--max-attempts` limitan la generación, no la duración de la grabación. `--stall-timeout` se aplica a la fase de grabación.

## Estabilización y control en pasajes estrechos (v_complex)
Para evitar colisiones contra las paredes en giros cerrados y pasillos angostos, el nodo de control simple_move admite parámetros de ajuste fino que pueden modificarse en navigation_controller/launch/controller.launch.xml o bien inyectarse directamente al final del script de recolección usando el separador --:

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

## Datos y seguridad

Por defecto, el script conserva datos y ejecuciones anteriores. Cada intento de generación usa una carpeta única `data/<escenario>/generation_seed_<semilla>...`; nunca sobrescribe una carpeta previa. Solo publica el pool plano cuando el pool y las poses pasan la validación completa de cuotas. `generation_manifest.json` registra el estado de la ejecución actual y evita usar pools antiguos si la generación falla.

`--purge-data` elimina todo `dataset/data/` antes de empezar y exige confirmar escribiendo `PURGE`; `--yes` permite omitir la pregunta en una ejecución no interactiva. Usar con cuidado porque también borra episodios previamente grabados.

Los logs de generación y grabación, junto con un archivo de estado por escenario, se guardan en el directorio definido con `--log-dir`.

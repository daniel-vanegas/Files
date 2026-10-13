# Geometry audit: E01

Offline diagnostic only; production pool and sampler were not changed.
- Source pool: `/home/devan/GIT/mapless/src/Navigation/mapless_navigation/dataset/data/E01/path_pool.json`
- Map: `/home/devan/GIT/mapless/src/Hardware/justina_description/webots_env/environments/maps/E01_C2_02/map_E01_C2_02_prohibition.yaml`
- Paths examined: 200
- Raw A* failures: {}
- Explicit path IDs: [98, 184, 577, 686, 732, 881, 891, 966, 1015, 1133, 1475, 1605, 1761, 1801, 2054, 2101, 2160, 2201, 2320, 2342, 2380, 2429, 2527, 2668, 2820, 2835, 2849, 2915, 3043, 3060, 3151, 3242, 3333, 3400, 3681, 3756, 3802, 4200, 4416, 4891]
- Requested IDs missing from pool: []
- CSV: `geometry_audit.csv`
- JSON: `geometry_audit_paths.json`
- Scaled obstacle-mask overlay: `geometry_audit_comparison.png`

Raw A* is regenerated for the same stored endpoints. This is a controlled comparison, but it may differ from the original generated route if the path planner/image version changed.
Inspect `raw_astar_*` against `stored_*` to determine whether grid-search geometry or post-processing changes the assigned morphology.

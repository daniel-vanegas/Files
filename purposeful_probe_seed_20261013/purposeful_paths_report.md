# Purposeful path probe: E01

Diagnostic only. This mode does not modify production pools or command the robot.
- Map: `/home/devan/GIT/mapless/src/Hardware/justina_description/webots_env/environments/maps/E01_C2_02/map_E01_C2_02_prohibition.yaml`
- Seed: 20261013
- Attempts: 1200/1200
- Elapsed: 408.0 s
- Constraints: Euclidean 2.00-3.00 m; length <= 3.00 m; clearance >= 0.15 m; preferred line/path clearance 0.45 m; near-optimal ratio <= 1.18

| Class | Accepted | Target |
|---|---:|---:|
| straight | 10 | 10 |
| semi_complex | 10 | 10 |
| complex | 10 | 10 |
| v_complex | 0 | 10 |

## Rejections

- `class_quota_full_semi_complex`: 434
- `path_too_long`: 267
- `class_quota_full_complex`: 221
- `class_quota_full_straight`: 194
- `turns_not_supported_by_obstacles`: 54

## Route design rule

A route is direct when the straight segment is safe with the preferred clearance. Otherwise the route follows clearance-aware A*; any non-straight class must have bends supported by obstacles. Free-space zigzags are rejected. The output is a candidate study, not a production path pool.

- CSV: `purposeful_paths.csv`
- JSON: `purposeful_paths.json`
- Plot: `purposeful_paths_by_class.png`

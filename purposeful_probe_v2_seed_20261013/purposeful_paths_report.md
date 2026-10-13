# Purposeful path probe: E01

Diagnostic only. This mode does not modify production pools or command the robot.
- Map: `/home/devan/GIT/mapless/src/Hardware/justina_description/webots_env/environments/maps/E01_C2_02/map_E01_C2_02_prohibition.yaml`
- Seed: 20261013
- Attempts: 1200/1200
- Elapsed: 61.1 s
- Constraints: Euclidean 2.00-3.00 m; length <= 3.00 m; hard footprint clearance >= 0.35 m; soft preferred clearance 0.45 m; near-optimal ratio <= 1.18

| Class | Accepted | Target |
|---|---:|---:|
| straight | 10 | 10 |
| semi_complex | 10 | 10 |
| complex | 10 | 10 |
| v_complex | 0 | 10 |

## Rejections

- `class_quota_full_straight`: 655
- `baseline_smoothing_not_safe`: 288
- `class_quota_full_semi_complex`: 182
- `path_too_long`: 25
- `class_quota_full_complex`: 17
- `no_endpoint_pair`: 2
- `detour_turns_not_supported_by_obstacles`: 1

## Route design rule

A direct route is accepted whenever it satisfies the hard robot-center clearance. Preferred clearance affects the soft route cost, not the class trigger. A* paths are line-of-sight simplified with dense checks; too-close paths are rejected. Complex classes require obstacle-supported maneuvers, and a U/return requires measurable backtracking or multiple obstacle interactions. Free-space zigzags are rejected. The output is a candidate study, not a production path pool.

- CSV: `purposeful_paths.csv`
- JSON: `purposeful_paths.json`
- Plot: `purposeful_paths_by_class.png`

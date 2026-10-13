# Directed candidate probe: E01

**Diagnostic only. Does not change path_pool.json and does not command the robot.**
- Map: `/home/devan/GIT/mapless/src/Hardware/justina_description/webots_env/environments/maps/E01_C2_02/map_E01_C2_02_prohibition.yaml`
- Seed: 20261013
- Constraints: Euclidean 2.00-3.00 m; path length <= 3.00 m; clearance >= 0.15 m
- Elapsed: 8.9 s
- Accepted paths: 40

| Requested shape | Accepted | Classified as intended class |
|---|---:|---:|
| straight | 10 | 2 (straight) |
| single_arc | 10 | 10 (semi_complex) |
| s_curve | 10 | 9 (complex) |
| u_turn_zigzag | 10 | 10 (v_complex) |

## Rejection / outcome counts

- `s_curve:accepted_complex`: 9
- `s_curve:accepted_v_complex`: 1
- `s_curve:clearance`: 2
- `single_arc:accepted_semi_complex`: 10
- `single_arc:clearance`: 2
- `single_arc:map_collision`: 4
- `single_arc:path_too_long`: 1
- `single_arc:waypoint_not_found`: 1
- `straight:accepted_complex`: 4
- `straight:accepted_semi_complex`: 4
- `straight:accepted_straight`: 2
- `straight:clearance`: 2
- `straight:map_collision`: 1
- `u_turn_zigzag:accepted_v_complex`: 10
- `u_turn_zigzag:clearance`: 5
- `u_turn_zigzag:distance_pair_missing`: 1
- `u_turn_zigzag:map_collision`: 48
- `u_turn_zigzag:path_too_long`: 168
- `u_turn_zigzag:waypoint_not_found`: 46

A directed via-point path is a geometry feasibility experiment only. The live classic planner still replans from start to goal; it will not follow these mandatory waypoints unless a separate path-following change is made.

- CSV: `directed_candidate_probe.csv`
- JSON: `directed_candidate_probe.json`
- Plot: `directed_candidate_probe_by_pattern.png`

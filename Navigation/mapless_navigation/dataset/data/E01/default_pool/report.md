# Path generator report — E01

- Seed: `108911`
- Accepted: **7**
- Attempts used: 390
- Attempt limit: unlimited; reached: False
- Time limit: 8.0 s; timeout hit: True
- Stop reason: **timeout**
- Complete: **False**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 3 | 3 | 0 |
| semi_complex | 3 | 3 | 0 |
| complex | 0 | 2 | 2 |
| v_complex | 1 | 2 | 1 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 3 |
| multiple_obstacle_navigation | 1 |
| single_obstacle_avoidance | 3 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 85 |
| v_complex:class_mismatch_unclassified | 68 |
| complex:path_length | 58 |
| complex:class_mismatch_unclassified | 45 |
| complex:class_mismatch_semi_complex | 42 |
| complex:direct_line_clear | 41 |
| v_complex:class_mismatch_complex | 32 |
| semi_complex:direct_line_clear | 7 |
| semi_complex:path_length | 4 |
| semi_complex:class_mismatch_unclassified | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 186 |
| v_exploration_start_batches | 83 |
| v_goal_candidates_evaluated | 1859 |
| v_hotspot_cells | 238 |
| v_hotspot_refreshes | 2 |
| v_hotspot_start_batches | 103 |
| v_theta_candidate_routes_evaluated | 558 |

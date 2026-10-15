# Path generator report — E03

- Seed: `107867`
- Accepted: **10**
- Attempts used: 193
- Attempt limit: unlimited; reached: False
- Time limit: 8.0 s; timeout hit: False
- Stop reason: **complete**
- Complete: **True**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 3 | 3 | 0 |
| semi_complex | 3 | 3 | 0 |
| complex | 2 | 2 | 0 |
| v_complex | 2 | 2 | 0 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 3 |
| constrained_passage | 2 |
| multiple_obstacle_navigation | 2 |
| single_obstacle_avoidance | 3 |

## Rejections

| Reason | Count |
|---|---:|
| complex:path_length | 51 |
| complex:class_mismatch_semi_complex | 34 |
| v_complex:class_mismatch_semi_complex | 26 |
| complex:direct_line_clear | 22 |
| complex:class_mismatch_unclassified | 22 |
| v_complex:class_mismatch_unclassified | 16 |
| v_complex:class_mismatch_complex | 7 |
| semi_complex:class_mismatch_unclassified | 3 |
| v_complex:no_promising_goal_bundle | 2 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 53 |
| v_exploration_start_batches | 42 |
| v_goal_candidates_evaluated | 530 |
| v_hotspot_cells | 378 |
| v_hotspot_refreshes | 2 |
| v_hotspot_start_batches | 11 |
| v_theta_candidate_routes_evaluated | 159 |

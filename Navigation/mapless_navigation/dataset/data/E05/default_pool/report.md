# Path generator report — E05

- Seed: `131604`
- Accepted: **7**
- Attempts used: 154
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
| complex | 1 | 2 | 1 |
| v_complex | 0 | 2 | 2 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 3 |
| constrained_passage | 1 |
| single_obstacle_avoidance | 3 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 31 |
| v_complex:class_mismatch_unclassified | 29 |
| complex:path_length | 25 |
| complex:direct_line_clear | 24 |
| complex:class_mismatch_unclassified | 13 |
| semi_complex:path_length | 6 |
| complex:class_mismatch_semi_complex | 6 |
| v_complex:class_mismatch_complex | 6 |
| v_complex:no_promising_goal_bundle | 3 |
| semi_complex:class_mismatch_unclassified | 2 |
| semi_complex:direct_line_clear | 1 |
| semi_complex:duplicate_endpoints | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 69 |
| v_exploration_start_batches | 69 |
| v_goal_candidates_evaluated | 651 |
| v_hotspot_cells | 0 |
| v_hotspot_refreshes | 2 |
| v_hotspot_start_batches | 0 |
| v_theta_candidate_routes_evaluated | 200 |

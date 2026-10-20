# Path generator report — E02

- Seed: `121132`
- Accepted: **200**
- Attempts used: 26315
- Attempt limit: unlimited; reached: False
- Time limit: 900.0 s; timeout hit: False
- Stop reason: **complete**
- Complete: **True**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 50 | 50 | 0 |
| semi_complex | 50 | 50 | 0 |
| complex | 50 | 50 | 0 |
| v_complex | 50 | 50 | 0 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 50 |
| multiple_obstacle_navigation | 50 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 8128 |
| v_complex:class_mismatch_unclassified | 5273 |
| v_complex:class_mismatch_complex | 3850 |
| complex:path_length | 3048 |
| complex:class_mismatch_semi_complex | 2289 |
| complex:direct_line_clear | 1652 |
| complex:class_mismatch_unclassified | 1485 |
| v_complex:no_promising_goal_bundle | 133 |
| semi_complex:path_length | 64 |
| v_complex:duplicate_endpoints | 53 |
| complex:duplicate_endpoints | 48 |
| semi_complex:class_mismatch_unclassified | 41 |
| semi_complex:direct_line_clear | 38 |
| complex:theta_star_no_path | 5 |
| semi_complex:duplicate_endpoints | 3 |
| semi_complex:class_mismatch_complex | 3 |
| straight:duplicate_endpoints | 1 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 17487 |
| v_exploration_start_batches | 3579 |
| v_goal_candidates_evaluated | 174032 |
| v_hotspot_cells | 2632 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 13908 |
| v_theta_candidate_routes_evaluated | 52364 |

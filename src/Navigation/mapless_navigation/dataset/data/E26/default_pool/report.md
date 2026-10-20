# Path generator report — E26

- Seed: `107697`
- Accepted: **173**
- Attempts used: 12813
- Attempt limit: unlimited; reached: False
- Time limit: 900.0 s; timeout hit: True
- Stop reason: **timeout**
- Complete: **False**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 50 | 50 | 0 |
| semi_complex | 50 | 50 | 0 |
| complex | 49 | 50 | 1 |
| v_complex | 24 | 50 | 26 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 49 |
| multiple_obstacle_navigation | 24 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 2826 |
| complex:path_length | 2776 |
| v_complex:class_mismatch_unclassified | 1743 |
| complex:class_mismatch_semi_complex | 1413 |
| v_complex:class_mismatch_complex | 1304 |
| complex:class_mismatch_unclassified | 1070 |
| complex:direct_line_clear | 490 |
| complex:theta_star_no_path | 432 |
| v_complex:no_promising_goal_bundle | 362 |
| semi_complex:path_length | 84 |
| complex:duplicate_endpoints | 44 |
| semi_complex:class_mismatch_unclassified | 35 |
| semi_complex:direct_line_clear | 16 |
| v_complex:duplicate_endpoints | 15 |
| semi_complex:theta_star_no_path | 14 |
| semi_complex:duplicate_endpoints | 5 |
| semi_complex:class_mismatch_complex | 4 |
| straight:duplicate_endpoints | 4 |
| straight:no_endpoint_pair | 3 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 6274 |
| v_exploration_start_batches | 1282 |
| v_goal_candidates_evaluated | 62740 |
| v_hotspot_cells | 3012 |
| v_hotspot_refreshes | 74 |
| v_hotspot_start_batches | 4992 |
| v_theta_candidate_routes_evaluated | 18822 |

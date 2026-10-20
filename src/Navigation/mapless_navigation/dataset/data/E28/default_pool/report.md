# Path generator report — E28

- Seed: `108666`
- Accepted: **200**
- Attempts used: 3734
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
| complex:path_length | 1376 |
| complex:class_mismatch_semi_complex | 531 |
| v_complex:class_mismatch_semi_complex | 421 |
| v_complex:class_mismatch_complex | 330 |
| complex:class_mismatch_unclassified | 271 |
| v_complex:class_mismatch_unclassified | 210 |
| semi_complex:path_length | 99 |
| complex:direct_line_clear | 87 |
| straight:no_endpoint_pair | 72 |
| v_complex:no_promising_goal_bundle | 65 |
| v_complex:duplicate_endpoints | 17 |
| semi_complex:class_mismatch_unclassified | 14 |
| complex:duplicate_endpoints | 13 |
| semi_complex:direct_line_clear | 7 |
| semi_complex:class_mismatch_complex | 7 |
| straight:duplicate_endpoints | 5 |
| complex:class_mismatch_v_complex | 4 |
| complex:no_endpoint_pair | 4 |
| semi_complex:duplicate_endpoints | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 1093 |
| v_exploration_start_batches | 241 |
| v_goal_candidates_evaluated | 10930 |
| v_hotspot_cells | 4416 |
| v_hotspot_refreshes | 79 |
| v_hotspot_start_batches | 852 |
| v_theta_candidate_routes_evaluated | 3279 |

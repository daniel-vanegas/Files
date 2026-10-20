# Path generator report — E12

- Seed: `125966`
- Accepted: **200**
- Attempts used: 11537
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
| complex:path_length | 2717 |
| v_complex:class_mismatch_semi_complex | 2014 |
| complex:class_mismatch_semi_complex | 1811 |
| complex:class_mismatch_unclassified | 1378 |
| complex:direct_line_clear | 1151 |
| v_complex:class_mismatch_unclassified | 1048 |
| v_complex:class_mismatch_complex | 882 |
| semi_complex:path_length | 79 |
| v_complex:no_promising_goal_bundle | 63 |
| v_complex:duplicate_endpoints | 58 |
| semi_complex:class_mismatch_unclassified | 47 |
| semi_complex:direct_line_clear | 37 |
| complex:duplicate_endpoints | 31 |
| semi_complex:duplicate_endpoints | 8 |
| straight:duplicate_endpoints | 3 |
| semi_complex:class_mismatch_complex | 3 |
| v_complex:path_length | 3 |
| complex:no_endpoint_pair | 2 |
| straight:no_endpoint_pair | 1 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 4118 |
| v_exploration_start_batches | 865 |
| v_goal_candidates_evaluated | 41177 |
| v_hotspot_cells | 3147 |
| v_hotspot_refreshes | 88 |
| v_hotspot_start_batches | 3253 |
| v_theta_candidate_routes_evaluated | 12354 |

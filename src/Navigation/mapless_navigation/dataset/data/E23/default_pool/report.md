# Path generator report — E23

- Seed: `111265`
- Accepted: **200**
- Attempts used: 4299
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
| complex:path_length | 1088 |
| v_complex:class_mismatch_complex | 644 |
| v_complex:class_mismatch_semi_complex | 640 |
| complex:class_mismatch_semi_complex | 600 |
| complex:class_mismatch_unclassified | 373 |
| v_complex:class_mismatch_unclassified | 359 |
| complex:direct_line_clear | 161 |
| semi_complex:path_length | 94 |
| v_complex:no_promising_goal_bundle | 47 |
| semi_complex:class_mismatch_unclassified | 31 |
| straight:no_endpoint_pair | 20 |
| v_complex:duplicate_endpoints | 20 |
| semi_complex:direct_line_clear | 7 |
| complex:duplicate_endpoints | 5 |
| semi_complex:class_mismatch_complex | 4 |
| semi_complex:duplicate_endpoints | 3 |
| straight:duplicate_endpoints | 1 |
| complex:class_mismatch_v_complex | 1 |
| complex:no_endpoint_pair | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 1760 |
| v_exploration_start_batches | 390 |
| v_goal_candidates_evaluated | 17600 |
| v_hotspot_cells | 4492 |
| v_hotspot_refreshes | 87 |
| v_hotspot_start_batches | 1370 |
| v_theta_candidate_routes_evaluated | 5280 |

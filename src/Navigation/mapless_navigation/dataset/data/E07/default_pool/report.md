# Path generator report — E07

- Seed: `117317`
- Accepted: **200**
- Attempts used: 14635
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
| v_complex:class_mismatch_semi_complex | 4384 |
| complex:path_length | 2984 |
| v_complex:class_mismatch_unclassified | 2524 |
| v_complex:class_mismatch_complex | 2006 |
| complex:class_mismatch_semi_complex | 1300 |
| complex:class_mismatch_unclassified | 530 |
| complex:direct_line_clear | 345 |
| v_complex:duplicate_endpoints | 104 |
| semi_complex:path_length | 103 |
| complex:duplicate_endpoints | 56 |
| v_complex:no_promising_goal_bundle | 36 |
| semi_complex:class_mismatch_unclassified | 26 |
| semi_complex:direct_line_clear | 17 |
| semi_complex:class_mismatch_complex | 6 |
| straight:duplicate_endpoints | 5 |
| semi_complex:duplicate_endpoints | 4 |
| complex:theta_star_no_path | 3 |
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
| v_attempts | 9104 |
| v_exploration_start_batches | 1794 |
| v_goal_candidates_evaluated | 91040 |
| v_hotspot_cells | 1500 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 7310 |
| v_theta_candidate_routes_evaluated | 27312 |

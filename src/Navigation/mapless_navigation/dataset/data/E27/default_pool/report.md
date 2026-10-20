# Path generator report — E27

- Seed: `130061`
- Accepted: **200**
- Attempts used: 7351
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
| complex:path_length | 1869 |
| complex:class_mismatch_semi_complex | 994 |
| v_complex:class_mismatch_semi_complex | 927 |
| complex:theta_star_no_path | 734 |
| v_complex:class_mismatch_complex | 700 |
| complex:class_mismatch_unclassified | 680 |
| v_complex:class_mismatch_unclassified | 551 |
| v_complex:no_promising_goal_bundle | 228 |
| complex:direct_line_clear | 226 |
| semi_complex:path_length | 82 |
| semi_complex:theta_star_no_path | 38 |
| semi_complex:class_mismatch_unclassified | 33 |
| complex:duplicate_endpoints | 24 |
| v_complex:duplicate_endpoints | 23 |
| straight:no_endpoint_pair | 22 |
| semi_complex:direct_line_clear | 10 |
| straight:duplicate_endpoints | 4 |
| semi_complex:class_mismatch_complex | 3 |
| semi_complex:duplicate_endpoints | 2 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2479 |
| v_exploration_start_batches | 530 |
| v_goal_candidates_evaluated | 24790 |
| v_hotspot_cells | 3583 |
| v_hotspot_refreshes | 82 |
| v_hotspot_start_batches | 1949 |
| v_theta_candidate_routes_evaluated | 7437 |

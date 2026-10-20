# Path generator report — E22

- Seed: `131342`
- Accepted: **200**
- Attempts used: 20583
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
| v_complex:class_mismatch_semi_complex | 4794 |
| complex:path_length | 4177 |
| complex:class_mismatch_semi_complex | 2981 |
| v_complex:class_mismatch_unclassified | 2483 |
| complex:class_mismatch_unclassified | 2279 |
| v_complex:class_mismatch_complex | 1930 |
| complex:direct_line_clear | 1349 |
| v_complex:no_promising_goal_bundle | 100 |
| semi_complex:path_length | 80 |
| v_complex:duplicate_endpoints | 63 |
| complex:duplicate_endpoints | 45 |
| semi_complex:class_mismatch_unclassified | 44 |
| semi_complex:direct_line_clear | 21 |
| straight:duplicate_endpoints | 10 |
| complex:theta_star_no_path | 10 |
| semi_complex:duplicate_endpoints | 5 |
| v_complex:path_length | 5 |
| straight:no_endpoint_pair | 3 |
| semi_complex:class_mismatch_complex | 3 |
| semi_complex:theta_star_no_path | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 9425 |
| v_exploration_start_batches | 1878 |
| v_goal_candidates_evaluated | 94250 |
| v_hotspot_cells | 2742 |
| v_hotspot_refreshes | 93 |
| v_hotspot_start_batches | 7547 |
| v_theta_candidate_routes_evaluated | 28275 |

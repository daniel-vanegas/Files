# Path generator report — E08

- Seed: `131165`
- Accepted: **200**
- Attempts used: 16236
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
| complex:theta_star_no_path | 3818 |
| complex:path_length | 3645 |
| v_complex:class_mismatch_semi_complex | 1705 |
| complex:class_mismatch_semi_complex | 1592 |
| v_complex:class_mismatch_unclassified | 1374 |
| complex:class_mismatch_unclassified | 1099 |
| v_complex:class_mismatch_complex | 1053 |
| v_complex:no_promising_goal_bundle | 725 |
| complex:direct_line_clear | 456 |
| semi_complex:theta_star_no_path | 151 |
| semi_complex:path_length | 136 |
| v_complex:duplicate_endpoints | 101 |
| complex:duplicate_endpoints | 64 |
| semi_complex:class_mismatch_unclassified | 49 |
| straight:no_endpoint_pair | 25 |
| semi_complex:direct_line_clear | 19 |
| straight:duplicate_endpoints | 13 |
| semi_complex:class_mismatch_complex | 6 |
| semi_complex:duplicate_endpoints | 3 |
| semi_complex:class_mismatch_v_complex | 1 |
| v_complex:path_length | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 5009 |
| v_exploration_start_batches | 995 |
| v_goal_candidates_evaluated | 50090 |
| v_hotspot_cells | 2830 |
| v_hotspot_refreshes | 86 |
| v_hotspot_start_batches | 4014 |
| v_theta_candidate_routes_evaluated | 15027 |

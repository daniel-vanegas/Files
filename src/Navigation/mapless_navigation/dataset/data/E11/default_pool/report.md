# Path generator report — E11

- Seed: `111671`
- Accepted: **200**
- Attempts used: 7727
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
| complex:path_length | 1902 |
| v_complex:class_mismatch_semi_complex | 1608 |
| v_complex:class_mismatch_complex | 1080 |
| complex:class_mismatch_semi_complex | 1039 |
| v_complex:class_mismatch_unclassified | 816 |
| complex:class_mismatch_unclassified | 555 |
| complex:direct_line_clear | 300 |
| semi_complex:path_length | 93 |
| v_complex:no_promising_goal_bundle | 34 |
| v_complex:duplicate_endpoints | 28 |
| semi_complex:direct_line_clear | 20 |
| semi_complex:class_mismatch_unclassified | 20 |
| complex:duplicate_endpoints | 11 |
| complex:theta_star_no_path | 7 |
| straight:no_endpoint_pair | 5 |
| straight:duplicate_endpoints | 3 |
| semi_complex:duplicate_endpoints | 3 |
| semi_complex:class_mismatch_complex | 2 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 3616 |
| v_exploration_start_batches | 754 |
| v_goal_candidates_evaluated | 36160 |
| v_hotspot_cells | 4916 |
| v_hotspot_refreshes | 92 |
| v_hotspot_start_batches | 2862 |
| v_theta_candidate_routes_evaluated | 10848 |

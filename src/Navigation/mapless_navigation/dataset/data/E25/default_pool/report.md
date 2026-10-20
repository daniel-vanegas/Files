# Path generator report — E25

- Seed: `121975`
- Accepted: **200**
- Attempts used: 6225
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
| complex:path_length | 1662 |
| complex:class_mismatch_semi_complex | 1081 |
| v_complex:class_mismatch_semi_complex | 834 |
| v_complex:class_mismatch_complex | 710 |
| complex:class_mismatch_unclassified | 664 |
| v_complex:class_mismatch_unclassified | 603 |
| complex:direct_line_clear | 217 |
| semi_complex:path_length | 74 |
| v_complex:duplicate_endpoints | 53 |
| semi_complex:class_mismatch_unclassified | 34 |
| complex:duplicate_endpoints | 25 |
| straight:duplicate_endpoints | 22 |
| v_complex:no_promising_goal_bundle | 10 |
| semi_complex:direct_line_clear | 9 |
| straight:no_endpoint_pair | 9 |
| semi_complex:duplicate_endpoints | 9 |
| semi_complex:class_mismatch_complex | 4 |
| complex:theta_star_no_path | 3 |
| complex:class_mismatch_v_complex | 2 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2260 |
| v_exploration_start_batches | 513 |
| v_goal_candidates_evaluated | 22600 |
| v_hotspot_cells | 2936 |
| v_hotspot_refreshes | 76 |
| v_hotspot_start_batches | 1747 |
| v_theta_candidate_routes_evaluated | 6780 |

# Path generator report — E19

- Seed: `120491`
- Accepted: **200**
- Attempts used: 6093
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
| complex:path_length | 1456 |
| v_complex:class_mismatch_semi_complex | 1010 |
| v_complex:class_mismatch_unclassified | 805 |
| complex:class_mismatch_semi_complex | 770 |
| v_complex:class_mismatch_complex | 580 |
| complex:class_mismatch_unclassified | 494 |
| complex:direct_line_clear | 264 |
| complex:theta_star_no_path | 133 |
| semi_complex:path_length | 102 |
| v_complex:no_promising_goal_bundle | 91 |
| complex:duplicate_endpoints | 52 |
| v_complex:duplicate_endpoints | 45 |
| semi_complex:class_mismatch_unclassified | 32 |
| straight:duplicate_endpoints | 14 |
| semi_complex:direct_line_clear | 13 |
| semi_complex:theta_star_no_path | 10 |
| semi_complex:duplicate_endpoints | 9 |
| straight:no_endpoint_pair | 5 |
| semi_complex:class_mismatch_complex | 5 |
| v_complex:path_length | 2 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2583 |
| v_exploration_start_batches | 519 |
| v_goal_candidates_evaluated | 25830 |
| v_hotspot_cells | 2169 |
| v_hotspot_refreshes | 90 |
| v_hotspot_start_batches | 2064 |
| v_theta_candidate_routes_evaluated | 7749 |

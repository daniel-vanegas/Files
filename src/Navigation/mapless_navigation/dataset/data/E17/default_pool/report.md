# Path generator report — E17

- Seed: `115281`
- Accepted: **200**
- Attempts used: 4125
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
| complex:path_length | 771 |
| v_complex:class_mismatch_semi_complex | 651 |
| complex:class_mismatch_semi_complex | 649 |
| v_complex:class_mismatch_complex | 576 |
| complex:class_mismatch_unclassified | 415 |
| v_complex:class_mismatch_unclassified | 328 |
| complex:direct_line_clear | 275 |
| semi_complex:path_length | 65 |
| complex:theta_star_no_path | 55 |
| semi_complex:class_mismatch_unclassified | 32 |
| v_complex:duplicate_endpoints | 32 |
| semi_complex:direct_line_clear | 22 |
| v_complex:no_promising_goal_bundle | 16 |
| complex:duplicate_endpoints | 14 |
| semi_complex:class_mismatch_complex | 7 |
| semi_complex:theta_star_no_path | 6 |
| straight:no_endpoint_pair | 3 |
| semi_complex:duplicate_endpoints | 3 |
| complex:class_mismatch_v_complex | 3 |
| straight:duplicate_endpoints | 1 |
| v_complex:path_length | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 1654 |
| v_exploration_start_batches | 374 |
| v_goal_candidates_evaluated | 16540 |
| v_hotspot_cells | 4166 |
| v_hotspot_refreshes | 87 |
| v_hotspot_start_batches | 1280 |
| v_theta_candidate_routes_evaluated | 4962 |

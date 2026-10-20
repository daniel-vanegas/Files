# Path generator report — E18

- Seed: `121594`
- Accepted: **200**
- Attempts used: 9220
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
| complex:path_length | 2724 |
| complex:class_mismatch_semi_complex | 1512 |
| complex:class_mismatch_unclassified | 1117 |
| v_complex:class_mismatch_semi_complex | 1077 |
| complex:direct_line_clear | 853 |
| v_complex:class_mismatch_unclassified | 649 |
| v_complex:class_mismatch_complex | 562 |
| v_complex:no_promising_goal_bundle | 181 |
| complex:theta_star_no_path | 127 |
| semi_complex:path_length | 66 |
| v_complex:duplicate_endpoints | 51 |
| semi_complex:direct_line_clear | 34 |
| semi_complex:class_mismatch_unclassified | 33 |
| complex:duplicate_endpoints | 19 |
| straight:no_endpoint_pair | 4 |
| semi_complex:duplicate_endpoints | 3 |
| straight:duplicate_endpoints | 2 |
| complex:no_endpoint_pair | 2 |
| complex:class_mismatch_v_complex | 2 |
| semi_complex:theta_star_no_path | 1 |
| semi_complex:class_mismatch_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2570 |
| v_exploration_start_batches | 518 |
| v_goal_candidates_evaluated | 25700 |
| v_hotspot_cells | 3021 |
| v_hotspot_refreshes | 73 |
| v_hotspot_start_batches | 2052 |
| v_theta_candidate_routes_evaluated | 7710 |

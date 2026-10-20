# Path generator report — E14

- Seed: `107490`
- Accepted: **200**
- Attempts used: 26989
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
| v_complex:class_mismatch_semi_complex | 6388 |
| complex:path_length | 6050 |
| v_complex:class_mismatch_unclassified | 4128 |
| complex:class_mismatch_semi_complex | 3252 |
| v_complex:class_mismatch_complex | 2716 |
| complex:class_mismatch_unclassified | 2272 |
| complex:direct_line_clear | 1396 |
| v_complex:no_promising_goal_bundle | 179 |
| v_complex:duplicate_endpoints | 138 |
| semi_complex:path_length | 125 |
| complex:duplicate_endpoints | 69 |
| semi_complex:class_mismatch_unclassified | 35 |
| semi_complex:direct_line_clear | 25 |
| semi_complex:duplicate_endpoints | 7 |
| straight:duplicate_endpoints | 4 |
| v_complex:path_length | 2 |
| straight:no_endpoint_pair | 1 |
| semi_complex:class_mismatch_complex | 1 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 13601 |
| v_exploration_start_batches | 2695 |
| v_goal_candidates_evaluated | 136010 |
| v_hotspot_cells | 3158 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 10906 |
| v_theta_candidate_routes_evaluated | 40803 |

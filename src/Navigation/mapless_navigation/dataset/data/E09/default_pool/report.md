# Path generator report — E09

- Seed: `112227`
- Accepted: **200**
- Attempts used: 3197
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
| complex:path_length | 1087 |
| complex:class_mismatch_semi_complex | 547 |
| complex:class_mismatch_unclassified | 319 |
| complex:direct_line_clear | 224 |
| v_complex:class_mismatch_complex | 214 |
| v_complex:class_mismatch_semi_complex | 143 |
| semi_complex:path_length | 104 |
| complex:duplicate_endpoints | 75 |
| v_complex:class_mismatch_unclassified | 74 |
| v_complex:duplicate_endpoints | 67 |
| semi_complex:class_mismatch_unclassified | 34 |
| semi_complex:direct_line_clear | 26 |
| v_complex:no_promising_goal_bundle | 22 |
| straight:duplicate_endpoints | 20 |
| semi_complex:class_mismatch_complex | 19 |
| semi_complex:duplicate_endpoints | 9 |
| straight:no_endpoint_pair | 8 |
| complex:class_mismatch_v_complex | 4 |
| semi_complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 570 |
| v_exploration_start_batches | 111 |
| v_goal_candidates_evaluated | 5700 |
| v_hotspot_cells | 1589 |
| v_hotspot_refreshes | 71 |
| v_hotspot_start_batches | 459 |
| v_theta_candidate_routes_evaluated | 1710 |

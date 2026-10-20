# Path generator report — E20

- Seed: `117673`
- Accepted: **200**
- Attempts used: 3419
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
| complex:path_length | 1018 |
| complex:class_mismatch_semi_complex | 579 |
| v_complex:class_mismatch_semi_complex | 402 |
| v_complex:class_mismatch_complex | 345 |
| complex:class_mismatch_unclassified | 306 |
| v_complex:class_mismatch_unclassified | 203 |
| semi_complex:path_length | 103 |
| complex:direct_line_clear | 95 |
| v_complex:duplicate_endpoints | 45 |
| straight:no_endpoint_pair | 33 |
| semi_complex:class_mismatch_unclassified | 30 |
| v_complex:no_promising_goal_bundle | 22 |
| complex:duplicate_endpoints | 12 |
| straight:duplicate_endpoints | 7 |
| semi_complex:class_mismatch_complex | 6 |
| semi_complex:direct_line_clear | 6 |
| semi_complex:duplicate_endpoints | 5 |
| complex:class_mismatch_v_complex | 2 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 1067 |
| v_exploration_start_batches | 235 |
| v_goal_candidates_evaluated | 10670 |
| v_hotspot_cells | 4002 |
| v_hotspot_refreshes | 76 |
| v_hotspot_start_batches | 832 |
| v_theta_candidate_routes_evaluated | 3201 |

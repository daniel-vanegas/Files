# Path generator report — E16

- Seed: `109535`
- Accepted: **200**
- Attempts used: 4374
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
| complex:class_mismatch_semi_complex | 871 |
| v_complex:class_mismatch_semi_complex | 750 |
| complex:path_length | 712 |
| v_complex:class_mismatch_complex | 544 |
| complex:class_mismatch_unclassified | 528 |
| v_complex:class_mismatch_unclassified | 386 |
| complex:direct_line_clear | 242 |
| semi_complex:path_length | 45 |
| semi_complex:class_mismatch_unclassified | 28 |
| v_complex:duplicate_endpoints | 16 |
| semi_complex:direct_line_clear | 14 |
| complex:duplicate_endpoints | 11 |
| straight:no_endpoint_pair | 8 |
| semi_complex:class_mismatch_complex | 6 |
| straight:duplicate_endpoints | 6 |
| v_complex:no_promising_goal_bundle | 3 |
| semi_complex:duplicate_endpoints | 2 |
| complex:class_mismatch_v_complex | 1 |
| complex:no_endpoint_pair | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 1749 |
| v_exploration_start_batches | 352 |
| v_goal_candidates_evaluated | 17490 |
| v_hotspot_cells | 5019 |
| v_hotspot_refreshes | 86 |
| v_hotspot_start_batches | 1397 |
| v_theta_candidate_routes_evaluated | 5247 |

# Path generator report — E06

- Seed: `116237`
- Accepted: **200**
- Attempts used: 10281
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
| v_complex:class_mismatch_semi_complex | 2717 |
| v_complex:class_mismatch_complex | 2233 |
| complex:path_length | 1564 |
| v_complex:class_mismatch_unclassified | 1487 |
| complex:class_mismatch_semi_complex | 1035 |
| complex:class_mismatch_unclassified | 523 |
| complex:direct_line_clear | 339 |
| semi_complex:path_length | 70 |
| v_complex:duplicate_endpoints | 30 |
| semi_complex:class_mismatch_unclassified | 25 |
| complex:duplicate_endpoints | 20 |
| semi_complex:direct_line_clear | 19 |
| v_complex:no_promising_goal_bundle | 8 |
| complex:class_mismatch_v_complex | 4 |
| v_complex:path_length | 3 |
| straight:duplicate_endpoints | 2 |
| semi_complex:duplicate_endpoints | 1 |
| semi_complex:class_mismatch_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 6528 |
| v_exploration_start_batches | 1355 |
| v_goal_candidates_evaluated | 65273 |
| v_hotspot_cells | 4586 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 5173 |
| v_theta_candidate_routes_evaluated | 19584 |

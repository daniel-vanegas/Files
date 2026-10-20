# Path generator report — E04

- Seed: `130546`
- Accepted: **200**
- Attempts used: 14002
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
| v_complex:class_mismatch_semi_complex | 3666 |
| v_complex:class_mismatch_unclassified | 2842 |
| v_complex:class_mismatch_complex | 2540 |
| complex:path_length | 1687 |
| complex:class_mismatch_semi_complex | 1394 |
| complex:class_mismatch_unclassified | 912 |
| complex:direct_line_clear | 561 |
| semi_complex:path_length | 60 |
| semi_complex:class_mismatch_unclassified | 41 |
| v_complex:duplicate_endpoints | 36 |
| semi_complex:direct_line_clear | 23 |
| complex:duplicate_endpoints | 19 |
| v_complex:path_length | 10 |
| straight:duplicate_endpoints | 6 |
| semi_complex:class_mismatch_complex | 4 |
| v_complex:no_promising_goal_bundle | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 9145 |
| v_exploration_start_batches | 1860 |
| v_goal_candidates_evaluated | 91450 |
| v_hotspot_cells | 3093 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 7285 |
| v_theta_candidate_routes_evaluated | 27435 |

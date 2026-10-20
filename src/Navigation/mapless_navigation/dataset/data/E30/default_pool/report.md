# Path generator report — E30

- Seed: `116499`
- Accepted: **200**
- Attempts used: 15377
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
| v_complex:class_mismatch_semi_complex | 5105 |
| v_complex:class_mismatch_unclassified | 2979 |
| v_complex:class_mismatch_complex | 2011 |
| complex:class_mismatch_semi_complex | 1567 |
| complex:path_length | 1242 |
| complex:class_mismatch_unclassified | 1057 |
| complex:direct_line_clear | 933 |
| semi_complex:path_length | 59 |
| semi_complex:class_mismatch_unclassified | 55 |
| semi_complex:direct_line_clear | 52 |
| v_complex:duplicate_endpoints | 52 |
| complex:duplicate_endpoints | 31 |
| v_complex:no_promising_goal_bundle | 15 |
| v_complex:path_length | 9 |
| semi_complex:duplicate_endpoints | 5 |
| semi_complex:class_mismatch_complex | 3 |
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
| v_attempts | 10221 |
| v_exploration_start_batches | 2044 |
| v_goal_candidates_evaluated | 102100 |
| v_hotspot_cells | 4461 |
| v_hotspot_refreshes | 98 |
| v_hotspot_start_batches | 8177 |
| v_theta_candidate_routes_evaluated | 30660 |

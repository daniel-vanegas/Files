# Path generator report — E21

- Seed: `119694`
- Accepted: **200**
- Attempts used: 5741
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
| complex:path_length | 1345 |
| v_complex:class_mismatch_semi_complex | 956 |
| complex:class_mismatch_semi_complex | 873 |
| v_complex:class_mismatch_complex | 708 |
| v_complex:class_mismatch_unclassified | 646 |
| complex:class_mismatch_unclassified | 517 |
| complex:direct_line_clear | 164 |
| semi_complex:path_length | 95 |
| complex:theta_star_no_path | 61 |
| v_complex:duplicate_endpoints | 43 |
| complex:duplicate_endpoints | 30 |
| semi_complex:class_mismatch_unclassified | 26 |
| v_complex:no_promising_goal_bundle | 21 |
| straight:no_endpoint_pair | 18 |
| straight:duplicate_endpoints | 15 |
| semi_complex:class_mismatch_complex | 9 |
| semi_complex:direct_line_clear | 5 |
| semi_complex:duplicate_endpoints | 4 |
| semi_complex:theta_star_no_path | 3 |
| v_complex:path_length | 1 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2425 |
| v_exploration_start_batches | 479 |
| v_goal_candidates_evaluated | 24250 |
| v_hotspot_cells | 3562 |
| v_hotspot_refreshes | 88 |
| v_hotspot_start_batches | 1946 |
| v_theta_candidate_routes_evaluated | 7275 |

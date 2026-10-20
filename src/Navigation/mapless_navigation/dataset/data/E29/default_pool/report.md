# Path generator report — E29

- Seed: `131074`
- Accepted: **178**
- Attempts used: 19052
- Attempt limit: unlimited; reached: False
- Time limit: 900.0 s; timeout hit: True
- Stop reason: **timeout**
- Complete: **False**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 50 | 50 | 0 |
| semi_complex | 50 | 50 | 0 |
| complex | 46 | 50 | 4 |
| v_complex | 32 | 50 | 18 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 46 |
| multiple_obstacle_navigation | 32 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 4717 |
| complex:path_length | 4080 |
| v_complex:class_mismatch_unclassified | 2977 |
| complex:class_mismatch_semi_complex | 2397 |
| complex:class_mismatch_unclassified | 1804 |
| v_complex:class_mismatch_complex | 1405 |
| complex:direct_line_clear | 994 |
| v_complex:no_promising_goal_bundle | 234 |
| semi_complex:path_length | 78 |
| complex:duplicate_endpoints | 51 |
| semi_complex:class_mismatch_unclassified | 32 |
| complex:theta_star_no_path | 25 |
| semi_complex:direct_line_clear | 21 |
| v_complex:duplicate_endpoints | 19 |
| v_complex:path_length | 17 |
| straight:duplicate_endpoints | 7 |
| semi_complex:duplicate_endpoints | 5 |
| complex:class_mismatch_v_complex | 4 |
| semi_complex:theta_star_no_path | 3 |
| straight:no_endpoint_pair | 3 |
| semi_complex:class_mismatch_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 9401 |
| v_exploration_start_batches | 1878 |
| v_goal_candidates_evaluated | 94010 |
| v_hotspot_cells | 3332 |
| v_hotspot_refreshes | 79 |
| v_hotspot_start_batches | 7523 |
| v_theta_candidate_routes_evaluated | 28203 |

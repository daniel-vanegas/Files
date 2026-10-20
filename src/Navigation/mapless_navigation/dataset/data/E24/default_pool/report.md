# Path generator report — E24

- Seed: `129902`
- Accepted: **176**
- Attempts used: 10773
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
| complex | 47 | 50 | 3 |
| v_complex | 29 | 50 | 21 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 47 |
| multiple_obstacle_navigation | 29 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 2477 |
| complex:path_length | 2034 |
| v_complex:class_mismatch_unclassified | 1326 |
| complex:class_mismatch_semi_complex | 1260 |
| complex:class_mismatch_unclassified | 1035 |
| v_complex:class_mismatch_complex | 922 |
| complex:direct_line_clear | 631 |
| v_complex:no_promising_goal_bundle | 440 |
| complex:theta_star_no_path | 192 |
| semi_complex:path_length | 99 |
| semi_complex:class_mismatch_unclassified | 49 |
| semi_complex:direct_line_clear | 35 |
| v_complex:duplicate_endpoints | 33 |
| complex:duplicate_endpoints | 29 |
| semi_complex:theta_star_no_path | 11 |
| semi_complex:class_mismatch_complex | 9 |
| straight:duplicate_endpoints | 5 |
| semi_complex:duplicate_endpoints | 4 |
| straight:no_endpoint_pair | 3 |
| v_complex:path_length | 2 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 5229 |
| v_exploration_start_batches | 1119 |
| v_goal_candidates_evaluated | 52280 |
| v_hotspot_cells | 3187 |
| v_hotspot_refreshes | 77 |
| v_hotspot_start_batches | 4110 |
| v_theta_candidate_routes_evaluated | 15687 |

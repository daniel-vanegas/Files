# Path generator report — E13

- Seed: `131918`
- Accepted: **200**
- Attempts used: 8434
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
| v_complex:class_mismatch_semi_complex | 2067 |
| v_complex:class_mismatch_unclassified | 1602 |
| v_complex:class_mismatch_complex | 1503 |
| complex:path_length | 1199 |
| complex:class_mismatch_semi_complex | 834 |
| complex:class_mismatch_unclassified | 525 |
| complex:direct_line_clear | 260 |
| semi_complex:path_length | 85 |
| v_complex:duplicate_endpoints | 60 |
| semi_complex:class_mismatch_unclassified | 26 |
| v_complex:no_promising_goal_bundle | 18 |
| semi_complex:direct_line_clear | 15 |
| complex:theta_star_no_path | 15 |
| complex:duplicate_endpoints | 10 |
| straight:duplicate_endpoints | 5 |
| semi_complex:class_mismatch_complex | 4 |
| semi_complex:duplicate_endpoints | 2 |
| complex:class_mismatch_v_complex | 2 |
| semi_complex:theta_star_no_path | 1 |
| v_complex:path_length | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 5301 |
| v_exploration_start_batches | 1063 |
| v_goal_candidates_evaluated | 53010 |
| v_hotspot_cells | 3932 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 4238 |
| v_theta_candidate_routes_evaluated | 15903 |

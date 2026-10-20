# Path generator report — E03

- Seed: `113846`
- Accepted: **200**
- Attempts used: 5072
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
| complex:path_length | 945 |
| v_complex:class_mismatch_semi_complex | 825 |
| complex:class_mismatch_semi_complex | 750 |
| v_complex:class_mismatch_complex | 621 |
| complex:class_mismatch_unclassified | 512 |
| complex:direct_line_clear | 484 |
| v_complex:class_mismatch_unclassified | 474 |
| semi_complex:path_length | 77 |
| v_complex:duplicate_endpoints | 47 |
| semi_complex:class_mismatch_unclassified | 30 |
| complex:theta_star_no_path | 28 |
| semi_complex:direct_line_clear | 27 |
| complex:duplicate_endpoints | 19 |
| v_complex:no_promising_goal_bundle | 17 |
| semi_complex:class_mismatch_complex | 6 |
| semi_complex:duplicate_endpoints | 3 |
| complex:class_mismatch_v_complex | 3 |
| v_complex:path_length | 2 |
| straight:duplicate_endpoints | 1 |
| complex:no_endpoint_pair | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 2036 |
| v_exploration_start_batches | 418 |
| v_goal_candidates_evaluated | 20360 |
| v_hotspot_cells | 4230 |
| v_hotspot_refreshes | 88 |
| v_hotspot_start_batches | 1618 |
| v_theta_candidate_routes_evaluated | 6108 |

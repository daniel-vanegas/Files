# Path generator report — E05

- Seed: `132607`
- Accepted: **157**
- Attempts used: 20573
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
| complex | 41 | 50 | 9 |
| v_complex | 16 | 50 | 34 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 41 |
| multiple_obstacle_navigation | 16 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 5177 |
| complex:path_length | 3714 |
| v_complex:class_mismatch_unclassified | 3167 |
| complex:direct_line_clear | 2956 |
| complex:class_mismatch_semi_complex | 1852 |
| v_complex:class_mismatch_complex | 1563 |
| complex:class_mismatch_unclassified | 1468 |
| v_complex:no_promising_goal_bundle | 173 |
| semi_complex:path_length | 119 |
| semi_complex:direct_line_clear | 97 |
| semi_complex:class_mismatch_unclassified | 45 |
| complex:no_endpoint_pair | 41 |
| complex:duplicate_endpoints | 28 |
| v_complex:duplicate_endpoints | 5 |
| semi_complex:no_endpoint_pair | 4 |
| semi_complex:class_mismatch_complex | 3 |
| semi_complex:duplicate_endpoints | 3 |
| complex:class_mismatch_v_complex | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 10101 |
| v_exploration_start_batches | 2165 |
| v_goal_candidates_evaluated | 98731 |
| v_hotspot_cells | 2562 |
| v_hotspot_refreshes | 58 |
| v_hotspot_start_batches | 7936 |
| v_theta_candidate_routes_evaluated | 29981 |

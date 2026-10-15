# Path generator report — E03

- Seed: `2026101403`
- Accepted: **4**
- Attempts used: 162
- Attempt limit: 25000; reached: False
- Time limit: 900.0 s; timeout hit: False
- Stop reason: **complete**
- Complete: **True**
- Hard clearance: 0.350 m

## Geometry distribution

| Class | Count | Target | Deficit |
|---|---:|---:|---:|
| straight | 1 | 1 | 0 |
| semi_complex | 1 | 1 | 0 |
| complex | 1 | 1 | 0 |
| v_complex | 1 | 1 | 0 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 1 |
| constrained_passage | 1 |
| multiple_obstacle_navigation | 1 |
| single_obstacle_avoidance | 1 |

## Rejections

| Reason | Count |
|---|---:|
| complex:class_mismatch_semi_complex | 34 |
| complex:path_length | 32 |
| complex:class_mismatch_unclassified | 24 |
| v_complex:class_mismatch_unclassified | 19 |
| complex:direct_line_clear | 19 |
| v_complex:class_mismatch_semi_complex | 18 |
| v_complex:class_mismatch_complex | 11 |
| complex:theta_star_no_path | 1 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 49 |
| v_exploration_start_batches | 49 |
| v_goal_candidates_evaluated | 490 |
| v_hotspot_cells | 0 |
| v_hotspot_refreshes | 1 |
| v_hotspot_start_batches | 0 |
| v_theta_candidate_routes_evaluated | 147 |

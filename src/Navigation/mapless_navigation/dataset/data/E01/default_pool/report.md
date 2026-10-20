# Path generator report — E01

- Seed: `126341`
- Accepted: **199**
- Attempts used: 36904
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
| complex | 50 | 50 | 0 |
| v_complex | 49 | 50 | 1 |

## Situation distribution

| Task | Count |
|---|---:|
| clear_travel | 50 |
| constrained_passage | 50 |
| multiple_obstacle_navigation | 49 |
| single_obstacle_avoidance | 50 |

## Rejections

| Reason | Count |
|---|---:|
| v_complex:class_mismatch_semi_complex | 10558 |
| v_complex:class_mismatch_unclassified | 7231 |
| v_complex:class_mismatch_complex | 4530 |
| complex:path_length | 4000 |
| complex:class_mismatch_semi_complex | 3984 |
| complex:direct_line_clear | 3116 |
| complex:class_mismatch_unclassified | 2882 |
| v_complex:duplicate_endpoints | 155 |
| complex:duplicate_endpoints | 87 |
| semi_complex:path_length | 59 |
| semi_complex:class_mismatch_unclassified | 40 |
| semi_complex:direct_line_clear | 38 |
| semi_complex:duplicate_endpoints | 8 |
| v_complex:path_length | 6 |
| v_complex:no_promising_goal_bundle | 6 |
| straight:duplicate_endpoints | 5 |

## Files

- `path_pool.json`: pool metadata and route attributes.
- `path_pool_poses.json`: route coordinates keyed by ID.
- `path_pool.csv`: flat metrics table.
- `routes_<class>.png`: one route per panel; not an overlay of every route.

## V-complex search diagnostics

| Counter | Value |
|---|---:|
| v_attempts | 22535 |
| v_exploration_start_batches | 4611 |
| v_goal_candidates_evaluated | 224549 |
| v_hotspot_cells | 2321 |
| v_hotspot_refreshes | 100 |
| v_hotspot_start_batches | 17924 |
| v_theta_candidate_routes_evaluated | 67596 |

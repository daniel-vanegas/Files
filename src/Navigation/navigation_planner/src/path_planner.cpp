#include "navigation_planner/path_planner.hpp"

#include <cmath>
#include <cstdio>
#include <cmath>

namespace navigation_planner {

bool PathPlanner::AStar(const nav_msgs::msg::OccupancyGrid& map,
                        const nav_msgs::msg::OccupancyGrid& cost_map,
                        const geometry_msgs::msg::Pose& start_pose,
                        const geometry_msgs::msg::Pose& goal_pose,
                        bool diagonal_paths,
                        nav_msgs::msg::Path& result_path)
{
    const int W = (int)map.info.width;
    const int H = (int)map.info.height;
    const float res = (float)map.info.resolution;
    const float ox = (float)map.info.origin.position.x;
    const float oy = (float)map.info.origin.position.y;

    int idx_start_x = (int)((start_pose.position.x - ox) / res);
    int idx_start_y = (int)((start_pose.position.y - oy) / res);
    int idx_goal_x  = (int)((goal_pose.position.x  - ox) / res);
    int idx_goal_y  = (int)((goal_pose.position.y  - oy) / res);

    // Bounds check (was missing in ROS1).
    if (idx_start_x < 0 || idx_start_x >= W || idx_start_y < 0 || idx_start_y >= H ||
        idx_goal_x  < 0 || idx_goal_x  >= W || idx_goal_y  < 0 || idx_goal_y  >= H) {
        std::printf("PathPlanner.->start or goal outside map bounds\n");
        return false;
    }

    int idx_start = idx_start_y * W + idx_start_x;
    int idx_goal  = idx_goal_y  * W + idx_goal_x;

    if (map.data[idx_goal] != 0) {
        std::printf("PathPlanner.->Goal point is inside non-free space\n");
        return false;
    }
    if (map.data[idx_start] != 0) {
        std::printf("PathPlanner.->Start point is inside non-free space\n");
        return false;
    }

    std::vector<Node> nodes((size_t)W * H);
    for (size_t i = 0; i < nodes.size(); i++) nodes[i].index = (int)i;

    std::vector<int> neighbors(diagonal_paths ? 8 : 4);
    std::priority_queue<Node*, std::vector<Node*>, CompareByFValue> open_list;

    Node* current = &nodes[idx_start];
    current->g_value = 0.0f;
    current->f_value = 0.0f;
    current->in_open_list = true;
    open_list.push(current);

    int steps = 0;
    while (!open_list.empty() && current->index != idx_goal) {
        current = open_list.top();
        open_list.pop();
        current->in_closed_list = true;

        const int cx = current->index % W;
        const int cy = current->index / W;

        neighbors[0] = (cy + 1 < H) ? current->index + W : -1;
        neighbors[1] = (cx + 1 < W) ? current->index + 1 : -1;
        neighbors[2] = (cy - 1 >= 0) ? current->index - W : -1;
        neighbors[3] = (cx - 1 >= 0) ? current->index - 1 : -1;
        if (diagonal_paths) {
            neighbors[4] = (cx + 1 < W && cy + 1 < H) ? current->index + W + 1 : -1;
            neighbors[5] = (cx - 1 >= 0 && cy + 1 < H) ? current->index + W - 1 : -1;
            neighbors[6] = (cx + 1 < W && cy - 1 >= 0) ? current->index - W + 1 : -1;
            neighbors[7] = (cx - 1 >= 0 && cy - 1 >= 0) ? current->index - W - 1 : -1;
        }

        for (size_t i = 0; i < neighbors.size(); i++) {
            int n_idx = neighbors[i];
            if (n_idx < 0) continue;
            if (map.data[n_idx] != 0) continue;
            Node* neighbor = &nodes[n_idx];
            if (neighbor->in_closed_list) continue;

            float step_cost = (i < 4) ? 1.0f : 1.4142135f;
            float g = current->g_value + step_cost + (float)cost_map.data[n_idx];
            int hx = (n_idx % W) - idx_goal_x;
            int hy = (n_idx / W) - idx_goal_y;
            float h = diagonal_paths
                ? std::sqrt((float)(hx * hx + hy * hy))
                : (float)(std::abs(hx) + std::abs(hy));

            if (g < neighbor->g_value) {
                neighbor->g_value = g;
                neighbor->f_value = g + h;
                neighbor->parent  = current;
            }
            if (!neighbor->in_open_list) {
                neighbor->in_open_list = true;
                open_list.push(neighbor);
            }
        }
        steps++;
    }

    std::printf("PathPlanner.->A* ended after %d steps\n", steps);

    if (current->index != idx_goal) return false;

    result_path.header.frame_id = "map";
    result_path.poses.clear();
    geometry_msgs::msg::PoseStamped p;
    p.header.frame_id = "map";
    while (current->parent != nullptr) {
        p.pose.position.x = (current->index % W) * res + ox;
        p.pose.position.y = (current->index / W) * res + oy;
        result_path.poses.insert(result_path.poses.begin(), p);
        current = current->parent;
    }
    std::printf("PathPlanner.->Result path has %zu points\n", result_path.poses.size());
    return true;
}

nav_msgs::msg::Path PathPlanner::SmoothPath(const nav_msgs::msg::Path& path,
                                            float weight_data,
                                            float weight_smooth,
                                            float tolerance)
{
    nav_msgs::msg::Path newPath;
    newPath.header.frame_id = "map";
    for (size_t i = 0; i < path.poses.size(); i++)
        newPath.poses.push_back(path.poses[i]);

    if (path.poses.size() < 3) return newPath;

    int attempts = 0;
    float tol = tolerance * (float)path.poses.size();
    float change = tol + 1.0f;

    while (change >= tol && ++attempts < 1000) {
        change = 0.0f;
        for (size_t i = 1; i + 1 < path.poses.size(); i++) {
            auto old_p  = path.poses[i].pose.position;
            auto new_p  = newPath.poses[i].pose.position;
            auto next_p = newPath.poses[i + 1].pose.position;
            auto prev_p = newPath.poses[i - 1].pose.position;

            float last_x = (float)newPath.poses[i].pose.position.x;
            float last_y = (float)newPath.poses[i].pose.position.y;

            new_p.x += weight_data * (old_p.x - new_p.x) +
                       weight_smooth * (next_p.x + prev_p.x - 2.0 * new_p.x);
            new_p.y += weight_data * (old_p.y - new_p.y) +
                       weight_smooth * (next_p.y + prev_p.y - 2.0 * new_p.y);

            change += std::fabs((float)new_p.x - last_x) +
                      std::fabs((float)new_p.y - last_y);
            newPath.poses[i].pose.position = new_p;
        }
    }
    std::printf("PathPlanner.->Smoothing finished after %d attempts\n", attempts);
    return newPath;
}

nav_msgs::msg::Path PathPlanner::ChaikinSmooth(const nav_msgs::msg::Path& path,
                                               int iterations)
{
    if (path.poses.size() < 3) return path;

    nav_msgs::msg::Path current = path;
    for (int it = 0; it < iterations; it++) {
        if (current.poses.size() < 3) break;

        nav_msgs::msg::Path next;
        next.header = current.header;
        next.poses.push_back(current.poses.front());

        for (size_t i = 0; i + 1 < current.poses.size(); i++) {
            const auto& p0 = current.poses[i].pose.position;
            const auto& p1 = current.poses[i + 1].pose.position;

            geometry_msgs::msg::PoseStamped a, b;
            a.header = current.header;
            b.header = current.header;

            a.pose.position.x = 0.75 * p0.x + 0.25 * p1.x;
            a.pose.position.y = 0.75 * p0.y + 0.25 * p1.y;
            b.pose.position.x = 0.25 * p0.x + 0.75 * p1.x;
            b.pose.position.y = 0.25 * p0.y + 0.75 * p1.y;

            next.poses.push_back(a);
            next.poses.push_back(b);
        }
        next.poses.push_back(current.poses.back());
        current = std::move(next);
    }
    return current;
}

nav_msgs::msg::Path PathPlanner::downsample(const nav_msgs::msg::Path& in,
                                            float min_dist)
{
    if (in.poses.size() < 2) return in;
    nav_msgs::msg::Path out;
    out.header = in.header;
    out.poses.push_back(in.poses.front());
    for (size_t i = 1; i < in.poses.size(); i++) {
        const auto& p = in.poses[i].pose.position;
        const auto& q = out.poses.back().pose.position;
        if (std::hypot(p.x - q.x, p.y - q.y) >= min_dist)
            out.poses.push_back(in.poses[i]);
    }
    const auto& last = in.poses.back().pose.position;
    const auto& cur  = out.poses.back().pose.position;
    if (last.x != cur.x || last.y != cur.y)
        out.poses.push_back(in.poses.back());
    return out;
}

} // namespace navigation_planner
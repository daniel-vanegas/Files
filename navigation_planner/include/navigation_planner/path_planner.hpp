#ifndef NAVIGATION_PLANNER_PATH_PLANNER_HPP
#define NAVIGATION_PLANNER_PATH_PLANNER_HPP

#include <limits>
#include <queue>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>

namespace navigation_planner {

class Node
{
public:
    Node() = default;
    int index = -1;
    float g_value = std::numeric_limits<float>::max();
    float f_value = std::numeric_limits<float>::max();
    bool in_open_list = false;
    bool in_closed_list = false;
    Node* parent = nullptr;
};

struct CompareByFValue
{
    bool operator()(const Node* a, const Node* b) const
    {
        return a->f_value > b->f_value;
    }
};

class PathPlanner
{
public:
    static bool AStar(const nav_msgs::msg::OccupancyGrid& map,
                      const nav_msgs::msg::OccupancyGrid& cost_map,
                      const geometry_msgs::msg::Pose& start_pose,
                      const geometry_msgs::msg::Pose& goal_pose,
                      bool diagonal_paths,
                      nav_msgs::msg::Path& result_path);

    static nav_msgs::msg::Path SmoothPath(const nav_msgs::msg::Path& path,
                                          float weight_data = 0.1f,
                                          float weight_smooth = 0.9f,
                                          float tolerance = 0.00001f);
                                          
    static nav_msgs::msg::Path ChaikinSmooth(const nav_msgs::msg::Path& path,
                                             int iterations = 2);

    static nav_msgs::msg::Path downsample(const nav_msgs::msg::Path& in,
                                          float min_dist = 0.10f);                                             
};

} // namespace navigation_planner

#endif
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>

#include "debugger/debug_client.hpp"

using namespace std::chrono_literals;

class MapAugmenterNode : public rclcpp::Node
{
public:
    MapAugmenterNode() : rclcpp::Node("map_augmenter")
    {
        declare_parameters();
        load_parameters();

        tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        augmented_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/global_costmap/costmap", rclcpp::QoS(10));

        costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/navigation/augmented_cost_map", rclcpp::QoS(1).transient_local());

        auto latched_qos = rclcpp::QoS(1).transient_local().reliable();
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic_, latched_qos,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                latest_static_map_ = msg;
                map_received_ = true;
            });
        prohibition_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            prohibition_topic_, latched_qos,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                latest_prohibition_map_ = msg;
                prohibition_received_ = true;
            });

        if (use_lidar_) {
            scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                laser_scan_topic_, rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::LaserScan::SharedPtr msg) { last_scan_ = msg; });
        }
        if (use_cloud_) {
            cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                cloud_topic_, rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { last_cloud_ = msg; });
        }

        setup_services();
        setup_debug();

        decay_timer_ = this->create_wall_timer(1s, [this]() { decay_and_publish(); });
    }

    void wait_for_maps()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for /map and prohibition map...");
        rclcpp::Rate rate(10.0);
        while (rclcpp::ok() && (!map_received_ || !prohibition_received_)) {
            rclcpp::spin_some(this->get_node_base_interface());
            rate.sleep();
        }
        if (!rclcpp::ok()) return;

        RCLCPP_INFO(this->get_logger(), "Maps received. Building static layers...");
        build_static_layers();
        RCLCPP_INFO(this->get_logger(), "Static layers ready.");
    }

private:
    void declare_parameters()
    {
        this->declare_parameter("use_lidar", true);
        this->declare_parameter("use_point_cloud", false);
        this->declare_parameter("laser_scan_topic", std::string("/hardware/base/scan"));
        this->declare_parameter("point_cloud_topic", std::string("/hardware/camera/points"));
        this->declare_parameter("map_topic", std::string("/map"));
        this->declare_parameter("prohibition_map_topic", std::string("/navigation/prohibition_map"));
        this->declare_parameter("base_link_name", std::string("base_link"));
        this->declare_parameter("min_x", 0.3);
        this->declare_parameter("max_x", 0.8);
        this->declare_parameter("min_y", -0.25);
        this->declare_parameter("max_y", 0.25);
        this->declare_parameter("min_z", 0.05);
        this->declare_parameter("max_z", 1.30);
        this->declare_parameter("inflation_radius", 0.2);
        this->declare_parameter("cost_radius", 0.5);
        this->declare_parameter("decay_factor", 10);
        this->declare_parameter("cloud_downsampling", 9);
        this->declare_parameter("lidar_downsampling", 2);
        this->declare_parameter("debug", false);
    }

    void load_parameters()
    {
        use_lidar_         = this->get_parameter("use_lidar").as_bool();
        use_cloud_         = this->get_parameter("use_point_cloud").as_bool();
        laser_scan_topic_  = this->get_parameter("laser_scan_topic").as_string();
        cloud_topic_       = this->get_parameter("point_cloud_topic").as_string();
        map_topic_         = this->get_parameter("map_topic").as_string();
        prohibition_topic_ = this->get_parameter("prohibition_map_topic").as_string();
        base_link_         = this->get_parameter("base_link_name").as_string();
        min_x_ = (float)this->get_parameter("min_x").as_double();
        max_x_ = (float)this->get_parameter("max_x").as_double();
        min_y_ = (float)this->get_parameter("min_y").as_double();
        max_y_ = (float)this->get_parameter("max_y").as_double();
        min_z_ = (float)this->get_parameter("min_z").as_double();
        max_z_ = (float)this->get_parameter("max_z").as_double();
        inflation_radius_  = (float)this->get_parameter("inflation_radius").as_double();
        cost_radius_       = (float)this->get_parameter("cost_radius").as_double();
        decay_factor_      = this->get_parameter("decay_factor").as_int();
        cloud_downsampling_ = this->get_parameter("cloud_downsampling").as_int();
        lidar_downsampling_ = this->get_parameter("lidar_downsampling").as_int();
        debug_ = this->get_parameter("debug").as_bool();

        RCLCPP_INFO(this->get_logger(),
            "Params: inflation=%.3f cost_radius=%.3f decay=%d base=%s",
            inflation_radius_, cost_radius_, decay_factor_, base_link_.c_str());
    }

    void setup_debug()
    {
        dbg_ = std::make_unique<debugger::DebugClient>(this, true);
        dbg_->send("Map augmenter ready", "NODE_STATUS", "info", -1.0f);
        heartbeat_timer_ = this->create_wall_timer(1s, [this]() {
            dbg_->send("Node initialized and ready.", "NODE_STATUS", "info", -1.0f);
        });
    }

    void setup_services()
    {
        srv_static_map_ = this->create_service<nav_msgs::srv::GetMap>(
            "/mapless_nav/map_augmenter/get_static_map",
            [this](const std::shared_ptr<nav_msgs::srv::GetMap::Request>,
                   std::shared_ptr<nav_msgs::srv::GetMap::Response> resp) {
                resp->map = static_map_;
            });

        srv_static_cost_map_ = this->create_service<nav_msgs::srv::GetMap>(
            "/mapless_nav/map_augmenter/get_static_cost_map",
            [this](const std::shared_ptr<nav_msgs::srv::GetMap::Request>,
                   std::shared_ptr<nav_msgs::srv::GetMap::Response> resp) {
                resp->map = static_cost_map_;
            });

        srv_augmented_map_ = this->create_service<nav_msgs::srv::GetMap>(
            "/mapless_nav/map_augmenter/get_augmented_map",
            [this](const std::shared_ptr<nav_msgs::srv::GetMap::Request>,
                   std::shared_ptr<nav_msgs::srv::GetMap::Response> resp) {
                if (use_lidar_ && last_scan_) read_scan_into_obstacles();
                if (use_cloud_ && last_cloud_) read_cloud_into_obstacles();
                obstacles_inflated_map_ = inflate_map(obstacles_map_, inflation_radius_);
                augmented_map_ = merge_maps(static_map_, obstacles_inflated_map_);
                resp->map = augmented_map_;
            });

        srv_augmented_cost_map_ = this->create_service<nav_msgs::srv::GetMap>(
            "/mapless_nav/map_augmenter/get_augmented_cost_map",
            [this](const std::shared_ptr<nav_msgs::srv::GetMap::Request>,
                   std::shared_ptr<nav_msgs::srv::GetMap::Response> resp) {
                nav_msgs::msg::OccupancyGrid obs_cost = get_cost_map(obstacles_inflated_map_, cost_radius_);
                resp->map = merge_maps(static_cost_map_, obs_cost);
            });

        srv_are_there_obstacles_ = this->create_service<std_srvs::srv::Trigger>(
            "/mapless_nav/map_augmenter/are_there_obstacles",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
                resp->success = are_there_obstacles_;
            });

        srv_is_inside_obstacles_ = this->create_service<std_srvs::srv::Trigger>(
            "/mapless_nav/map_augmenter/is_inside_obstacles",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
                float rx, ry, rt;
                if (!get_robot_pose(rx, ry, rt)) { resp->success = false; return; }
                int cell_x = (int)((rx - augmented_map_.info.origin.position.x) / augmented_map_.info.resolution);
                int cell_y = (int)((ry - augmented_map_.info.origin.position.y) / augmented_map_.info.resolution);
                int cell   = cell_y * augmented_map_.info.width + cell_x;
                if (cell < 0 || cell >= (int)augmented_map_.data.size()) { resp->success = false; return; }
                resp->success = augmented_map_.data[cell] > 0;
            });
    }

    void build_static_layers()
    {
        if (!latest_static_map_ || !latest_prohibition_map_) {
            RCLCPP_ERROR(this->get_logger(), "Cannot build static layers: maps missing");
            return;
        }

        static_map_ = merge_maps(*latest_static_map_, *latest_prohibition_map_);
        static_map_ = inflate_map(static_map_, inflation_radius_);
        static_cost_map_ = get_cost_map(static_map_, cost_radius_);

        obstacles_map_ = static_map_;
        std::fill(obstacles_map_.data.begin(), obstacles_map_.data.end(), 0);
        obstacles_inflated_map_ = obstacles_map_;
        augmented_map_ = static_map_;

        RCLCPP_INFO(this->get_logger(),
            "Static layers ready (%ux%u, res=%.3f)",
            static_map_.info.width, static_map_.info.height,
            static_map_.info.resolution);
    }

    bool get_robot_pose(float& x, float& y, float& t)
    {
        try {
            auto tf = tf_buffer_->lookupTransform("map", base_link_, tf2::TimePointZero);
            x = (float)tf.transform.translation.x;
            y = (float)tf.transform.translation.y;
            tf2::Quaternion q(tf.transform.rotation.x, tf.transform.rotation.y,
                              tf.transform.rotation.z, tf.transform.rotation.w);
            t = (float)q.getAngle() * (q.z() < 0 ? -1.0f : 1.0f);
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF map->%s not available: %s", base_link_.c_str(), e.what());
            return false;
        }
    }

    Eigen::Affine3d tf_to_eigen(const std::string& target, const std::string& source)
    {
        auto tf = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);
        return tf2::transformToEigen(tf);
    }

    void read_scan_into_obstacles()
    {
        if (!last_scan_) return;
        const auto& scan = *last_scan_;
        if (scan.ranges.empty()) return;

        Eigen::Affine3d lidar_to_robot;
        Eigen::Affine3d robot_to_map;
        try {
            lidar_to_robot = tf_to_eigen(base_link_, scan.header.frame_id);
            robot_to_map   = tf_to_eigen("map", base_link_);
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Scan TF unavailable: %s", e.what());
            return;
        }

        for (size_t i = 0; i < scan.ranges.size(); i += lidar_downsampling_) {
            float a = scan.angle_min + i * scan.angle_increment;
            Eigen::Vector3d v(scan.ranges[i] * std::cos(a),
                              scan.ranges[i] * std::sin(a), 0.0);
            v = lidar_to_robot * v;
            if (v.x() > min_x_ && v.x() < max_x_ &&
                v.y() > min_y_ && v.y() < max_y_ &&
                v.z() > min_z_ && v.z() < max_z_) {
                v = robot_to_map * v;
                int cx = (int)((v.x() - obstacles_map_.info.origin.position.x) / obstacles_map_.info.resolution);
                int cy = (int)((v.y() - obstacles_map_.info.origin.position.y) / obstacles_map_.info.resolution);
                int c  = cy * obstacles_map_.info.width + cx;
                if (c >= 0 && c < (int)obstacles_map_.data.size()) {
                    obstacles_map_.data[c] = 100;
                    are_there_obstacles_ = true;
                }
            }
        }
    }

    void read_cloud_into_obstacles()
    {
        if (!last_cloud_) return;
        const auto& pc = *last_cloud_;
        if (pc.data.empty()) return;

        Eigen::Affine3d cam_to_robot;
        Eigen::Affine3d robot_to_map;
        try {
            cam_to_robot = tf_to_eigen(base_link_, pc.header.frame_id);
            robot_to_map = tf_to_eigen("map", base_link_);
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Cloud TF unavailable: %s", e.what());
            return;
        }

        const uint8_t* p = pc.data.data();
        const size_t n_points = (size_t)pc.width * pc.height;
        for (size_t i = 0; i < n_points; i += cloud_downsampling_) {
            float px, py, pz;
            std::memcpy(&px, p, sizeof(float));
            std::memcpy(&py, p + 4, sizeof(float));
            std::memcpy(&pz, p + 8, sizeof(float));
            Eigen::Vector3d v(px, py, pz);
            v = cam_to_robot * v;
            if (v.x() > min_x_ && v.x() < max_x_ &&
                v.y() > min_y_ && v.y() < max_y_ &&
                v.z() > min_z_ && v.z() < max_z_) {
                v = robot_to_map * v;
                int cx = (int)((v.x() - obstacles_map_.info.origin.position.x) / obstacles_map_.info.resolution);
                int cy = (int)((v.y() - obstacles_map_.info.origin.position.y) / obstacles_map_.info.resolution);
                int c  = cy * obstacles_map_.info.width + cx;
                if (c >= 0 && c < (int)obstacles_map_.data.size()) {
                    obstacles_map_.data[c] = 100;
                    are_there_obstacles_ = true;
                }
            }
            p += cloud_downsampling_ * pc.point_step;
        }
    }

    nav_msgs::msg::OccupancyGrid merge_maps(const nav_msgs::msg::OccupancyGrid& a,
                                            const nav_msgs::msg::OccupancyGrid& b)
    {
        if (a.info.width != b.info.width || a.info.height != b.info.height) {
            RCLCPP_WARN(this->get_logger(), "Cannot merge maps of different sizes");
            return a;
        }
        nav_msgs::msg::OccupancyGrid c = a;
        for (size_t i = 0; i < c.data.size(); i++)
            c.data[i] = (int8_t)std::max((unsigned char)a.data[i], (unsigned char)b.data[i]);
        return c;
    }

    nav_msgs::msg::OccupancyGrid inflate_map(const nav_msgs::msg::OccupancyGrid& map, float inflation)
    {
        if (inflation <= 0) return map;
        nav_msgs::msg::OccupancyGrid newMap = map;
        int n = (int)(inflation / map.info.resolution);
        int lower_limit = n * (int)map.info.width + n;
        int upper_limit = (int)map.data.size() - n * (int)map.info.width - n;
        for (int k = lower_limit; k < upper_limit; k++)
            if (map.data[k] > 0)
                for (int i = -n; i <= n; i++)
                    for (int j = -n; j <= n; j++)
                        newMap.data[k + j * (int)map.info.width + i] = map.data[k];
        return newMap;
    }

    nav_msgs::msg::OccupancyGrid get_cost_map(const nav_msgs::msg::OccupancyGrid& map, float cost_radius)
    {
        if (cost_radius < 0) return map;
        nav_msgs::msg::OccupancyGrid cost_map = map;
        int steps = (int)(cost_radius / map.info.resolution);
        int boxSize = (steps * 2 + 1) * (steps * 2 + 1);
        std::vector<int> cell_costs(boxSize);
        std::vector<int> neighbors(boxSize);
        int startIdx = steps * (int)map.info.width + steps;
        int endIdx   = (int)map.data.size() - steps * (int)map.info.width - steps;

        int counter = 0;
        for (int i = -steps; i <= steps; i++)
            for (int j = -steps; j <= steps; j++) {
                neighbors[counter]  = i * (int)map.info.width + j;
                cell_costs[counter] = (steps - std::max(std::abs(i), std::abs(j)) + 1) * 2;
                counter++;
            }

        for (int i = startIdx; i < endIdx; i++)
            if (map.data[i] > 0)
                for (int j = 0; j < boxSize; j++)
                    if (cost_map.data[i + neighbors[j]] < cell_costs[j])
                        cost_map.data[i + neighbors[j]] = (int8_t)cell_costs[j];

        return cost_map;
    }

    bool decay_and_check()
    {
        bool obs = false;
        for (auto& v : obstacles_map_.data) {
            int val = (int)v - decay_factor_;
            if (val < 0) val = 0;
            v = (int8_t)val;
            obs |= (v > 0);
        }
        return obs;
    }

    void decay_and_publish()
    {
        are_there_obstacles_ = decay_and_check();
        obstacles_inflated_map_ = inflate_map(obstacles_map_, inflation_radius_);
        augmented_map_ = merge_maps(static_map_, obstacles_inflated_map_);
        augmented_pub_->publish(augmented_map_);

        nav_msgs::msg::OccupancyGrid obs_cost = get_cost_map(obstacles_inflated_map_, cost_radius_);
        nav_msgs::msg::OccupancyGrid aug_cost = merge_maps(static_cost_map_, obs_cost);
        aug_cost.header = augmented_map_.header;
        costmap_pub_->publish(aug_cost);
    }

    // Params
    bool use_lidar_, use_cloud_;
    float min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
    float inflation_radius_, cost_radius_;
    int cloud_downsampling_, lidar_downsampling_;
    int decay_factor_;
    bool debug_;
    std::string base_link_, map_topic_, prohibition_topic_;
    std::string laser_scan_topic_, cloud_topic_;

    // State
    nav_msgs::msg::OccupancyGrid static_map_, static_cost_map_;
    nav_msgs::msg::OccupancyGrid obstacles_map_, obstacles_inflated_map_, augmented_map_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_static_map_, latest_prohibition_map_;
    sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_;
    bool map_received_ = false, prohibition_received_ = false;
    bool are_there_obstacles_ = false;

    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Subs / pubs
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_, prohibition_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr augmented_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;

    // Services
    rclcpp::Service<nav_msgs::srv::GetMap>::SharedPtr srv_static_map_, srv_static_cost_map_,
                                                     srv_augmented_map_, srv_augmented_cost_map_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_are_there_obstacles_, srv_is_inside_obstacles_;

    // Timers
    rclcpp::TimerBase::SharedPtr decay_timer_, heartbeat_timer_;

    // Debug
    std::unique_ptr<debugger::DebugClient> dbg_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapAugmenterNode>();
    node->wait_for_maps();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
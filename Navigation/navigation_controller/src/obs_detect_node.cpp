#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "debugger/debug_client.hpp"

using namespace std::chrono_literals;

static constexpr int RATE = 30;

class ObsDetectNode : public rclcpp::Node
{
public:
    ObsDetectNode() : rclcpp::Node("obs_detect")
    {
        declare_parameters();
        load_parameters();

        tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        auto cb_enable = [this](std_msgs::msg::Bool::SharedPtr m) {
            enable_ = m->data;
            if (!enable_) {
                collision_risk_lidar_ = false;
                collision_risk_cloud_ = false;
            }
        };
        auto cb_path = [this](nav_msgs::msg::Path::SharedPtr m) {
            if (m->poses.empty()) return;
            global_goal_x_ = (float)m->poses.back().pose.position.x;
            global_goal_y_ = (float)m->poses.back().pose.position.y;
        };

        // Classic (canonical)
        sub_enable_classic_ = this->create_subscription<std_msgs::msg::Bool>(
            "/navigation/obs_detector/enable", 1, cb_enable);
        sub_goal_path_classic_ = this->create_subscription<nav_msgs::msg::Path>(
            "/navigation/simple_move/goal_path", 10, cb_path);

        // Mapless mirror
        sub_enable_ml_ = this->create_subscription<std_msgs::msg::Bool>(
            "/mapless_nav/obs_detector_enable", 1, cb_enable);
        sub_goal_path_ml_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mapless_nav/goal_path", 10, cb_path);

        // Command velocity (read-only), always from hardware
        sub_cmd_vel_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/hardware/base/cmd_vel", 1,
            [this](geometry_msgs::msg::Twist::SharedPtr m) {
                current_speed_linear_  = m->linear.x;
                current_speed_angular_ = m->angular.z;
            });

        sub_lidar_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            laser_scan_topic_, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::LaserScan::SharedPtr m) { cb_lidar(m); });

        if (use_cloud_) {
            sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                point_cloud_topic_, rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::PointCloud2::SharedPtr m) { cb_cloud(m); });
        }

        // Publications: classic + mirror
        pub_collision_risk_classic_ = this->create_publisher<std_msgs::msg::Bool>(
            "/navigation/obs_detector/collision_risk", 1);
        pub_collision_risk_ml_ = this->create_publisher<std_msgs::msg::Bool>(
            "/mapless_nav/collision_risk", 1);
        pub_rejection_force_classic_ = this->create_publisher<geometry_msgs::msg::Vector3>(
            "/navigation/obs_detector/pf_rejection_force", 1);
        pub_rejection_force_ml_ = this->create_publisher<geometry_msgs::msg::Vector3>(
            "/mapless_nav/pf_rejection_force", 1);
        pub_markers_classic_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/navigation/obs_detector/pot_field_markers", 1);
        pub_markers_ml_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mapless_nav/pot_field_markers", 1);

        // Services: classic + mirror
        auto cb_in_front = [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
            double fx, fy;
            bool risk = false;
            if (use_lidar_ && last_scan_) risk |= check_lidar(last_scan_, fx, fy);
            if (use_cloud_ && last_cloud_) risk |= check_cloud(last_cloud_, fx, fy);
            resp->success = risk;
        };
        srv_in_front_classic_ = this->create_service<std_srvs::srv::Trigger>(
            "/navigation/obs_detector/obstacle_in_front", cb_in_front);
        srv_in_front_ml_ = this->create_service<std_srvs::srv::Trigger>(
            "/mapless_nav/obstacle_in_front", cb_in_front);

        loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / RATE),
            [this]() { publish_loop(); });

        dbg_ = std::make_unique<debugger::DebugClient>(this, true);
        dbg_->send("obs_detect ready", "NODE_STATUS", "info", -1.0f);
        heartbeat_timer_ = this->create_wall_timer(1s, [this]() {
            dbg_->send("Node initialized and ready.", "NODE_STATUS", "info", -1.0f);
        });

        RCLCPP_INFO(this->get_logger(),
            "obs_detect: lidar=%s cloud=%s pf=%s k_rej=%.2f d0=%.2f base=%s",
            use_lidar_ ? "on" : "off", use_cloud_ ? "on" : "off",
            use_pot_fields_ ? "on" : "off", pot_k_rej_, pot_d0_, base_link_.c_str());
    }

    void wait_for_sensors()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for first sensor messages...");
        rclcpp::Rate rate(10.0);
        int attempts = 0;
        while (rclcpp::ok() && attempts < 100) {
            rclcpp::spin_some(this->get_node_base_interface());
            bool ok_l = !use_lidar_ || last_scan_ != nullptr;
            bool ok_c = !use_cloud_ || last_cloud_ != nullptr;
            if (ok_l && ok_c) break;
            attempts++;
            rate.sleep();
        }
        RCLCPP_INFO(this->get_logger(), "Waiting for sensor transforms...");
        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());
            bool ok = true;
            if (use_lidar_ && last_scan_)
                ok &= tf_buffer_->canTransform(base_link_, last_scan_->header.frame_id, tf2::TimePointZero);
            if (use_cloud_ && last_cloud_)
                ok &= tf_buffer_->canTransform(base_link_, last_cloud_->header.frame_id, tf2::TimePointZero);
            if (ok) break;
            rate.sleep();
        }
        RCLCPP_INFO(this->get_logger(), "Sensors ready.");
    }

private:
    void declare_parameters()
    {
        this->declare_parameter("debug", false);
        this->declare_parameter("use_pot_fields", false);
        this->declare_parameter("use_lidar", true);
        this->declare_parameter("use_point_cloud", false);
        this->declare_parameter("min_x", 0.30);
        this->declare_parameter("max_x", 0.80);
        this->declare_parameter("min_y", -0.30);
        this->declare_parameter("max_y", 0.30);
        this->declare_parameter("min_z", 0.05);
        this->declare_parameter("max_z", 1.50);
        this->declare_parameter("pot_fields_k_rej", 1.0);
        this->declare_parameter("pot_fields_d0", 1.0);
        this->declare_parameter("cloud_points_threshold", 100);
        this->declare_parameter("cloud_downsampling", 9);
        this->declare_parameter("lidar_points_threshold", 10);
        this->declare_parameter("lidar_downsampling", 1);
        this->declare_parameter("point_cloud_topic", std::string("/hardware/camera/points"));
        this->declare_parameter("laser_scan_topic", std::string("/hardware/base/scan"));
        this->declare_parameter("base_link_name", std::string("base_link"));
    }

    void load_parameters()
    {
        debug_         = this->get_parameter("debug").as_bool();
        use_pot_fields_= this->get_parameter("use_pot_fields").as_bool();
        use_lidar_     = this->get_parameter("use_lidar").as_bool();
        use_cloud_     = this->get_parameter("use_point_cloud").as_bool();
        min_x_ = (float)this->get_parameter("min_x").as_double();
        max_x_ = (float)this->get_parameter("max_x").as_double();
        min_y_ = (float)this->get_parameter("min_y").as_double();
        max_y_ = (float)this->get_parameter("max_y").as_double();
        min_z_ = (float)this->get_parameter("min_z").as_double();
        max_z_ = (float)this->get_parameter("max_z").as_double();
        pot_k_rej_ = (float)this->get_parameter("pot_fields_k_rej").as_double();
        pot_d0_    = (float)this->get_parameter("pot_fields_d0").as_double();
        cloud_threshold_ = this->get_parameter("cloud_points_threshold").as_int();
        cloud_downsampling_ = this->get_parameter("cloud_downsampling").as_int();
        lidar_threshold_ = this->get_parameter("lidar_points_threshold").as_int();
        lidar_downsampling_ = this->get_parameter("lidar_downsampling").as_int();
        point_cloud_topic_ = this->get_parameter("point_cloud_topic").as_string();
        laser_scan_topic_  = this->get_parameter("laser_scan_topic").as_string();
        base_link_         = this->get_parameter("base_link_name").as_string();
    }

    float get_search_distance()
    {
        float rx, ry, rt;
        if (!get_robot_pose(rx, ry, rt)) return max_x_;
        float d = std::sqrt(std::pow(global_goal_x_ - rx, 2.0f) +
                            std::pow(global_goal_y_ - ry, 2.0f));
        return d < max_x_ ? d : max_x_;
    }

    bool get_robot_pose(float& x, float& y, float& t)
    {
        try {
            auto tf = tf_buffer_->lookupTransform("odom", base_link_, tf2::TimePointZero);
            x = (float)tf.transform.translation.x;
            y = (float)tf.transform.translation.y;
            const auto& q = tf.transform.rotation;
            t = std::atan2((float)q.z, (float)q.w) * 2.0f;
            return true;
        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    Eigen::Affine3d tf_to_eigen(const std::string& target, const std::string& source)
    {
        auto tf = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);
        return tf2::transformToEigen(tf);
    }

    void cb_lidar(const sensor_msgs::msg::LaserScan::SharedPtr m)
    {
        last_scan_ = m;
        double fx, fy;
        collision_risk_lidar_ = check_lidar(m, fx, fy);
        rejection_force_lidar_.x = (float)fx;
        rejection_force_lidar_.y = (float)fy;
        rejection_force_lidar_.z = 0.0f;
    }

    void cb_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr m)
    {
        last_cloud_ = m;
        double fx, fy;
        collision_risk_cloud_ = check_cloud(m, fx, fy);
        rejection_force_cloud_.x = (float)fx;
        rejection_force_cloud_.y = (float)fy;
        rejection_force_cloud_.z = 0.0f;
    }

    bool check_lidar(const sensor_msgs::msg::LaserScan::ConstSharedPtr& msg,
                     double& rej_x, double& rej_y)
    {
        if (current_speed_linear_ <= 0 && !debug_) return false;
        float opt_x = get_search_distance();
        int obstacle_count = 0, force_count = 0;
        double raw_rej_x = 0.0, raw_rej_y = 0.0;

        Eigen::Affine3d tf;
        try { tf = tf_to_eigen(base_link_, msg->header.frame_id); }
        catch (const tf2::TransformException&) { return false; }

        for (size_t i = 0; i < msg->ranges.size(); i += lidar_downsampling_) {
            float a = msg->angle_min + i * msg->angle_increment;
            float r = msg->ranges[i];
            if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;

            Eigen::Vector3d v(r * std::cos(a), r * std::sin(a), 0.0);
            v = tf * v;

            if (v.x() > min_x_ && v.x() < opt_x &&
                v.y() > min_y_ && v.y() < max_y_ &&
                v.z() > min_z_ && v.z() < max_z_) obstacle_count++;

            if (v.norm() < pot_d0_) {
                float mag = pot_k_rej_ * std::sqrt(1.0f / (float)v.norm() - 1.0f / pot_d0_);
                raw_rej_x -= mag * v.x();
                raw_rej_y -= mag * v.y();
                force_count++;
            }
        }

        if (force_count > 0) {
            raw_rej_x /= force_count;
            raw_rej_y /= force_count;
        }

        // Strict magnitude clamp to prevent sudden physical impulses and AMCL jumps
        double max_force = 4.0;
        double total_mag = std::hypot(raw_rej_x, raw_rej_y);
        if (total_mag > max_force) {
            rej_x = (raw_rej_x / total_mag) * max_force;
            rej_y = (raw_rej_y / total_mag) * max_force;
        } else {
            rej_x = raw_rej_x;
            rej_y = raw_rej_y;
        }

        return obstacle_count > lidar_threshold_;
    }

    bool check_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                    double& rej_x, double& rej_y)
    {
        if (current_speed_linear_ <= 0 && !debug_) return false;
        float opt_x = get_search_distance();
        int obstacle_count = 0, force_count = 0;
        double raw_rej_x = 0.0, raw_rej_y = 0.0;

        Eigen::Affine3d tf;
        try { tf = tf_to_eigen(base_link_, msg->header.frame_id); }
        catch (const tf2::TransformException&) { return false; }

        const uint8_t* p = msg->data.data();
        size_t n = (size_t)msg->width * msg->height;
        for (size_t i = 0; i < n; i += cloud_downsampling_) {
            float px, py, pz;
            std::memcpy(&px, p, sizeof(float));
            std::memcpy(&py, p + 4, sizeof(float));
            std::memcpy(&pz, p + 8, sizeof(float));
            Eigen::Vector3d v(px, py, pz);
            v = tf * v;

            if (v.x() > min_x_ && v.x() < opt_x &&
                v.y() > min_y_ && v.y() < max_y_ &&
                v.z() > min_z_ && v.z() < max_z_) obstacle_count++;

            if (v.z() > min_z_ && v.norm() < pot_d0_) {
                float mag = pot_k_rej_ * std::sqrt(1.0f / (float)v.norm() - 1.0f / pot_d0_);
                raw_rej_x -= mag * v.x();
                raw_rej_y -= mag * v.y();
                force_count++;
            }
            p += cloud_downsampling_ * msg->point_step;
        }

        if (force_count > 0) {
            raw_rej_x /= force_count;
            raw_rej_y /= force_count;
        }

        // Strict magnitude clamp to prevent sudden physical impulses and AMCL jumps
        double max_force = 4.0;
        double total_mag = std::hypot(raw_rej_x, raw_rej_y);
        if (total_mag > max_force) {
            rej_x = (raw_rej_x / total_mag) * max_force;
            rej_y = (raw_rej_y / total_mag) * max_force;
        } else {
            rej_x = raw_rej_x;
            rej_y = raw_rej_y;
        }

        return obstacle_count > cloud_threshold_;
    }
    
    visualization_msgs::msg::MarkerArray get_force_arrow_markers(
        const geometry_msgs::msg::Vector3& rej_total)
    {
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = base_link_;
        m.header.stamp = this->get_clock()->now();
        m.ns = "pot_fields";
        m.type = visualization_msgs::msg::Marker::ARROW;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.05;
        m.scale.y = 0.10;
        m.points.resize(2);

        m.id = 0;
        m.points[0].x = 0.0f; m.points[0].y = 0.0f; m.points[0].z = 0.10f;
        m.points[1].x = 10.0f * rej_total.x;
        m.points[1].y = 10.0f * rej_total.y;
        m.points[1].z = 0.10f;
        m.color.a = 1.0f; m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f;
        markers.markers.push_back(m);

        return markers;
    }

    void publish_loop()
    {
        if (!enable_) return;

        std_msgs::msg::Bool risk;
        risk.data = collision_risk_lidar_ || collision_risk_cloud_;
        pub_collision_risk_classic_->publish(risk);
        pub_collision_risk_ml_->publish(risk);

        if (use_pot_fields_) {
            geometry_msgs::msg::Vector3 r;
            r.x = rejection_force_lidar_.x + rejection_force_cloud_.x;
            r.y = rejection_force_lidar_.y + rejection_force_cloud_.y;
            r.z = 0.0f;
            pub_rejection_force_classic_->publish(r);
            pub_rejection_force_ml_->publish(r);

            auto markers = get_force_arrow_markers(r);
            pub_markers_classic_->publish(markers);
            pub_markers_ml_->publish(markers);
        }
    }

    // params
    bool debug_, use_pot_fields_, use_lidar_, use_cloud_;
    float min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
    float pot_k_rej_, pot_d0_;
    int cloud_threshold_, cloud_downsampling_;
    int lidar_threshold_, lidar_downsampling_;
    std::string point_cloud_topic_, laser_scan_topic_, base_link_;

    // state
    bool enable_ = false;
    bool collision_risk_lidar_ = false, collision_risk_cloud_ = false;
    float current_speed_linear_ = 0.0f, current_speed_angular_ = 0.0f;
    float global_goal_x_ = 999999.0f, global_goal_y_ = 999999.0f;
    geometry_msgs::msg::Vector3 rejection_force_lidar_, rejection_force_cloud_;
    sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_;

    // ROS handles
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_enable_classic_, sub_enable_ml_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_goal_path_classic_, sub_goal_path_ml_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_lidar_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_collision_risk_classic_, pub_collision_risk_ml_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_rejection_force_classic_, pub_rejection_force_ml_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_classic_, pub_markers_ml_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_in_front_classic_, srv_in_front_ml_;
    rclcpp::TimerBase::SharedPtr loop_timer_, heartbeat_timer_;
    std::unique_ptr<debugger::DebugClient> dbg_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObsDetectNode>();
    node->wait_for_sensors();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
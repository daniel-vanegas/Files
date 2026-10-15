#include <memory>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "debugger/debug_client.hpp"

class GetGoalPoint : public rclcpp::Node
{
public:
    GetGoalPoint()
    : Node("get_goal_point"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_),
      dbg_(this, true)
    {
        // ================= DEBUGGER =================
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");

        // Parameters
        this->declare_parameter("robot_frame", "base_link");
        this->declare_parameter("goal_frame", "odom");
        this->declare_parameter("goal_topic", "/clicked_point");
        this->declare_parameter("output_topic", "/NN_goal");
        this->declare_parameter("rate_hz", 50.0);

        robot_frame_ = this->get_parameter("robot_frame").as_string();
        goal_frame_  = this->get_parameter("goal_frame").as_string();
        goal_topic_  = this->get_parameter("goal_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        double rate_hz = this->get_parameter("rate_hz").as_double();

        // Subscriber to clicked point
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            goal_topic_, 10,
            std::bind(&GetGoalPoint::goalCallback, this, std::placeholders::_1));

        // Publisher for [distance, angle]
        pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(output_topic_, 10);

        // Timer for periodic publishing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate_hz)),
            std::bind(&GetGoalPoint::publishGoal, this));

        RCLCPP_INFO(this->get_logger(), "GetGoalPoint node started");

        heartbeat_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GetGoalPoint::heartbeatCallback, this));
    }

private:
    void heartbeatCallback()
    {
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");
    }

    void goalCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        goal_set_ = true;
        goal_ = *msg;

        std::string log_msg = "Goal received: (" +
                              std::to_string(goal_.point.x) + ", " +
                              std::to_string(goal_.point.y) + ")";
        RCLCPP_INFO(this->get_logger(), "%s", log_msg.c_str());
        dbg_.send(log_msg, "DEBUG", "info", 7.0);
    }

    void publishGoal()
    {
        if (!goal_set_) {
            // No goal yet – publish zero distance/angle
            std_msgs::msg::Float32MultiArray msg;
            msg.data = {0.0f, 0.0f};
            pub_->publish(msg);
            return;
        }

        try {
            // Lookup transform from goal frame to robot frame
            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_.lookupTransform(
                robot_frame_,
                goal_.header.frame_id,
                tf2::TimePointZero);

            // Transform goal into robot frame
            geometry_msgs::msg::PointStamped goal_robot;
            tf2::doTransform(goal_, goal_robot, transform_stamped);

            double dx = goal_robot.point.x;
            double dy = goal_robot.point.y;
            double distance = std::hypot(dx, dy);
            double angle = std::atan2(dy, dx);

            std_msgs::msg::Float32MultiArray msg;
            msg.data = {static_cast<float>(distance), static_cast<float>(angle)};
            pub_->publish(msg);

            RCLCPP_DEBUG(this->get_logger(), "Goal relative: d=%.2f, ang=%.2f", distance, angle);
        } catch (const tf2::TransformException & ex) {
            std::string err_msg = "Transform failed: " + std::string(ex.what());
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "%s", err_msg.c_str());
            dbg_.send(err_msg, "DEBUG", "warn");
        }
    }

    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    debugger::DebugClient dbg_;

    geometry_msgs::msg::PointStamped goal_;
    bool goal_set_ = false;
    std::string robot_frame_, goal_frame_, goal_topic_, output_topic_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GetGoalPoint>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
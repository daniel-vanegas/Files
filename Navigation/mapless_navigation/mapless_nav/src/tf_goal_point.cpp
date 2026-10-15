// #include <memory>
// #include <cmath>
// #include <string>

// #include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/float32_multi_array.hpp"
// #include "std_msgs/msg/bool.hpp"
// #include "tf2_ros/transform_listener.h"
// #include "tf2_ros/buffer.h"

// #include "known_locations_interfaces/srv/navigate_mapless.hpp"

// class TFGoalPoint : public rclcpp::Node
// {
// public:
//   using NavigateMapless = known_locations_interfaces::srv::NavigateMapless;

//   TFGoalPoint()
//   : Node("tf_goal_point"),
//     tf_buffer_(this->get_clock()),
//     tf_listener_(tf_buffer_)
//   {
//     this->declare_parameter("robot_frame", "base_link");
//     this->declare_parameter("output_topic", "/NN_goal");
//     this->declare_parameter("goal_active_topic", "/mapless_nav/goal_active");
//     this->declare_parameter("service_name", "/navigation/mapless_goto");
//     this->declare_parameter("rate_hz", 50.0);

//     robot_frame_ = this->get_parameter("robot_frame").as_string();
//     output_topic_ = this->get_parameter("output_topic").as_string();
//     goal_active_topic_ = this->get_parameter("goal_active_topic").as_string();
//     service_name_ = this->get_parameter("service_name").as_string();

//     double rate_hz = this->get_parameter("rate_hz").as_double();

//     nn_goal_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
//       output_topic_, 10);

//     goal_active_pub_ = this->create_publisher<std_msgs::msg::Bool>(
//       goal_active_topic_, 10);

//     service_ = this->create_service<NavigateMapless>(
//       service_name_,
//       std::bind(
//         &TFGoalPoint::navigateCallback,
//         this,
//         std::placeholders::_1,
//         std::placeholders::_2));

//     timer_ = this->create_wall_timer(
//       std::chrono::milliseconds(static_cast<int>(1000.0 / rate_hz)),
//       std::bind(&TFGoalPoint::publishGoal, this));

//     RCLCPP_INFO(this->get_logger(), "TFGoalPoint ready. Service: %s", service_name_.c_str());
//   }

// private:
//   void navigateCallback(
//     const std::shared_ptr<NavigateMapless::Request> request,
//     std::shared_ptr<NavigateMapless::Response> response)
//   {
//     if (request->target_frame.empty()) {
//       response->success = false;
//       response->message = "target_frame is empty";
//       return;
//     }

//     target_frame_ = request->target_frame;
//     active_ = true;

//     publishGoalActive(true);

//     response->success = true;
//     response->message = "Mapless goal accepted: " + target_frame_;

//     RCLCPP_INFO(
//       this->get_logger(),
//       "New mapless target frame: %s",
//       target_frame_.c_str());
//   }

//   void publishGoal()
//   {
//     std_msgs::msg::Float32MultiArray msg;

//     if (!active_ || target_frame_.empty()) {
//       msg.data = {0.0f, 0.0f};
//       nn_goal_pub_->publish(msg);
//       return;
//     }

//     try {
//       auto tf = tf_buffer_.lookupTransform(
//         robot_frame_,
//         target_frame_,
//         tf2::TimePointZero);

//       double dx = tf.transform.translation.x;
//       double dy = tf.transform.translation.y;

//       double distance = std::hypot(dx, dy);
//       double angle = std::atan2(dy, dx);

//       msg.data = {
//         static_cast<float>(distance),
//         static_cast<float>(angle)
//       };

//       nn_goal_pub_->publish(msg);

//       RCLCPP_DEBUG(
//         this->get_logger(),
//         "Target %s relative to %s: d=%.3f ang=%.3f",
//         target_frame_.c_str(),
//         robot_frame_.c_str(),
//         distance,
//         angle);

//     } catch (const tf2::TransformException & ex) {
//       RCLCPP_WARN_THROTTLE(
//         this->get_logger(),
//         *this->get_clock(),
//         2000,
//         "Could not transform %s -> %s: %s",
//         robot_frame_.c_str(),
//         target_frame_.c_str(),
//         ex.what());

//       msg.data = {0.0f, 0.0f};
//       nn_goal_pub_->publish(msg);
//     }
//   }

//   void publishGoalActive(bool active)
//   {
//     std_msgs::msg::Bool msg;
//     msg.data = active;
//     goal_active_pub_->publish(msg);
//   }

//   tf2_ros::Buffer tf_buffer_;
//   tf2_ros::TransformListener tf_listener_;

//   rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr nn_goal_pub_;
//   rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_active_pub_;
//   rclcpp::Service<NavigateMapless>::SharedPtr service_;
//   rclcpp::TimerBase::SharedPtr timer_;

//   std::string robot_frame_;
//   std::string output_topic_;
//   std::string goal_active_topic_;
//   std::string service_name_;
//   std::string target_frame_;

//   bool active_ = false;
// };

// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<TFGoalPoint>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
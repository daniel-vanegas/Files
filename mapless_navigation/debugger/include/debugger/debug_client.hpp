#ifndef DEBUG_CLIENT_HPP
#define DEBUG_CLIENT_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>

namespace debugger {

class DebugClient {
public:
    DebugClient(rclcpp::Node* node, bool verbose = true)
        : node_(node), verbose_(verbose)
    {
        pub_ = node_->create_publisher<std_msgs::msg::String>(
            "/debugger/debug_msgs", 10);
        node_name_ = node_->get_name();
    }

    void send(const std::string& message,
              const std::string& msg_id = "General",
              const std::string& level = "info",
              float ttl = -1.0) // Added TTL
    {
        if (msg_id == "NODE_STATUS" || verbose_) {
            std_msgs::msg::String out;
            out.data = node_name_ + "|" + msg_id + "|" + level + "|" + 
                       std::to_string(ttl) + "|" + message;
            pub_->publish(out);
        }
    }

private:
    rclcpp::Node* node_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    std::string node_name_;
    bool verbose_;
};

} // namespace debugger

#endif
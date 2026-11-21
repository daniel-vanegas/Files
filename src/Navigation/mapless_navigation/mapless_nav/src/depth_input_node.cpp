#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class DepthInputNode : public rclcpp::Node
{
public:
    DepthInputNode() : Node("depth_input_node")
    {
        input_topic_ = declare_parameter<std::string>(
            "input_topic", "/hardware/camera/depth/image_raw");

        output_topic_ = declare_parameter<std::string>(
            "output_topic", "/mapless/depth_input");

        output_width_ = declare_parameter<int>(
            "output_width", 160);

        output_height_ = declare_parameter<int>(
            "output_height", 120);

        min_depth_ = declare_parameter<double>(
            "min_depth", 0.1);

        max_depth_ = declare_parameter<double>(
            "max_depth", 8.0);

        if (output_width_ <= 0 || output_height_ <= 0 ||
            min_depth_ < 0.0 || max_depth_ <= min_depth_) {
            throw std::runtime_error("Invalid depth parameters");
        }

        auto qos = rclcpp::SensorDataQoS();

        publisher_ = create_publisher<sensor_msgs::msg::Image>(
            output_topic_, qos);

        subscriber_ = create_subscription<sensor_msgs::msg::Image>(
            input_topic_,
            qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                processImage(msg);
            });

        RCLCPP_INFO(
            get_logger(),
            "Depth input ready: %s -> %s, output=%dx%d, "
            "valid range=[%.2f, %.2f] m",
            input_topic_.c_str(),
            output_topic_.c_str(),
            output_width_,
            output_height_,
            min_depth_,
            max_depth_);
    }

private:
    static bool hostIsBigEndian()
    {
        const uint16_t value = 0x0102;
        const uint8_t* p =
            reinterpret_cast<const uint8_t*>(&value);
        return p[0] == 0x01;
    }

    static float readFloat32(
        const uint8_t* bytes,
        bool source_big_endian)
    {
        uint8_t native[4];

        if (source_big_endian == hostIsBigEndian()) {
            std::memcpy(native, bytes, 4);
        } else {
            native[0] = bytes[3];
            native[1] = bytes[2];
            native[2] = bytes[1];
            native[3] = bytes[0];
        }

        float value;
        std::memcpy(&value, native, sizeof(float));
        return value;
    }

    void processImage(
        const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        if (msg->encoding !=
            sensor_msgs::image_encodings::TYPE_32FC1) {

            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Unsupported depth encoding: %s",
                msg->encoding.c_str());
            return;
        }

        const int in_width = static_cast<int>(msg->width);
        const int in_height = static_cast<int>(msg->height);

        if (in_width <= 0 || in_height <= 0 ||
            msg->step < msg->width * sizeof(float) ||
            msg->data.size() <
                static_cast<size_t>(msg->step) * msg->height) {

            RCLCPP_WARN(
                get_logger(), "Invalid depth image layout");
            return;
        }

        sensor_msgs::msg::Image output;
        output.header = msg->header;
        output.height = static_cast<uint32_t>(output_height_);
        output.width = static_cast<uint32_t>(output_width_);
        output.encoding =
            sensor_msgs::image_encodings::TYPE_32FC1;
        output.is_bigendian = hostIsBigEndian() ? 1 : 0;
        output.step =
            static_cast<uint32_t>(output_width_ * sizeof(float));

        output.data.resize(
            static_cast<size_t>(output.step) * output.height);

        const float invalid =
            std::numeric_limits<float>::quiet_NaN();

        std::vector<float> values;
        values.reserve(
            (in_width / output_width_ + 2) *
            (in_height / output_height_ + 2));

        size_t valid_output_pixels = 0;

        for (int oy = 0; oy < output_height_; ++oy) {
            const int y0 =
                static_cast<int>(
                    static_cast<int64_t>(oy) * in_height /
                    output_height_);

            const int y1 =
                static_cast<int>(
                    static_cast<int64_t>(oy + 1) * in_height /
                    output_height_);

            for (int ox = 0; ox < output_width_; ++ox) {
                const int x0 =
                    static_cast<int>(
                        static_cast<int64_t>(ox) * in_width /
                        output_width_);

                const int x1 =
                    static_cast<int>(
                        static_cast<int64_t>(ox + 1) * in_width /
                        output_width_);

                values.clear();

                for (int y = y0; y < y1; ++y) {
                    const uint8_t* row =
                        msg->data.data() +
                        static_cast<size_t>(y) * msg->step;

                    for (int x = x0; x < x1; ++x) {
                        const float depth = readFloat32(
                            row + static_cast<size_t>(x) * 4,
                            msg->is_bigendian != 0);

                        if (std::isfinite(depth) &&
                            depth >= min_depth_ &&
                            depth <= max_depth_) {
                            values.push_back(depth);
                        }
                    }
                }

                float result = invalid;

                if (!values.empty()) {
                    const size_t middle = values.size() / 2;

                    std::nth_element(
                        values.begin(),
                        values.begin() + middle,
                        values.end());

                    result = values[middle];
                    ++valid_output_pixels;
                }

                const size_t offset =
                    (static_cast<size_t>(oy) * output_width_
                     + ox) * sizeof(float);

                std::memcpy(
                    output.data.data() + offset,
                    &result,
                    sizeof(float));
            }
        }

        publisher_->publish(output);

        RCLCPP_DEBUG(
            get_logger(),
            "Depth output: %zu/%zu valid pixels",
            valid_output_pixels,
            static_cast<size_t>(output_width_) *
                output_height_);
    }

    std::string input_topic_;
    std::string output_topic_;

    int output_width_;
    int output_height_;

    double min_depth_;
    double max_depth_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
        subscriber_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
        publisher_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthInputNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

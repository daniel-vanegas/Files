// virtual_pointcloud.cpp

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

class DepthToPointCloud : public rclcpp::Node
{
public:
    DepthToPointCloud()
    : Node("depth_to_pointcloud")
    {
        // Parameters
        declare_parameter("max_frequency", 15.0);
        declare_parameter("min_depth", 0.1);
        declare_parameter("max_depth", 8.0);
        declare_parameter("downsample_step", 2);

        max_freq_ = get_parameter("max_frequency").as_double();
        min_depth_ = get_parameter("min_depth").as_double();
        max_depth_ = get_parameter("max_depth").as_double();
        downsample_step_ = get_parameter("downsample_step").as_int();

        // Subscribers
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/hardware/camera/depth/image_raw",
            rclcpp::SensorDataQoS(),
            std::bind(&DepthToPointCloud::depthCallback, this, std::placeholders::_1));

        info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/hardware/camera/depth/camera_info",
            rclcpp::SensorDataQoS(),
            std::bind(&DepthToPointCloud::infoCallback, this, std::placeholders::_1));

        // Publisher
        pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/hardware/camera/points",
            rclcpp::SensorDataQoS());

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        last_pub_time_ = this->now();

        RCLCPP_INFO(get_logger(),
            "DepthToPointCloud started | step=%d | max_freq=%.2f",
            downsample_step_, max_freq_);
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_info_received_) {
            camera_info_ = *msg;
            camera_info_received_ = true;
            RCLCPP_INFO(get_logger(), "CameraInfo received");
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!camera_info_received_)
            return;

        // Throttle
        rclcpp::Time now = this->now();
        if ((now - last_pub_time_).seconds() < (1.0 / max_freq_))
            return;
        last_pub_time_ = now;

        // Convert depth image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg,
                sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        const cv::Mat &depth = cv_ptr->image;
        if (depth.empty())
            return;

        // Intrinsics
        double fx = camera_info_.k[0];
        double fy = camera_info_.k[4];
        double cx = camera_info_.k[2];
        double cy = camera_info_.k[5];

        // TF lookup
        geometry_msgs::msg::TransformStamped tf_msg;
        try {
            tf_msg = tf_buffer_->lookupTransform(
                "base_link",
                "camera_depth_frame",
                msg->header.stamp,
                tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(get_logger(), "TF failed: %s", ex.what());
            return;
        }

        tf2::Transform tf_cam_to_base;
        tf2::fromMsg(tf_msg.transform, tf_cam_to_base);

        // Point cloud buffer
        std::vector<float> cloud;
        cloud.reserve(depth.rows * depth.cols * 3 / (downsample_step_ * downsample_step_));

        for (int v = 0; v < depth.rows; v += downsample_step_)
        {
            const float* row = depth.ptr<float>(v);

            for (int u = 0; u < depth.cols; u += downsample_step_)
            {
                float d = row[u];

                if (!std::isfinite(d))
                    continue;

                if (d < min_depth_ || d > max_depth_)
                    continue;

                // camera frame
                float x = (u - cx) * d / fx;
                float y = (v - cy) * d / fy;
                float z = d;

                tf2::Vector3 p_cam(x, y, z);
                tf2::Vector3 p_base = tf_cam_to_base * p_cam;

                cloud.push_back(p_base.x());
                cloud.push_back(p_base.y());
                cloud.push_back(p_base.z());
            }
        }

        if (cloud.empty())
            return;

        sensor_msgs::msg::PointCloud2 out;
        out.header.stamp = msg->header.stamp;
        out.header.frame_id = "base_link";

        out.height = 1;
        out.width = cloud.size() / 3;

        out.is_bigendian = false;
        out.is_dense = true;

        sensor_msgs::PointCloud2Modifier modifier(out);
        modifier.setPointCloud2Fields(
            3,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32);

        modifier.resize(out.width);

        sensor_msgs::PointCloud2Iterator<float> ix(out, "x");
        sensor_msgs::PointCloud2Iterator<float> iy(out, "y");
        sensor_msgs::PointCloud2Iterator<float> iz(out, "z");

        for (size_t i = 0; i < cloud.size(); i += 3)
        {
            *ix = cloud[i];
            *iy = cloud[i + 1];
            *iz = cloud[i + 2];

            ++ix; ++iy; ++iz;
        }

        pc_pub_->publish(out);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    sensor_msgs::msg::CameraInfo camera_info_;
    bool camera_info_received_ = false;

    rclcpp::Time last_pub_time_;

    double max_freq_;
    double min_depth_;
    double max_depth_;
    int downsample_step_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthToPointCloud>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
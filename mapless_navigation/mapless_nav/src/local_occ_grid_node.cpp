#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Dense>
#include <cmath>

#include "debugger/debug_client.hpp"

class LocalOccGrid : public rclcpp::Node
{
public:
    LocalOccGrid()
        : Node("local_occ_grid_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_),
          dbg_(this, true)
    {
        // ================= DEBUGGER =================
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");

        // ================= PARAMETERS =================
        frame_id_      = declare_parameter("frame_id", "base_link");
        map_topic_     = declare_parameter("map_topic", "/mapless/local_occ_grid");
        array_topic_   = declare_parameter("array_topic", "");

        resolution_    = declare_parameter("resolution", 0.05);
        min_x_ = declare_parameter("min_x", -5.0);
        max_x_ = declare_parameter("max_x",  5.0);
        min_y_ = declare_parameter("min_y", -5.0);
        max_y_ = declare_parameter("max_y",  5.0);
        min_z_ = declare_parameter("min_z", 0.05);
        max_z_ = declare_parameter("max_z", 1.5);

        use_lidar_      = declare_parameter("use_lidar", true);
        use_cloud_      = declare_parameter("use_cloud", true);
        lidar_downsample_ = declare_parameter("lidar_downsample", 1);
        cloud_downsample_ = declare_parameter("cloud_downsample", 1);

        width_  = static_cast<int>((max_x_ - min_x_) / resolution_);
        height_ = static_cast<int>((max_y_ - min_y_) / resolution_);

        // ================= GRID INIT =================
        grid_.header.frame_id = frame_id_;
        grid_.info.resolution = resolution_;
        grid_.info.width  = width_;
        grid_.info.height = height_;
        grid_.info.origin.position.x = min_x_;
        grid_.info.origin.position.y = min_y_;
        grid_.info.origin.orientation.w = 1.0;
        grid_.data.resize(width_ * height_, 0);

        // ================= PUBLISHERS =================
        pub_grid_ = create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic_, 10);
        if (!array_topic_.empty()) {
            pub_array_ = create_publisher<std_msgs::msg::Float32MultiArray>(array_topic_, 10);
        }

        // ================= SUBSCRIBERS =================
        if (use_lidar_) {
            sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
                "/hardware/base/scan", 10,
                std::bind(&LocalOccGrid::scanCallback, this, std::placeholders::_1));
        }
        if (use_cloud_) {
            rclcpp::QoS qos(10);
            qos.best_effort();
            sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                "/hardware/camera/points", qos,
                std::bind(&LocalOccGrid::cloudCallback, this, std::placeholders::_1));
        }

        // ================= TIMER =================
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&LocalOccGrid::update, this));

        RCLCPP_INFO(get_logger(),
                    "OccGrid initialized: topic='%s', frame='%s', size=%dx%d, bounds [%.1f,%.1f]x[%.1f,%.1f]",
                    map_topic_.c_str(), frame_id_.c_str(), width_, height_,
                    min_x_, max_x_, min_y_, max_y_);
    }

private:
    // ================= CALLBACKS =================
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) { last_scan_ = msg; }
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) { last_cloud_ = msg; }

    // ================= TF =================
    Eigen::Affine3d getTF(const std::string &target, const std::string &source)
    {
        auto tf = tf_buffer_.lookupTransform(target, source, tf2::TimePointZero);
        const auto &t = tf.transform.translation;
        const auto &r = tf.transform.rotation;
        Eigen::Quaterniond q(r.w, r.x, r.y, r.z);
        Eigen::Affine3d out = Eigen::Affine3d::Identity();
        out.linear() = q.toRotationMatrix();
        out.translation() << t.x, t.y, t.z;
        return out;
    }

    // ================= UPDATE LOOP =================
    void update()
    {
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");
        std::fill(grid_.data.begin(), grid_.data.end(), 0);
        if (use_lidar_ && last_scan_)  processScan(last_scan_);
        if (use_cloud_ && last_cloud_) processCloud(last_cloud_);
        publish();
    }

    // ================= LASER PROCESSING (FIXED ANGLE CALC) =================
    void processScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        try {
            Eigen::Affine3d tf = getTF(frame_id_, msg->header.frame_id);
            const double angle_min = msg->angle_min;
            const double angle_inc = msg->angle_increment;
            const size_t N = msg->ranges.size();

            for (size_t i = 0; i < N; i += lidar_downsample_) {
                float r = msg->ranges[i];
                if (std::isnan(r) || std::isinf(r)) continue;

                // Recalculate angle from index – safe and independent of accumulated increments
                double angle = angle_min + i * angle_inc;
                Eigen::Vector3d p(r * std::cos(angle), r * std::sin(angle), 0.0);
                p = tf * p;
                markCell(p.x(), p.y(), 100);
            }
        } catch (...) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "TF scan failed");
            dbg_.send("TF scan failed", "DEBUG", "warn");
        }
    }

    // ================= CLOUD PROCESSING =================
    void processCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try {
            Eigen::Affine3d tf = getTF(frame_id_, msg->header.frame_id);
            sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");

            size_t idx = 0;
            for (; ix != ix.end(); ++ix, ++iy, ++iz, ++idx) {
                if (idx % cloud_downsample_ != 0) continue;
                if (*iz < min_z_ || *iz > max_z_) continue;
                Eigen::Vector3d p(*ix, *iy, *iz);
                p = tf * p;
                markCell(p.x(), p.y(), 100);
            }
        } catch (...) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "TF cloud failed");
            dbg_.send("TF cloud failed", "DEBUG", "err");
        }
    }

    // ================= GRID MARKING =================
    void markCell(double x, double y, int value)
    {
        int gx = static_cast<int>((x - min_x_) / resolution_);
        int gy = static_cast<int>((y - min_y_) / resolution_);
        if (gx < 0 || gy < 0 || gx >= width_ || gy >= height_) return;
        grid_.data[gy * width_ + gx] = value;
    }

    // ================= PUBLISH =================
    void publish()
    {
        grid_.header.stamp = now();
        pub_grid_->publish(grid_);
        if (pub_array_) {
            std_msgs::msg::Float32MultiArray arr;
            arr.data.resize(grid_.data.size());
            std::transform(grid_.data.begin(), grid_.data.end(), arr.data.begin(),
                           [](int8_t v) { return static_cast<float>(v); });
            pub_array_->publish(arr);
        }
    }

    // ROS2 members
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_grid_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_array_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::TimerBase::SharedPtr timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    debugger::DebugClient dbg_;

    // Data
    sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_;
    nav_msgs::msg::OccupancyGrid grid_;

    // Parameters
    double resolution_, min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
    int width_, height_, lidar_downsample_, cloud_downsample_;
    bool use_lidar_, use_cloud_;
    std::string frame_id_, map_topic_, array_topic_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LocalOccGrid>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
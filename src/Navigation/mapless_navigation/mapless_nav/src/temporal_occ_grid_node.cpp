#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <deque>

class UnifiedOccGrid : public rclcpp::Node
{
public:
    UnifiedOccGrid()
        : Node("unified_occ_grid_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        base_frame_  = declare_parameter("base_frame", "base_link");
        odom_frame_  = declare_parameter("odom_frame", "odom");
        
        lom_topic_ = declare_parameter("map_topic", "/mapless/local_occ_grid");
        tlom_topic_ = declare_parameter("temporal_map_topic", "/mapless/temporal_occ_grid");

        resolution_ = declare_parameter("resolution", 0.05);
        min_x_ = declare_parameter("min_x", -1.0);
        max_x_ = declare_parameter("max_x",  6.0);
        min_y_ = declare_parameter("min_y", -3.0);
        max_y_ = declare_parameter("max_y",  3.0);

        min_z_ = declare_parameter("min_z", 0.20);
        max_z_ = declare_parameter("max_z", 2.0);
        
        use_lidar_ = declare_parameter("use_lidar", true);
        use_cloud_ = declare_parameter("use_cloud", true);
        lidar_downsample_ = declare_parameter("lidar_downsample", 1);
        cloud_downsample_ = declare_parameter("cloud_downsample", 2);

        width_  = static_cast<int>((max_x_ - min_x_) / resolution_);
        height_ = static_cast<int>((max_y_ - min_y_) / resolution_);

        pub_lom_ = create_publisher<nav_msgs::msg::OccupancyGrid>(lom_topic_, 10);
        pub_tlom_ = create_publisher<nav_msgs::msg::OccupancyGrid>(tlom_topic_, 10);

        if (use_lidar_) {
            sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
                "/hardware/base/scan", 10,
                std::bind(&UnifiedOccGrid::scanCallback, this, std::placeholders::_1));
        }
        if (use_cloud_) {
            rclcpp::QoS qos(10);
            qos.best_effort();
            sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                "/hardware/camera/points", qos,
                std::bind(&UnifiedOccGrid::cloudCallback, this, std::placeholders::_1));
        }

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&UnifiedOccGrid::update, this));
    }

private:
    struct Pose2D {
        double x, y, yaw;
    };

    struct TimestampedGrid {
        nav_msgs::msg::OccupancyGrid grid;
        Pose2D pose_at_capture;
    };

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) { last_scan_ = msg; }
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) { last_cloud_ = msg; }

    Pose2D get2DPose(const std::string &target, const std::string &source)
    {
        auto tf_msg = tf_buffer_.lookupTransform(target, source, tf2::TimePointZero);
        Pose2D p;
        p.x = tf_msg.transform.translation.x;
        p.y = tf_msg.transform.translation.y;
        tf2::Quaternion q;
        tf2::fromMsg(tf_msg.transform.rotation, q);
        double roll, pitch;
        tf2::Matrix3x3(q).getRPY(roll, pitch, p.yaw);
        return p;
    }

    Eigen::Affine3d get3DTF(const std::string &target, const std::string &source)
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

    void markCell(nav_msgs::msg::OccupancyGrid &grid, double x, double y, int value)
    {
        int gx = static_cast<int>((x - min_x_) / resolution_);
        int gy = static_cast<int>((y - min_y_) / resolution_);
        if (gx < 0 || gy < 0 || gx >= width_ || gy >= height_) return;
        grid.data[gy * width_ + gx] = value;
    }

    void processScan(nav_msgs::msg::OccupancyGrid &grid)
    {
        if (!last_scan_) return;
        try {
            Eigen::Affine3d tf = get3DTF(base_frame_, last_scan_->header.frame_id);
            const double angle_min = last_scan_->angle_min;
            const double angle_inc = last_scan_->angle_increment;
            const size_t N = last_scan_->ranges.size();

            for (size_t i = 0; i < N; i += lidar_downsample_) {
                float r = last_scan_->ranges[i];
                if (std::isnan(r) || std::isinf(r)) continue;

                double angle = angle_min + i * angle_inc;
                Eigen::Vector3d p(r * std::cos(angle), r * std::sin(angle), 0.0);
                p = tf * p;
                markCell(grid, p.x(), p.y(), 100);
            }
        } catch (...) {}
    }

    void processCloud(nav_msgs::msg::OccupancyGrid &grid)
    {
        if (!last_cloud_) return;
        try {
            Eigen::Affine3d tf = get3DTF(base_frame_, last_cloud_->header.frame_id);
            sensor_msgs::PointCloud2ConstIterator<float> ix(*last_cloud_, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iy(*last_cloud_, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iz(*last_cloud_, "z");

            size_t idx = 0;
            for (; ix != ix.end(); ++ix, ++iy, ++iz, ++idx) {
                if (idx % cloud_downsample_ != 0) continue;
                if (!std::isfinite(*ix) || !std::isfinite(*iy) || !std::isfinite(*iz)) continue;

                Eigen::Vector3d p(*ix, *iy, *iz);
                p = tf * p;

                if (p.z() < min_z_ || p.z() > max_z_) continue;
                markCell(grid, p.x(), p.y(), 100);
            }
        } catch (...) {}
    }

    void update()
    {
        rclcpp::Time now = this->now();
        
        // 1. Generar LOM Original
        nav_msgs::msg::OccupancyGrid lom;
        lom.header.frame_id = base_frame_;
        lom.header.stamp = now;
        lom.info.resolution = resolution_;
        lom.info.width = width_;
        lom.info.height = height_;
        lom.info.origin.position.x = min_x_;
        lom.info.origin.position.y = min_y_;
        lom.info.origin.orientation.w = 1.0;
        lom.data.assign(width_ * height_, 0);

        processScan(lom);
        processCloud(lom);

        // 2. Anclar esta LOM al historial usando la pose exacta en este instante
        Pose2D current_pose;
        try {
            current_pose = get2DPose(odom_frame_, base_frame_);
            history_grids_.push_back({lom, current_pose});
            while (history_grids_.size() > 10) {
                history_grids_.pop_front();
            }
        } catch (...) {}

        // 3. Generar T-LOM
        nav_msgs::msg::OccupancyGrid tlom;
        tlom.header.frame_id = base_frame_;
        tlom.header.stamp = now;
        tlom.info.resolution = resolution_;
        tlom.info.width = width_;
        tlom.info.height = height_;
        tlom.info.origin.position.x = min_x_;
        tlom.info.origin.position.y = min_y_;
        tlom.info.origin.orientation.w = 1.0;
        tlom.data.assign(width_ * height_, 0);

        if (!history_grids_.empty()) {
            try {
                std::vector<int> match_count(width_ * height_, 0);

                for (const auto &item : history_grids_) {
                    double dx = item.pose_at_capture.x - current_pose.x;
                    double dy = item.pose_at_capture.y - current_pose.y;
                    double dyaw = item.pose_at_capture.yaw - current_pose.yaw;

                    while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
                    while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

                    double cos_curr = std::cos(current_pose.yaw);
                    double sin_curr = std::sin(current_pose.yaw);
                    double tx = cos_curr * dx + sin_curr * dy;
                    double ty = -sin_curr * dx + cos_curr * dy;

                    double c = std::cos(dyaw);
                    double s = std::sin(dyaw);

                    for (int y = 0; y < height_; ++y) {
                        for (int x = 0; x < width_; ++x) {
                            int idx = y * width_ + x;
                            if (item.grid.data[idx] <= 0) continue;

                            double lx = min_x_ + (x + 0.5) * resolution_;
                            double ly = min_y_ + (y + 0.5) * resolution_;

                            double ox = lx * c - ly * s + tx;
                            double oy = lx * s + ly * c + ty;

                            int gx = static_cast<int>(std::round((ox - min_x_) / resolution_));
                            int gy = static_cast<int>(std::round((oy - min_y_) / resolution_));

                            if (gx >= 0 && gx < width_ && gy >= 0 && gy < height_) {
                                match_count[gy * width_ + gx]++;
                            }
                        }
                    }
                }

                std::vector<int8_t> mem_layer(width_ * height_, 0);
                for (int i = 0; i < width_ * height_; ++i) {
                    if (match_count[i] >= 4) {
                        mem_layer[i] = 50;
                    }
                }

                std::vector<int> lom_dilated(width_ * height_, 0);
                for (int y = 1; y < height_ - 1; ++y) {
                    for (int x = 1; x < width_ - 1; ++x) {
                        if (lom.data[y * width_ + x] > 0) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    lom_dilated[(y + dy) * width_ + (x + dx)] = 1;
                                }
                            }
                        }
                    }
                }

                int best_dx = 0;
                int best_dy = 0;
                int max_score = -1;
                int search_radius = 15;

                for (int dy = -search_radius; dy <= search_radius; ++dy) {
                    for (int dx = -search_radius; dx <= search_radius; ++dx) {
                        int score = 0;
                        for (int y = search_radius; y < height_ - search_radius; ++y) {
                            for (int x = search_radius; x < width_ - search_radius; ++x) {
                                if (mem_layer[y * width_ + x] == 50) {
                                    int lx = x + dx;
                                    int ly = y + dy;
                                    if (lom_dilated[ly * width_ + lx] > 0) {
                                        score++;
                                    }
                                }
                            }
                        }
                        if (score > max_score) {
                            max_score = score;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                for (int y = search_radius; y < height_ - search_radius; ++y) {
                    for (int x = search_radius; x < width_ - search_radius; ++x) {
                        if (mem_layer[y * width_ + x] == 50) {
                            int lx = x + best_dx;
                            int ly = y + best_dy;
                            if (lx >= 1 && lx < width_ - 1 && ly >= 1 && ly < height_ - 1) {
                                for (int dy_off = -1; dy_off <= 1; ++dy_off) {
                                    for (int dx_off = -1; dx_off <= 1; ++dx_off) {
                                        tlom.data[(ly + dy_off) * width_ + (lx + dx_off)] = 50;
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (...) {}
        }

        // Superponer LOM original en la T-LOM
        for (size_t i = 0; i < lom.data.size(); ++i) {
            if (lom.data[i] > 0) {
                tlom.data[i] = lom.data[i];
            }
        }

        pub_lom_->publish(lom);
        pub_tlom_->publish(tlom);
    }

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_lom_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_tlom_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_;

    int width_, height_;
    double resolution_, min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
    bool use_lidar_, use_cloud_;
    int lidar_downsample_, cloud_downsample_;
    std::string base_frame_, odom_frame_, lom_topic_, tlom_topic_;

    std::deque<TimestampedGrid> history_grids_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UnifiedOccGrid>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
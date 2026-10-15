#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>

#include "debugger/debug_client.hpp"

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static std::atomic<bool> g_shutdown_requested{false};
static void sigint_handler(int) { g_shutdown_requested.store(true); }

static double normalize_angle(double a)
{
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

class DatasetRecorder : public rclcpp::Node
{
public:
    DatasetRecorder() : Node("dataset_recorder")
    {
        output_dir_        = declare_parameter<std::string>("output_dir", "");
        if (output_dir_.empty()) {
            output_dir_ = ament_index_cpp::get_package_share_directory("mapless_dataset") + "/data";
        }
        scenario_          = declare_parameter<std::string>("scenario", "EXX");
        teacher_           = declare_parameter<std::string>("teacher", "nav2_dwb_abstract");
        record_hz_         = declare_parameter<double>("record_hz", 5.0);
        map_frame_         = declare_parameter<std::string>("map_frame", "map");
        base_frame_        = declare_parameter<std::string>("base_frame", "base_link");
        max_goals_         = declare_parameter<int>("max_goals", 100);
        min_cmd_magnitude_ = declare_parameter<double>("min_cmd_magnitude", 0.01);
        stabilization_s_   = declare_parameter<double>("stabilization_time_sec", 3.0);

        costmap_topic_      = declare_parameter<std::string>("costmap_topic", "/mapless/local_occ_grid");
        cmd_vel_topic_      = declare_parameter<std::string>("cmd_vel_topic", "/hardware/base/cmd_vel");
        nn_goal_topic_      = declare_parameter<std::string>("nn_goal_topic", "/NN_goal");
        current_goal_topic_ = declare_parameter<std::string>("current_goal_topic", "/mapless_nav/current_goal");
        goal_active_topic_  = declare_parameter<std::string>("goal_active_topic", "/mapless_nav/goal_active");
        episode_id_topic_   = declare_parameter<std::string>("episode_id_topic", "/mapless_nav/episode_id");
        goal_success_topic_ = declare_parameter<std::string>("goal_success_topic", "/mapless_nav/goal_success");
        collision_topic_    = declare_parameter<std::string>("collision_topic", "/navigation/collision");

        stuck_window_s_          = declare_parameter<double>("stuck_window_s", 2.0);
        stuck_eps_pos_m_         = declare_parameter<double>("stuck_eps_pos_m", 0.02);
        stuck_eps_yaw_rad_       = declare_parameter<double>("stuck_eps_yaw_rad", 0.10);
        stuck_eps_cmd_lin_       = declare_parameter<double>("stuck_eps_cmd_lin", 0.05);
        stuck_eps_cmd_ang_       = declare_parameter<double>("stuck_eps_cmd_ang", 0.10);
        stuck_min_duration_s_    = declare_parameter<double>("stuck_min_duration_s", 1.5);
        collision_lookback_frames_ = declare_parameter<int>("collision_lookback_frames", 5);
        stuck_lookback_frames_     = declare_parameter<int>("stuck_lookback_frames", 5);

        prepareOutputDirectory();
        openCsv();
        writeDescriptorHeader();

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        auto sensor_qos = rclcpp::SensorDataQoS();

        sub_costmap_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            costmap_topic_, sensor_qos,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                last_costmap_ = m;
            });

        sub_cmd_ = create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic_, 10,
            [this](geometry_msgs::msg::Twist::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                last_cmd_ = *m;
            });

        sub_nn_goal_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            nn_goal_topic_, 10,
            [this](std_msgs::msg::Float32MultiArray::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                if (m->data.size() >= 2) {
                    last_d_ = m->data[0];
                    last_theta_ = m->data[1];
                }
            });

        sub_goal_ = create_subscription<geometry_msgs::msg::PointStamped>(
            current_goal_topic_, 10,
            [this](geometry_msgs::msg::PointStamped::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                last_goal_ = *m;
                have_goal_ = true;
            });

        sub_active_ = create_subscription<std_msgs::msg::Bool>(
            goal_active_topic_, 10,
            [this](std_msgs::msg::Bool::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                goal_active_ = m->data;
            });

        sub_episode_ = create_subscription<std_msgs::msg::Int32>(
            episode_id_topic_, 10,
            [this](std_msgs::msg::Int32::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                if (m->data != current_episode_) {
                    if (!episode_buffer_.empty()) {
                        RCLCPP_WARN(get_logger(),
                            "Discarding %zu stale samples from episode %d",
                            episode_buffer_.size(), current_buffer_episode_);
                        episode_buffer_.clear();
                    }
                    prev_episode_id_ = current_episode_;
                    current_episode_ = m->data;
                    episode_just_changed_ = true;

                    // goals_completed_ is NOT incremented here.
                    // It counts committed episodes, not started ones.

                    pose_hist_.clear();
                    stuck_active_ = false;
                    stuck_event_counted_ = false;
                    collision_latch_ = false;
                    episode_had_collision_ = false;
                    episode_had_stuck_ = false;
                    episode_collision_events_ = 0;
                    was_in_collision_ = false;

                    RCLCPP_INFO(get_logger(),
                        "Episode %d -> %d",
                        prev_episode_id_, current_episode_);

                    sendCollisionStatus(0, total_collision_events_.load());
                }
            });

        sub_goal_success_ = create_subscription<std_msgs::msg::Bool>(
            goal_success_topic_, 10,
            [this](std_msgs::msg::Bool::SharedPtr m) {
                handleGoalSuccess(m->data);
            });

        sub_collision_ = create_subscription<std_msgs::msg::Bool>(
            collision_topic_, 10,
            [this](std_msgs::msg::Bool::SharedPtr m) {
                std::lock_guard<std::mutex> l(mtx_);
                if (m->data) collision_latch_ = true;
            });

        auto period = std::chrono::duration<double>(1.0 / record_hz_);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&DatasetRecorder::recordTick, this));

        heartbeat_timer_ = create_wall_timer(
            1s, std::bind(&DatasetRecorder::heartbeatTick, this));

        start_time_ = now();
        dbg_ = std::make_shared<debugger::DebugClient>(this, true);
        dbg_->send("Recorder ready", "NODE_STATUS", "info", -1.0f);
        dbg_->send("committed=0 discarded=0", "TOTALS", "info", -1.0f);

        sendCollisionStatus(0, 0);

        RCLCPP_INFO(get_logger(),
            "Recorder ready. scenario=%s teacher=%s dir=%s rate=%.1fHz "
            "collision_topic=%s",
            scenario_.c_str(), teacher_.c_str(), run_dir_.c_str(), record_hz_,
            collision_topic_.c_str());
    }

    ~DatasetRecorder() override { finalize(); }

    bool shouldShutdown() const { return g_shutdown_requested.load(); }
    int goalsCompleted() const { return goals_completed_.load(); }
    int maxGoals() const { return max_goals_; }

    void finalize()
    {
        if (finalized_) return;
        finalized_ = true;

        {
            std::lock_guard<std::mutex> l(mtx_);
            if (!episode_buffer_.empty()) {
                RCLCPP_WARN(get_logger(),
                    "Discarding %zu unfinished samples on shutdown",
                    episode_buffer_.size());
                episode_buffer_.clear();
            }
        }

        if (csv_stream_.is_open()) {
            csv_stream_.flush();
            csv_stream_.close();
        }
        writeDescriptorSummary();

        RCLCPP_INFO(get_logger(),
            "Done. committed=%zu discarded_static=%zu "
            "collision_frames=%zu collision_events=%zu stuck_events=%zu "
            "episodes_with_collision=%zu episodes_with_stuck=%zu",
            committed_idx_.load(), discarded_idx_.load(),
            collision_frames_.load(), total_collision_events_.load(),
            stuck_events_.load(),
            episodes_with_collision_.load(), episodes_with_stuck_.load());

        if (dbg_) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "committed=%zu discarded=%zu coll_fr=%zu coll_ev=%zu "
                "stuck=%zu e_stuck=%zu",
                committed_idx_.load(), discarded_idx_.load(),
                collision_frames_.load(), total_collision_events_.load(),
                stuck_events_.load(), episodes_with_stuck_.load());
            dbg_->send(buf, "TOTALS", "info", -1.0f);
            dbg_->send("Recording stopped", "NODE_STATUS", "info", -1.0f);

            sendCollisionStatus(
                episode_collision_events_.load(),
                total_collision_events_.load());
        }
    }

private:
    struct Sample {
        std::array<uint8_t, 80 * 80> lom;
        double ts;
        int    episode_id;
        int    is_start;
        double d;
        double theta;
        double vx, vy, wz;
        double pose_x, pose_y, pose_yaw;
        double goal_x, goal_y;
        int    collision_flag;
        int    stuck_flag;
    };

    struct PoseSample {
        double t, x, y, yaw;
        double cmd_lin;
        double cmd_ang;
    };

    void prepareOutputDirectory()
    {
        auto now_t = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now_t);
        std::tm tm{};
        localtime_r(&tt, &tm);
        std::ostringstream oss;
        oss << "run_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << teacher_;
        run_name_ = oss.str();

        run_dir_ = fs::path(output_dir_) / scenario_ / run_name_;
        fs::create_directories(run_dir_);
        fs::create_directories(run_dir_ / "lom");
    }

    void openCsv()
    {
        csv_path_ = (run_dir_ / "data.csv").string();
        csv_stream_.open(csv_path_, std::ios::out);
        if (!csv_stream_.is_open()) {
            RCLCPP_FATAL(get_logger(), "Cannot open %s", csv_path_.c_str());
            rclcpp::shutdown();
            return;
        }
        csv_stream_
            << "timestamp,episode_id,is_episode_start,"
               "d,theta,vx,vy,wz,"
               "pose_x,pose_y,pose_yaw,goal_x,goal_y,lom_file,"
               "collision_flag,stuck_flag\n";
    }

    void writeDescriptorHeader()
    {
        descriptor_path_ = (run_dir_ / "descriptor.yaml").string();
        std::ofstream f(descriptor_path_);
        f << "scenario: "    << scenario_ << "\n"
          << "teacher: "     << teacher_  << "\n"
          << "record_hz: "   << record_hz_ << "\n"
          << "map_frame: "   << map_frame_ << "\n"
          << "base_frame: "  << base_frame_ << "\n"
          << "max_goals: "   << max_goals_ << "\n"
          << "lom_width: 80\n"
          << "lom_height: 80\n"
          << "lom_resolution: 0.05\n"
          << "lom_encoding: raw_0_100\n"
          << "min_cmd_magnitude: " << min_cmd_magnitude_ << "\n"
          << "collision_topic: "   << collision_topic_ << "\n"
          << "stuck_window_s: "    << stuck_window_s_ << "\n"
          << "stuck_eps_pos_m: "   << stuck_eps_pos_m_ << "\n"
          << "stuck_eps_yaw_rad: " << stuck_eps_yaw_rad_ << "\n"
          << "stuck_eps_cmd_lin: " << stuck_eps_cmd_lin_ << "\n"
          << "stuck_eps_cmd_ang: " << stuck_eps_cmd_ang_ << "\n"
          << "stuck_min_duration_s: " << stuck_min_duration_s_ << "\n"
          << "collision_lookback_frames: " << collision_lookback_frames_ << "\n"
          << "stuck_lookback_frames: "     << stuck_lookback_frames_ << "\n";
    }

    void writeDescriptorSummary()
    {
        int ep_count;
        {
            std::lock_guard<std::mutex> l(mtx_);
            ep_count = current_episode_ + 1;
        }

        std::ofstream f(descriptor_path_, std::ios::app);
        f << "num_samples: "              << committed_idx_.load()  << "\n"
          << "num_discarded_static: "     << discarded_idx_.load() << "\n"
          << "num_episodes: "             << ep_count << "\n"
          << "n_collision_frames: "       << collision_frames_.load() << "\n"
          << "n_collision_events: "       << total_collision_events_.load() << "\n"
          << "n_stuck_events: "           << stuck_events_.load() << "\n"
          << "n_episodes_with_collision: " << episodes_with_collision_.load() << "\n"
          << "n_episodes_with_stuck: "    << episodes_with_stuck_.load() << "\n";
    }

    // Persistent collision status line for the debugger.
    // duration = -1.0 means the row stays until replaced.
    void sendCollisionStatus(size_t sample_events, size_t total_events)
    {
        if (!dbg_) return;

        int total = max_goals_;
        int sample = goals_completed_.load() + 1;
        if (total > 0 && sample > total) {
            sample = total;
        }

        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "collisions in sample %d/%d detected %zu, total of collisions %zu",
            sample, total, sample_events, total_events);
        dbg_->send(buf, "INFO_Coll", "warn", -1.0f);
    }

    // Update collision counters from a per-tick latch. A rising edge
    // (transition false -> true) is one collision event. Held contacts
    // stay as a single event but keep incrementing the frame counter.
    void updateCollisionCounters(bool coll_latch)
    {
        if (coll_latch) {
            if (!was_in_collision_) {
                episode_collision_events_++;
                total_collision_events_++;
                was_in_collision_ = true;
                sendCollisionStatus(
                    episode_collision_events_.load(),
                    total_collision_events_.load());
            }
            collision_frames_++;
            episode_had_collision_ = true;
        } else {
            was_in_collision_ = false;
        }
    }

    void handleGoalSuccess(bool success)
    {
        std::vector<Sample> local;
        bool had_collision = false;
        bool had_stuck = false;
        size_t ep_events = 0;
        {
            std::lock_guard<std::mutex> l(mtx_);
            local.swap(episode_buffer_);
            had_collision = episode_had_collision_;
            had_stuck = episode_had_stuck_;
            ep_events = episode_collision_events_.load();
            episode_had_collision_ = false;
            episode_had_stuck_ = false;
        }

        if (local.empty()) {
            RCLCPP_WARN(get_logger(),
                "goal_success received with empty buffer.");
            return;
        }

        const int ep = local.front().episode_id;
        char buf[200];

        if (success) {
            for (const auto & s : local) {
                writeSampleToDisk(s);
            }
            csv_stream_.flush();

            // Only a committed episode counts as a completed goal.
            goals_completed_++;
            if (had_collision) episodes_with_collision_++;
            if (had_stuck) episodes_with_stuck_++;

            RCLCPP_INFO(get_logger(),
                "Episode %d committed: %zu samples "
                "(total=%zu, goals=%d, collision=%d, stuck=%d, coll_events=%zu)",
                ep, local.size(), committed_idx_.load(),
                goals_completed_.load(),
                static_cast<int>(had_collision),
                static_cast<int>(had_stuck),
                ep_events);

            std::snprintf(buf, sizeof(buf),
                "Ep %d: %zu committed%s%s", ep, local.size(),
                had_collision ? " [COLL]" : "",
                had_stuck ? " [STUCK]" : "");
            dbg_->send(buf, "RESULT", "info", 20.0f);

            std::snprintf(buf, sizeof(buf),
                "committed=%zu discarded=%zu coll_fr=%zu stuck=%zu",
                committed_idx_.load(), discarded_idx_.load(),
                collision_frames_.load(), stuck_events_.load());
            dbg_->send(buf, "TOTALS", "info", -1.0f);
        } else {
            RCLCPP_INFO(get_logger(),
                "Episode %d discarded: %zu samples "
                "(collision=%d, stuck=%d, coll_events=%zu)",
                ep, local.size(),
                static_cast<int>(had_collision),
                static_cast<int>(had_stuck),
                ep_events);

            std::snprintf(buf, sizeof(buf),
                "Ep %d: %zu samples discarded", ep, local.size());
            dbg_->send(buf, "RESULT", "warn", 20.0f);

            std::snprintf(buf, sizeof(buf),
                "committed=%zu discarded=%zu coll_fr=%zu stuck=%zu",
                committed_idx_.load(), discarded_idx_.load(),
                collision_frames_.load(), stuck_events_.load());
            dbg_->send(buf, "TOTALS", "info", -1.0f);
        }
    }

    void writeSampleToDisk(const Sample & s)
    {
        std::ostringstream fname;
        fname << std::setw(7) << std::setfill('0') << committed_idx_.load() << ".png";
        fs::path lom_path = run_dir_ / "lom" / fname.str();

        std::error_code ec;
        fs::create_directories(lom_path.parent_path(), ec);

        cv::Mat lom(80, 80, CV_8UC1, const_cast<uint8_t *>(s.lom.data()));
        if (!cv::imwrite(lom_path.string(), lom)) {
            RCLCPP_WARN(get_logger(), "Failed to write %s", lom_path.c_str());
            return;
        }

        csv_stream_ << std::fixed << std::setprecision(6)
                    << s.ts            << ","
                    << s.episode_id    << ","
                    << s.is_start      << ","
                    << s.d             << ","
                    << s.theta         << ","
                    << s.vx            << ","
                    << s.vy            << ","
                    << s.wz            << ","
                    << s.pose_x        << ","
                    << s.pose_y        << ","
                    << s.pose_yaw      << ","
                    << s.goal_x        << ","
                    << s.goal_y        << ","
                    << fname.str()     << ","
                    << s.collision_flag << ","
                    << s.stuck_flag    << "\n";

        committed_idx_++;
    }

    cv::Mat costmapToLom(const nav_msgs::msg::OccupancyGrid::SharedPtr & grid)
    {
        const int w = static_cast<int>(grid->info.width);
        const int h = static_cast<int>(grid->info.height);
        cv::Mat out(h, w, CV_8UC1);
        for (int i = 0; i < w * h; ++i) {
            const int8_t v = grid->data[i];
            out.data[i] = (v >= 50) ? 100 : 0;
        }
        cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
        return out;
    }

    // Stuck detection.
    bool computeStuckFlag(double px, double py, double pyaw,
                          const geometry_msgs::msg::Twist & cmd,
                          double now_s,
                          bool & just_entered)
    {
        just_entered = false;

        const double cmd_lin = std::hypot(cmd.linear.x, cmd.linear.y);
        const double cmd_ang = std::abs(cmd.angular.z);

        pose_hist_.push_back({now_s, px, py, pyaw, cmd_lin, cmd_ang});
        while (!pose_hist_.empty() &&
               (now_s - pose_hist_.front().t) > stuck_window_s_) {
            pose_hist_.pop_front();
        }

        bool translation_stalled = false;
        bool rotation_stalled = false;
        double max_cmd_lin = 0.0;
        double max_cmd_ang = 0.0;

        if (pose_hist_.size() >= 2) {
            const auto & oldest = pose_hist_.front();
            const double dpos = std::hypot(px - oldest.x, py - oldest.y);
            const double dyaw = std::abs(normalize_angle(pyaw - oldest.yaw));
            translation_stalled = (dpos < stuck_eps_pos_m_);
            rotation_stalled    = (dyaw < stuck_eps_yaw_rad_);

            for (const auto & s : pose_hist_) {
                if (s.cmd_lin > max_cmd_lin) max_cmd_lin = s.cmd_lin;
                if (s.cmd_ang > max_cmd_ang) max_cmd_ang = s.cmd_ang;
            }
        }

        const bool pushing_without_move =
            (max_cmd_lin > stuck_eps_cmd_lin_) && translation_stalled;
        const bool turning_without_move =
            (max_cmd_ang > stuck_eps_cmd_ang_) && rotation_stalled;

        const bool stuck_now = pushing_without_move || turning_without_move;

        bool stuck_flag = false;
        if (stuck_now) {
            if (!stuck_active_) {
                stuck_active_ = true;
                stuck_since_ = now();
                stuck_event_counted_ = false;
            }
            if ((now() - stuck_since_).seconds() >= stuck_min_duration_s_) {
                stuck_flag = true;
                if (!stuck_event_counted_) {
                    stuck_events_++;
                    stuck_event_counted_ = true;
                    just_entered = true;
                }
            }
        } else {
            stuck_active_ = false;
            stuck_event_counted_ = false;
        }

        return stuck_flag;
    }

    void recordTick()
    {
        if (!stabilized_) {
            if ((now() - start_time_).seconds() >= stabilization_s_) {
                stabilized_ = true;
                RCLCPP_INFO(get_logger(), "Stabilized. Recording...");
            }
            return;
        }

        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "TF %s -> %s not available: %s",
                map_frame_.c_str(), base_frame_.c_str(), ex.what());
            return;
        }

        const double px   = tf.transform.translation.x;
        const double py   = tf.transform.translation.y;
        const double pyaw = tf2::getYaw(tf.transform.rotation);

        nav_msgs::msg::OccupancyGrid::SharedPtr costmap;
        geometry_msgs::msg::Twist cmd;
        double d, theta;
        geometry_msgs::msg::PointStamped goal;
        bool have_goal, goal_active, just_changed;
        bool coll_latch;
        int episode;
        {
            std::lock_guard<std::mutex> l(mtx_);
            costmap      = last_costmap_;
            cmd          = last_cmd_;
            d            = last_d_;
            theta        = last_theta_;
            goal         = last_goal_;
            have_goal    = have_goal_;
            goal_active  = goal_active_;
            just_changed = episode_just_changed_;
            coll_latch   = collision_latch_;
            collision_latch_ = false;
            episode      = current_episode_;
            episode_just_changed_ = false;
        }

        if (!costmap) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No LOM yet. Skipping.");
            return;
        }
        if (!have_goal) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No goal yet. Skipping.");
            return;
        }
        if (!goal_active) {
            updateCollisionCounters(coll_latch);
            return;
        }

        const double cmd_norm = std::hypot(std::hypot(cmd.linear.x, cmd.linear.y),
                                            cmd.angular.z);
        const bool is_static = cmd_norm < min_cmd_magnitude_;

        const double now_s = now().seconds();
        bool just_entered_stuck = false;
        const bool stuck_flag = computeStuckFlag(
            px, py, pyaw, cmd, now_s, just_entered_stuck);

        if (is_static && !just_changed) {
            updateCollisionCounters(coll_latch);
            discarded_idx_++;
            return;
        }

        Sample s;
        cv::Mat lom = costmapToLom(costmap);
        std::memcpy(s.lom.data(), lom.data, 80 * 80);
        s.ts          = now_s;
        s.episode_id  = episode;
        s.is_start    = just_changed ? 1 : 0;
        s.d           = d;
        s.theta       = theta;
        s.vx          = cmd.linear.x;
        s.vy          = cmd.linear.y;
        s.wz          = cmd.angular.z;
        s.pose_x      = px;
        s.pose_y      = py;
        s.pose_yaw    = pyaw;
        s.goal_x      = goal.point.x;
        s.goal_y      = goal.point.y;
        s.collision_flag = coll_latch ? 1 : 0;
        s.stuck_flag     = stuck_flag ? 1 : 0;

        // Capture values used by logging before the move.
        const int coll_val  = s.collision_flag;
        const int stuck_val = s.stuck_flag;

        updateCollisionCounters(coll_latch);
        if (stuck_flag) {
            episode_had_stuck_ = true;
        }

        size_t n;
        {
            std::lock_guard<std::mutex> l(mtx_);
            episode_buffer_.push_back(std::move(s));
            n = episode_buffer_.size();
            if (n == 1) {
                current_buffer_episode_ = episode;
            }

            if (coll_latch && collision_lookback_frames_ > 0) {
                const size_t lookback = std::min(
                    static_cast<size_t>(collision_lookback_frames_), n);
                for (size_t i = n - lookback; i < n; ++i) {
                    episode_buffer_[i].collision_flag = 1;
                }
            }

            if (just_entered_stuck && stuck_lookback_frames_ > 0) {
                const size_t lookback = std::min(
                    static_cast<size_t>(stuck_lookback_frames_), n);
                for (size_t i = n - lookback; i < n; ++i) {
                    episode_buffer_[i].stuck_flag = 1;
                }
            }
        }

        if (n == 1 || n % 5 == 0) {
            RCLCPP_INFO(get_logger(),
                "  buffered=%03zu | d=%.2f th=%+.2f | cmd=(%+.2f, %+.2f, %+.2f) | coll=%d stuck=%d",
                n, d, theta, cmd.linear.x, cmd.linear.y, cmd.angular.z,
                coll_val, stuck_val);

            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "buffered=%03zu d=%.2f th=%+.2f cmd=(%+.2f,%+.2f,%+.2f) c=%d s=%d",
                n, d, theta, cmd.linear.x, cmd.linear.y, cmd.angular.z,
                coll_val, stuck_val);
            dbg_->send(buf, "BUFFER", "info", 5.0f);
        }
    }

    void heartbeatTick()
    {
        if (!dbg_) return;

        size_t buffered;
        {
            std::lock_guard<std::mutex> l(mtx_);
            buffered = episode_buffer_.size();
        }

        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "buffered=%zu committed=%zu coll_fr=%zu stuck=%zu e_stuck=%zu",
            buffered, committed_idx_.load(),
            collision_frames_.load(), stuck_events_.load(),
            episodes_with_stuck_.load());
        dbg_->send(buf, "NODE_STATUS", "info", -1.0f);

        sendCollisionStatus(episode_collision_events_.load(),
                            total_collision_events_.load());
    }

    std::string output_dir_, scenario_, teacher_, map_frame_, base_frame_;
    std::string costmap_topic_, cmd_vel_topic_, nn_goal_topic_;
    std::string current_goal_topic_, goal_active_topic_, episode_id_topic_;
    std::string goal_success_topic_, collision_topic_;
    std::string run_name_, csv_path_, descriptor_path_;
    fs::path    run_dir_;

    double record_hz_{5.0};
    double min_cmd_magnitude_{0.01};
    double stabilization_s_{3.0};
    double stuck_window_s_{2.0};
    double stuck_eps_pos_m_{0.02};
    double stuck_eps_yaw_rad_{0.10};
    double stuck_eps_cmd_lin_{0.05};
    double stuck_eps_cmd_ang_{0.10};
    double stuck_min_duration_s_{1.5};
    int    collision_lookback_frames_{5};
    int    stuck_lookback_frames_{5};
    int    max_goals_{100};

    std::ofstream csv_stream_;
    bool stabilized_{false};
    rclcpp::Time start_time_;
    std::atomic<size_t> sample_idx_{0};
    std::atomic<size_t> committed_idx_{0};
    std::atomic<size_t> discarded_idx_{0};
    std::atomic<size_t> collision_frames_{0};
    std::atomic<size_t> stuck_events_{0};
    std::atomic<size_t> episodes_with_collision_{0};
    std::atomic<size_t> episodes_with_stuck_{0};
    std::atomic<size_t> episode_collision_events_{0};
    std::atomic<size_t> total_collision_events_{0};
    std::atomic<int>    goals_completed_{0};
    bool finalized_{false};

    std::mutex mtx_;
    nav_msgs::msg::OccupancyGrid::SharedPtr last_costmap_;
    geometry_msgs::msg::Twist last_cmd_{};
    double last_d_{0.0}, last_theta_{0.0};
    geometry_msgs::msg::PointStamped last_goal_;
    bool have_goal_{false};
    bool goal_active_{false};
    int  current_episode_{0};
    int  prev_episode_id_{0};
    int  current_buffer_episode_{-1};
    bool episode_just_changed_{false};
    bool collision_latch_{false};
    bool episode_had_collision_{false};
    bool episode_had_stuck_{false};
    bool was_in_collision_{false};

    // Stuck detector state
    std::deque<PoseSample> pose_hist_;
    bool   stuck_active_{false};
    bool   stuck_event_counted_{false};
    rclcpp::Time stuck_since_;

    std::vector<Sample> episode_buffer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_costmap_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_nn_goal_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_goal_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_active_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_episode_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_goal_success_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_collision_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
    std::shared_ptr<debugger::DebugClient> dbg_;
};

int main(int argc, char ** argv)
{
    std::signal(SIGINT, sigint_handler);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DatasetRecorder>();

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);

    while (rclcpp::ok()) {
        exec.spin_some(std::chrono::milliseconds(100));
        if (node->shouldShutdown()) break;
        if (node->maxGoals() > 0 && node->goalsCompleted() >= node->maxGoals()) {
            RCLCPP_INFO(node->get_logger(),
                "Reached max_goals=%d. Closing.", node->maxGoals());
            break;
        }
    }

    node->finalize();
    rclcpp::shutdown();
    return 0;
}
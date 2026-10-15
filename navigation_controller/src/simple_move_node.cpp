#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <action_msgs/msg/goal_status.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "debugger/debug_client.hpp"

using namespace std::chrono_literals;

static constexpr int RATE = 30;

static constexpr int SM_INIT                        = 0;
static constexpr int SM_WAITING_FOR_TASK            = 1;

static constexpr int SM_GOAL_POSE_ACCEL             = 10;
static constexpr int SM_GOAL_POSE_CRUISE            = 11;
static constexpr int SM_GOAL_POSE_DECCEL            = 12;
static constexpr int SM_GOAL_POSE_CORRECT_ANGLE     = 13;
static constexpr int SM_GOAL_POSE_FINISH            = 14;
static constexpr int SM_GOAL_POSE_FAILED            = 15;

static constexpr int SM_GOAL_PATH_ACCEL             = 20;
static constexpr int SM_GOAL_PATH_CRUISE            = 21;
static constexpr int SM_GOAL_PATH_DECCEL            = 22;
static constexpr int SM_GOAL_PATH_FINAL_ALIGN       = 23;
static constexpr int SM_GOAL_PATH_FINAL_APPROACH    = 24;
static constexpr int SM_GOAL_PATH_FINISH            = 25;
static constexpr int SM_GOAL_PATH_FAILED            = 26;

class SimpleMoveNode : public rclcpp::Node
{
public:
    SimpleMoveNode() : rclcpp::Node("simple_move")
    {
        declare_parameters();
        load_parameters();

        tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        auto cb_dist = [this](std_msgs::msg::Float32::SharedPtr m) { cb_goal_dist(m); };
        auto cb_dist_angle = [this](std_msgs::msg::Float32MultiArray::SharedPtr m) { cb_goal_dist_angle(m); };
        auto cb_lateral = [this](std_msgs::msg::Float32::SharedPtr m) { cb_goal_lateral(m); };
        auto cb_path = [this](nav_msgs::msg::Path::SharedPtr m) { cb_goal_path(m); };
        auto cb_stop_msg = [this](std_msgs::msg::Empty::SharedPtr) { cb_stop(); };
        auto cb_risk = [this](std_msgs::msg::Bool::SharedPtr m) { collision_risk_ = m->data; };
        auto cb_rej = [this](geometry_msgs::msg::Vector3::SharedPtr m) { rejection_force_ = *m; };

        sub_goal_dist_       = this->create_subscription<std_msgs::msg::Float32>(
            "/simple_move/goal_dist", 1, cb_dist);
        sub_goal_dist_angle_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/simple_move/goal_dist_angle", 1, cb_dist_angle);
        sub_goal_dist_lateral_ = this->create_subscription<std_msgs::msg::Float32>(
            "/simple_move/goal_dist_lateral", 1, cb_lateral);
        sub_goal_path_       = this->create_subscription<nav_msgs::msg::Path>(
            "/navigation/simple_move/goal_path", 1, cb_path);
        sub_stop_classic_    = this->create_subscription<std_msgs::msg::Empty>(
            "/simple_move/stop", 1, cb_stop_msg);
        sub_stop_nav_        = this->create_subscription<std_msgs::msg::Empty>(
            "/navigation/stop", 1, cb_stop_msg);
        sub_stop_global_     = this->create_subscription<std_msgs::msg::Empty>(
            "/stop", 1, cb_stop_msg);
        sub_risk_classic_    = this->create_subscription<std_msgs::msg::Bool>(
            "/navigation/obs_detector/collision_risk", 10, cb_risk);
        sub_rej_classic_     = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/navigation/obs_detector/pf_rejection_force", 1, cb_rej);

        sub_goal_dist_angle_ml_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/mapless_nav/goal_dist_angle", 1, cb_dist_angle);
        sub_goal_path_ml_       = this->create_subscription<nav_msgs::msg::Path>(
            "/mapless_nav/goal_path", 1, cb_path);
        sub_stop_ml_            = this->create_subscription<std_msgs::msg::Empty>(
            "/mapless_nav/stop", 1, cb_stop_msg);
        sub_risk_ml_            = this->create_subscription<std_msgs::msg::Bool>(
            "/mapless_nav/collision_risk", 10, cb_risk);
        sub_rej_ml_             = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/mapless_nav/pf_rejection_force", 1, cb_rej);

        pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 1);
        pub_goal_reached_classic_ = this->create_publisher<action_msgs::msg::GoalStatus>(
            "/simple_move/goal_reached", 1);
        pub_goal_reached_ml_ = this->create_publisher<action_msgs::msg::GoalStatus>(
            "/mapless_nav/simple_move_goal_reached", 1);

        loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / RATE),
            [this]() { control_step(); });

        dbg_ = std::make_unique<debugger::DebugClient>(this, true);
        dbg_->send("simple_move ready", "NODE_STATUS", "info", -1.0f);
        heartbeat_timer_ = this->create_wall_timer(1s, [this]() {
            dbg_->send("Node initialized and ready.", "NODE_STATUS", "info", -1.0f);
        });

        RCLCPP_INFO(this->get_logger(),
            "simple_move: v_max=%.2f v_min=%.2f w_max=%.2f alpha=%.4f beta=%.4f pot=%s",
            max_linear_speed_, min_linear_speed_, max_angular_speed_,
            alpha_, beta_, use_pot_fields_ ? "on" : "off");
    }

    void wait_for_transforms()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for map->%s and odom->%s...",
                    base_link_.c_str(), base_link_.c_str());
        rclcpp::Rate rate(10.0);
        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());
            if (tf_buffer_->canTransform("map",  base_link_, tf2::TimePointZero) &&
                tf_buffer_->canTransform("odom", base_link_, tf2::TimePointZero))
                break;
            rate.sleep();
        }
        RCLCPP_INFO(this->get_logger(), "Transforms available.");
    }

private:
    void declare_parameters()
    {
        this->declare_parameter("max_linear_speed", 0.3);
        this->declare_parameter("min_linear_speed", 0.05);
        this->declare_parameter("max_angular_speed", 1.0);
        this->declare_parameter("control_alpha", 0.6548);
        this->declare_parameter("control_beta", 0.2);
        this->declare_parameter("linear_acceleration", 0.1);
        this->declare_parameter("fine_dist_tolerance", 0.03);
        this->declare_parameter("goal_dmin_tolerance", 0.20);
        this->declare_parameter("goal_dmax_tolerance", 0.40);
        this->declare_parameter("final_align_angle_tol", 0.20);
        this->declare_parameter("final_approach_radius", 0.50);
        this->declare_parameter("path_lookahead_m", 0.25);
        this->declare_parameter("turn_speed_far_m", 1.5);
        this->declare_parameter("turn_speed_near_m", 0.7);
        this->declare_parameter("turn_speed_factor_far", 0.6);
        this->declare_parameter("turn_speed_factor_near", 0.35);
        this->declare_parameter("angle_tolerance", 0.05);
        this->declare_parameter("use_pot_fields", false);
        this->declare_parameter("base_link_name", std::string("base_link"));
        this->declare_parameter("cmd_vel_topic", std::string("/hardware/base/cmd_vel"));
        this->declare_parameter("flip_forward_sign", false);
        this->declare_parameter("flip_lateral_sign", false);
        this->declare_parameter("flip_angular_sign", false);
    }

    void load_parameters()
    {
        max_linear_speed_       = (float)this->get_parameter("max_linear_speed").as_double();
        min_linear_speed_       = (float)this->get_parameter("min_linear_speed").as_double();
        max_angular_speed_      = (float)this->get_parameter("max_angular_speed").as_double();
        alpha_                  = (float)this->get_parameter("control_alpha").as_double();
        beta_                   = (float)this->get_parameter("control_beta").as_double();
        linear_acceleration_    = (float)this->get_parameter("linear_acceleration").as_double();
        fine_dist_tolerance_    = (float)this->get_parameter("fine_dist_tolerance").as_double();
        goal_dmin_tolerance_    = (float)this->get_parameter("goal_dmin_tolerance").as_double();
        goal_dmax_tolerance_    = (float)this->get_parameter("goal_dmax_tolerance").as_double();
        final_align_angle_tol_  = (float)this->get_parameter("final_align_angle_tol").as_double();
        final_approach_radius_  = (float)this->get_parameter("final_approach_radius").as_double();
        path_lookahead_m_       = (float)this->get_parameter("path_lookahead_m").as_double();
        turn_speed_far_m_       = (float)this->get_parameter("turn_speed_far_m").as_double();
        turn_speed_near_m_      = (float)this->get_parameter("turn_speed_near_m").as_double();
        turn_speed_factor_far_  = (float)this->get_parameter("turn_speed_factor_far").as_double();
        turn_speed_factor_near_ = (float)this->get_parameter("turn_speed_factor_near").as_double();
        angle_tolerance_        = (float)this->get_parameter("angle_tolerance").as_double();
        use_pot_fields_         = this->get_parameter("use_pot_fields").as_bool();
        base_link_              = this->get_parameter("base_link_name").as_string();
        cmd_vel_topic_          = this->get_parameter("cmd_vel_topic").as_string();
        flip_forward_sign_      = this->get_parameter("flip_forward_sign").as_bool();
        flip_lateral_sign_      = this->get_parameter("flip_lateral_sign").as_bool();
        flip_angular_sign_      = this->get_parameter("flip_angular_sign").as_bool();
    }

    void cb_goal_dist(const std_msgs::msg::Float32::SharedPtr m)
    {
        goal_distance_ = m->data;
        goal_angle_    = 0.0f;
        new_pose_      = true;
        new_path_      = false;
        move_lat_      = false;
    }

    void cb_goal_dist_angle(const std_msgs::msg::Float32MultiArray::SharedPtr m)
    {
        if (m->data.size() < 2) return;
        goal_distance_ = m->data[0];
        goal_angle_    = m->data[1];
        new_pose_      = true;
        new_path_      = false;
        move_lat_      = false;
    }

    void cb_goal_lateral(const std_msgs::msg::Float32::SharedPtr m)
    {
        goal_distance_ = m->data;
        goal_angle_    = 0.0f;
        new_pose_      = true;
        new_path_      = false;
        move_lat_      = true;
    }

    void cb_goal_path(const nav_msgs::msg::Path::SharedPtr m)
    {
        move_lat_ = false;
        if (m->poses.empty()) {
            new_pose_ = false;
            new_path_ = false;
            return;
        }
        goal_path_ = *m;
        new_pose_  = false;
        new_path_  = true;
    }

    void cb_stop()
    {
        stop_     = true;
        new_pose_ = false;
        new_path_ = false;
        move_lat_ = false;
    }

    static float wrap_pi(float a)
    {
        while (a >  M_PI) a -= 2.0f * (float)M_PI;
        while (a <= -M_PI) a += 2.0f * (float)M_PI;
        return a;
    }

    geometry_msgs::msg::Twist calculate_speeds(float rx, float ry, float rt,
                                               float gx, float gy,
                                               float vmin, float vmax,
                                               float wmax, float a, float b,
                                               bool backwards, bool lateral,
                                               bool use_pf = false,
                                               double rej_y = 0.0)
    {
        float angle_error = backwards
            ? std::atan2(ry - gy, rx - gx) - rt
            : std::atan2(gy - ry, gx - rx) - rt;
        angle_error = wrap_pi(angle_error);
        if (lateral) angle_error = wrap_pi(angle_error - (float)M_PI / 2.0f);
        if (backwards) vmax = -vmax;

        geometry_msgs::msg::Twist r;
        float v = vmax * std::exp(-(angle_error * angle_error) / a);
        if (lateral) r.linear.y = v;
        else         r.linear.x = v;
        r.angular.z = wmax * (2.0f / (1.0f + std::exp(-angle_error / b)) - 1.0f);

        if (std::fabs(r.linear.x) < vmin) r.linear.x = 0.0f;
        if (std::fabs(r.linear.y) < vmin) r.linear.y = 0.0f;

        if (use_pf && !lateral) r.linear.y = (float)rej_y;
        if (flip_forward_sign_) r.linear.x = -r.linear.x;
        if (flip_lateral_sign_) r.linear.y = -r.linear.y;
        if (flip_angular_sign_) r.angular.z = -r.angular.z;
        return r;
    }

    geometry_msgs::msg::Twist calculate_speeds_angle(float ra, float ga, float wmax, float b)
    {
        float e = wrap_pi(ga - ra);
        geometry_msgs::msg::Twist r;
        r.linear.x  = 0.0f;
        r.angular.z = wmax * (2.0f / (1.0f + std::exp(-e / b)) - 1.0f);
        if (flip_angular_sign_) r.angular.z = -r.angular.z;
        return r;
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
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF odom->%s: %s", base_link_.c_str(), e.what());
            return false;
        }
    }

    bool get_robot_pose_in_map(float& x, float& y, float& t)
    {
        try {
            auto tf = tf_buffer_->lookupTransform("map", base_link_, tf2::TimePointZero);
            x = (float)tf.transform.translation.x;
            y = (float)tf.transform.translation.y;
            const auto& q = tf.transform.rotation;
            t = std::atan2((float)q.z, (float)q.w) * 2.0f;
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF map->%s: %s", base_link_.c_str(), e.what());
            return false;
        }
    }

    void get_goal_position_wrt_odom(float gd, float ga,
                                    float& gx, float& gy, float& gt)
    {
        float rx, ry, rt;
        if (!get_robot_pose(rx, ry, rt)) { gx = gy = gt = 0.0f; return; }
        gt = wrap_pi(rt + ga);
        float off = move_lat_ ? (float)M_PI / 2.0f : 0.0f;
        gx = rx + gd * std::cos(rt + ga + off);
        gy = ry + gd * std::sin(rt + ga + off);
    }

    float get_path_total_distance() const
    {
        float d = 0.0f;
        for (size_t i = 1; i < goal_path_.poses.size(); i++) {
            float dx = (float)(goal_path_.poses[i].pose.position.x -
                               goal_path_.poses[i - 1].pose.position.x);
            float dy = (float)(goal_path_.poses[i].pose.position.y -
                               goal_path_.poses[i - 1].pose.position.y);
            d += std::sqrt(dx * dx + dy * dy);
        }
        return d;
    }

    // Estimate the turn severity ahead on the path and return a speed cap.
    // Uses the angle between the path direction at the current waypoint and
    // the path direction N waypoints ahead. Big difference means a sharp turn.
    float compute_turn_speed_cap(int idx)
    {
        int n = (int)goal_path_.poses.size();
        if (n < 4 || idx >= n - 1) return max_linear_speed_;

        int i0 = std::min(idx, n - 2);
        int i1 = std::min(idx + 1, n - 1);
        float a0 = std::atan2(
            (float)(goal_path_.poses[i1].pose.position.y -
                    goal_path_.poses[i0].pose.position.y),
            (float)(goal_path_.poses[i1].pose.position.x -
                    goal_path_.poses[i0].pose.position.x));

        int k0 = std::min(idx + 20, n - 2);
        int k1 = std::min(idx + 21, n - 1);
        float a1 = std::atan2(
            (float)(goal_path_.poses[k1].pose.position.y -
                    goal_path_.poses[k0].pose.position.y),
            (float)(goal_path_.poses[k1].pose.position.x -
                    goal_path_.poses[k0].pose.position.x));

        float delta = std::fabs(wrap_pi(a1 - a0));
        if (delta > turn_speed_far_m_)  return max_linear_speed_ * turn_speed_factor_near_;
        if (delta > turn_speed_near_m_) return max_linear_speed_ * turn_speed_factor_far_;
        return max_linear_speed_;
    }

    void get_next_goal_from_path(float rx, float ry,
                                float& gx, float& gy,
                                int& last_idx, int& next_idx)
    {
        int n = (int)goal_path_.poses.size();
        if (next_idx >= n) next_idx = n - 1;
        float px = (float)goal_path_.poses[last_idx].pose.position.x;
        float py = (float)goal_path_.poses[last_idx].pose.position.y;
        float ex = rx - px, ey = ry - py;
        float error = 0.0f;
        do {
            gx = (float)goal_path_.poses[next_idx].pose.position.x;
            gy = (float)goal_path_.poses[next_idx].pose.position.y;
            float vx = gx - px, vy = gy - py;
            float mag = std::sqrt(vx * vx + vy * vy);
            vx = mag > 0 ? vx / mag : 0;
            vy = mag > 0 ? vy / mag : 0;
            float proj = ex * vx + ey * vy;
            error = mag - proj;
            if (error < 0.25f) last_idx = next_idx;
        } while (error < 0.25f && ++next_idx < n);

        int look_idx = next_idx;
        while (look_idx < n - 1) {
            float lx = (float)goal_path_.poses[look_idx].pose.position.x;
            float ly = (float)goal_path_.poses[look_idx].pose.position.y;
            if (std::hypot(lx - rx, ly - ry) >= path_lookahead_m_) break;
            look_idx++;
        }
        gx = (float)goal_path_.poses[look_idx].pose.position.x;
        gy = (float)goal_path_.poses[look_idx].pose.position.y;
        next_idx = look_idx;
    }

    void publish_goal_status(int status)
    {
        action_msgs::msg::GoalStatus msg;
        msg.status = (int8_t)status;
        msg.goal_info.stamp = this->get_clock()->now();
        pub_goal_reached_classic_->publish(msg);
        pub_goal_reached_ml_->publish(msg);
    }

    void publish_stop()
    {
        geometry_msgs::msg::Twist t;
        pub_cmd_vel_->publish(t);
    }

    void control_step()
    {
        if (stop_) {
            stop_ = false;
            state_ = SM_INIT;
            publish_stop();
            publish_goal_status(action_msgs::msg::GoalStatus::STATUS_ABORTED);
        }
        if (new_pose_ || new_path_) state_ = SM_WAITING_FOR_TASK;

        switch (state_) {
        case SM_INIT:
            state_ = SM_WAITING_FOR_TASK;
            break;

        case SM_WAITING_FOR_TASK:
            if (new_pose_) {
                get_goal_position_wrt_odom(goal_distance_, goal_angle_, goal_x_, goal_y_, goal_t_);
                state_ = SM_GOAL_POSE_ACCEL;
                new_pose_ = false;
                attempts_ = (int)((std::fabs(goal_distance_) * 5.0f) / max_linear_speed_ * RATE +
                                  std::fabs(goal_angle_ * 5.0f) / max_angular_speed_ * RATE + RATE * 5);
            }
            if (new_path_) {
                state_ = SM_GOAL_PATH_ACCEL;
                new_path_ = false;
                prev_pose_idx_ = 0;
                next_pose_idx_ = 0;
                approach_ticks_ = 0;
                int n = (int)goal_path_.poses.size();
                global_goal_x_ = (float)goal_path_.poses[n - 1].pose.position.x;
                global_goal_y_ = (float)goal_path_.poses[n - 1].pose.position.y;
                attempts_ = (int)(get_path_total_distance() / max_linear_speed_ * 4.0f * RATE + 5 * RATE);
            }
            break;

        case SM_GOAL_POSE_ACCEL: {
            float rx, ry, rt;
            if (!get_robot_pose(rx, ry, rt)) break;
            float err = std::hypot(goal_x_ - rx, goal_y_ - ry);
            if (err < fine_dist_tolerance_) state_ = SM_GOAL_POSE_CORRECT_ANGLE;
            else if (err < current_linear_speed_ * current_linear_speed_ / (linear_acceleration_ * 5.0f)) {
                state_ = SM_GOAL_POSE_DECCEL;
                temp_k_ = current_linear_speed_ / std::sqrt(err);
            } else if (current_linear_speed_ >= max_linear_speed_) {
                current_linear_speed_ = max_linear_speed_;
                state_ = SM_GOAL_POSE_CRUISE;
            }
            if (--attempts_ <= 0) state_ = SM_GOAL_POSE_FAILED;
            auto cmd = calculate_speeds(rx, ry, rt, goal_x_, goal_y_,
                min_linear_speed_, current_linear_speed_, max_angular_speed_,
                alpha_ * 2.0f, beta_ / 4.0f, goal_distance_ < 0, move_lat_);
            pub_cmd_vel_->publish(cmd);
            current_linear_speed_ += (linear_acceleration_ * 5.0f) / RATE;
            break;
        }

        case SM_GOAL_POSE_CRUISE: {
            float rx, ry, rt;
            if (!get_robot_pose(rx, ry, rt)) break;
            float err = std::hypot(goal_x_ - rx, goal_y_ - ry);
            if (err < fine_dist_tolerance_) state_ = SM_GOAL_POSE_CORRECT_ANGLE;
            else if (err < current_linear_speed_ * current_linear_speed_ / (linear_acceleration_ * 5.0f)) {
                temp_k_ = current_linear_speed_ / std::sqrt(err);
                state_ = SM_GOAL_POSE_DECCEL;
            }
            if (--attempts_ <= 0) state_ = SM_GOAL_POSE_FAILED;
            auto cmd = calculate_speeds(rx, ry, rt, goal_x_, goal_y_,
                min_linear_speed_, current_linear_speed_, max_angular_speed_,
                alpha_ * 2.0f, beta_ / 4.0f, goal_distance_ < 0, move_lat_);
            pub_cmd_vel_->publish(cmd);
            break;
        }

        case SM_GOAL_POSE_DECCEL: {
            float rx, ry, rt;
            if (!get_robot_pose(rx, ry, rt)) break;
            float err = std::hypot(goal_x_ - rx, goal_y_ - ry);
            if (err < fine_dist_tolerance_) state_ = SM_GOAL_POSE_CORRECT_ANGLE;
            if (--attempts_ <= 0) state_ = SM_GOAL_POSE_FAILED;
            current_linear_speed_ = temp_k_ * std::sqrt(err);
            if (current_linear_speed_ < min_linear_speed_) current_linear_speed_ = min_linear_speed_;
            auto cmd = calculate_speeds(rx, ry, rt, goal_x_, goal_y_,
                min_linear_speed_, current_linear_speed_, max_angular_speed_,
                alpha_ * 2.0f, beta_ / 4.0f, goal_distance_ < 0, move_lat_,
                use_pot_fields_, rejection_force_.y);
            pub_cmd_vel_->publish(cmd);
            break;
        }

        case SM_GOAL_POSE_CORRECT_ANGLE: {
            float rx, ry, rt;
            if (!get_robot_pose(rx, ry, rt)) break;
            float err = std::fabs(wrap_pi(goal_t_ - rt));
            if (err < angle_tolerance_) state_ = SM_GOAL_POSE_FINISH;
            auto cmd = calculate_speeds_angle(rt, goal_t_, max_angular_speed_, beta_ / 4.0f);
            pub_cmd_vel_->publish(cmd);
            if (--attempts_ <= 0) state_ = SM_GOAL_POSE_FAILED;
            break;
        }

        case SM_GOAL_POSE_FINISH:
            state_ = SM_INIT;
            publish_goal_status(action_msgs::msg::GoalStatus::STATUS_SUCCEEDED);
            publish_stop();
            current_linear_speed_ = 0.0f;
            break;

        case SM_GOAL_POSE_FAILED:
            state_ = SM_INIT;
            publish_goal_status(action_msgs::msg::GoalStatus::STATUS_ABORTED);
            publish_stop();
            current_linear_speed_ = 0.0f;
            break;

        case SM_GOAL_PATH_ACCEL:
        case SM_GOAL_PATH_CRUISE:
        case SM_GOAL_PATH_DECCEL: {
            if (collision_risk_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FAILED;
                break;
            }
            float rx, ry, rt;
            if (!get_robot_pose_in_map(rx, ry, rt)) break;
            float gx, gy;
            get_next_goal_from_path(rx, ry, gx, gy, prev_pose_idx_, next_pose_idx_);
            float gerr = std::hypot(global_goal_x_ - rx, global_goal_y_ - ry);

            // Enter final align only when the path follower is close to the
            // end of the path. Prevents cutting curves mid-route.
            int n = (int)goal_path_.poses.size();
            bool near_end = next_pose_idx_ >= n - 8;
            if (near_end && gerr < final_approach_radius_) {
                attempts_ = RATE * 10;
                approach_ticks_ = 0;
                state_ = SM_GOAL_PATH_FINAL_ALIGN;
                break;
            }

            // Cap speed based on upcoming turn severity.
            float speed_cap = compute_turn_speed_cap(next_pose_idx_);
            if (current_linear_speed_ > speed_cap) current_linear_speed_ = speed_cap;

            float tol = (current_linear_speed_ < 0.1f)
                        ? goal_dmax_tolerance_
                        : goal_dmin_tolerance_;

            if (state_ == SM_GOAL_PATH_ACCEL) {
                if (gerr < tol) {
                    publish_stop();
                    state_ = SM_GOAL_PATH_FINISH;
                    break;
                }
                else if (gerr < current_linear_speed_ * current_linear_speed_ / linear_acceleration_) {
                    state_ = SM_GOAL_PATH_DECCEL;
                    temp_k_ = current_linear_speed_ / std::sqrt(gerr);
                } else if (current_linear_speed_ >= max_linear_speed_) {
                    current_linear_speed_ = max_linear_speed_;
                    state_ = SM_GOAL_PATH_CRUISE;
                }
                current_linear_speed_ += linear_acceleration_ / RATE;
                if (current_linear_speed_ > speed_cap) current_linear_speed_ = speed_cap;
            } else if (state_ == SM_GOAL_PATH_CRUISE) {
                if (gerr < tol) {
                    publish_stop();
                    state_ = SM_GOAL_PATH_FINISH;
                    break;
                }
                else if (gerr < current_linear_speed_ * current_linear_speed_ / linear_acceleration_) {
                    temp_k_ = current_linear_speed_ / std::sqrt(gerr);
                    state_ = SM_GOAL_PATH_DECCEL;
                }
            } else {
                if (gerr < tol) {
                    publish_stop();
                    state_ = SM_GOAL_PATH_FINISH;
                    break;
                }
                current_linear_speed_ = temp_k_ * std::sqrt(gerr);
                if (current_linear_speed_ < min_linear_speed_) current_linear_speed_ = min_linear_speed_;
                if (current_linear_speed_ > speed_cap) current_linear_speed_ = speed_cap;
            }
            if (--attempts_ <= 0) state_ = SM_GOAL_PATH_FAILED;

            auto cmd = calculate_speeds(rx, ry, rt, gx, gy,
                min_linear_speed_, current_linear_speed_, max_angular_speed_,
                alpha_, beta_, false, move_lat_, use_pot_fields_, rejection_force_.y);
            pub_cmd_vel_->publish(cmd);
            break;
        }

        case SM_GOAL_PATH_FINAL_ALIGN: {
            if (collision_risk_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FAILED;
                break;
            }
            float rx, ry, rt;
            if (!get_robot_pose_in_map(rx, ry, rt)) break;
            float gerr = std::hypot(global_goal_x_ - rx, global_goal_y_ - ry);

            if (gerr < goal_dmin_tolerance_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FINISH;
                break;
            }

            float target_yaw = std::atan2(global_goal_y_ - ry,
                                          global_goal_x_ - rx);
            float ang_err = wrap_pi(target_yaw - rt);
            if (std::fabs(ang_err) < final_align_angle_tol_) {
                approach_ticks_ = 0;
                state_ = SM_GOAL_PATH_FINAL_APPROACH;
                break;
            }

            if (--attempts_ <= 0) {
                // Could not align: try direct approach anyway.
                approach_ticks_ = 0;
                state_ = SM_GOAL_PATH_FINAL_APPROACH;
                break;
            }

            geometry_msgs::msg::Twist cmd;
            cmd.linear.x  = 0.0f;
            cmd.angular.z = max_angular_speed_ * 0.5f
                * (2.0f / (1.0f + std::exp(-ang_err / beta_)) - 1.0f);
            if (flip_angular_sign_) cmd.angular.z = -cmd.angular.z;
            pub_cmd_vel_->publish(cmd);
            break;
        }

        case SM_GOAL_PATH_FINAL_APPROACH: {
            if (collision_risk_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FAILED;
                break;
            }
            float rx, ry, rt;
            if (!get_robot_pose_in_map(rx, ry, rt)) break;
            float gerr = std::hypot(global_goal_x_ - rx, global_goal_y_ - ry);

            if (gerr < goal_dmin_tolerance_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FINISH;
                break;
            }

            // Accept at dmax after a short grace period to avoid chasing
            // the tight tolerance forever.
            approach_ticks_++;
            if (approach_ticks_ > RATE && gerr < goal_dmax_tolerance_) {
                publish_stop();
                state_ = SM_GOAL_PATH_FINISH;
                break;
            }

            if (--attempts_ <= 0) state_ = SM_GOAL_PATH_FAILED;

            // If heading drifted too much, realign before continuing.
            float target_yaw = std::atan2(global_goal_y_ - ry,
                                          global_goal_x_ - rx);
            float ang_err = wrap_pi(target_yaw - rt);
            if (std::fabs(ang_err) > final_align_angle_tol_ * 2.0f) {
                state_ = SM_GOAL_PATH_FINAL_ALIGN;
                break;
            }

            float v = min_linear_speed_ * 4.0f;
            if (gerr < goal_dmax_tolerance_) v = min_linear_speed_ * 2.0f;

            geometry_msgs::msg::Twist cmd;
            cmd.linear.x  = v;
            cmd.angular.z = 0.0f;
            if (flip_forward_sign_) cmd.linear.x = -cmd.linear.x;
            pub_cmd_vel_->publish(cmd);
            break;
        }

        case SM_GOAL_PATH_FINISH:
            RCLCPP_INFO(this->get_logger(), "simple_move: PATH_FINISH");
            state_ = SM_INIT;
            publish_goal_status(action_msgs::msg::GoalStatus::STATUS_SUCCEEDED);
            publish_stop();
            current_linear_speed_ = 0.0f;
            break;

        case SM_GOAL_PATH_FAILED:
            state_ = SM_INIT;
            publish_goal_status(action_msgs::msg::GoalStatus::STATUS_ABORTED);
            publish_stop();
            current_linear_speed_ = 0.0f;
            break;

        default:
            RCLCPP_ERROR(this->get_logger(), "simple_move: invalid state %d", state_);
            state_ = SM_INIT;
            break;
        }
    }

    float max_linear_speed_, min_linear_speed_, max_angular_speed_;
    float alpha_, beta_, linear_acceleration_;
    float fine_dist_tolerance_, angle_tolerance_;
    float goal_dmin_tolerance_ = 0.20f;
    float goal_dmax_tolerance_ = 0.40f;
    float final_align_angle_tol_ = 0.20f;
    float final_approach_radius_ = 0.50f;
    float path_lookahead_m_ = 0.25f;
    float turn_speed_far_m_ = 1.5f;
    float turn_speed_near_m_ = 0.7f;
    float turn_speed_factor_far_ = 0.6f;
    float turn_speed_factor_near_ = 0.35f;
    bool  use_pot_fields_;
    bool flip_forward_sign_ = false;
    bool flip_lateral_sign_ = false;
    bool flip_angular_sign_ = false;
    std::string base_link_, cmd_vel_topic_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_goal_dist_, sub_goal_dist_lateral_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_goal_dist_angle_, sub_goal_dist_angle_ml_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_goal_path_, sub_goal_path_ml_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr sub_stop_classic_, sub_stop_nav_, sub_stop_global_, sub_stop_ml_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_risk_classic_, sub_risk_ml_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr sub_rej_classic_, sub_rej_ml_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
    rclcpp::Publisher<action_msgs::msg::GoalStatus>::SharedPtr pub_goal_reached_classic_, pub_goal_reached_ml_;
    rclcpp::TimerBase::SharedPtr loop_timer_, heartbeat_timer_;
    std::unique_ptr<debugger::DebugClient> dbg_;

    int state_ = SM_INIT;
    float goal_distance_ = 0.0f, goal_angle_ = 0.0f;
    bool  new_pose_ = false, new_path_ = false;
    bool  collision_risk_ = false, move_lat_ = false;
    bool  stop_ = false;
    nav_msgs::msg::Path goal_path_;
    geometry_msgs::msg::Vector3 rejection_force_;

    float goal_x_ = 0.0f, goal_y_ = 0.0f, goal_t_ = 0.0f;
    float global_goal_x_ = 0.0f, global_goal_y_ = 0.0f;
    float current_linear_speed_ = 0.0f;
    float temp_k_ = 0.0f;
    int   attempts_ = 0;
    int   approach_ticks_ = 0;
    int   prev_pose_idx_ = 0, next_pose_idx_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleMoveNode>();
    node->wait_for_transforms();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
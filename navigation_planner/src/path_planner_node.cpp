#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <action_msgs/msg/goal_status.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <nav_msgs/srv/get_plan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "navigation_planner/path_planner.hpp"
#include "debugger/debug_client.hpp"

using namespace std::chrono_literals;

static constexpr int    STATE_RATE_HZ    = 10;
static constexpr int    SLOW_FACTOR      = 10;   // 10 Hz / 10 = 1 Hz slow loop

enum State {
    SM_INIT                     = 0,
    SM_WAITING_FOR_TASK         = 1,
    SM_CALCULATE_PATH           = 2,
    SM_CHECK_IF_INSIDE_OBS      = 3,
    SM_WAIT_MOVE_BACKWARDS      = 4,
    SM_CHECK_IF_OBSTACLES       = 5,
    SM_WAIT_FOR_NO_OBSTACLES    = 6,
    SM_ENABLE_OBS_DETECT        = 7,
    SM_WAIT_FOR_OBS_DETECT      = 8,
    SM_START_MOVE_PATH          = 9,
    SM_WAIT_FOR_MOVE_FINISHED   = 10,
    SM_CORRECT_FINAL_ANGLE      = 11,
    SM_WAIT_FOR_ANGLE_CORRECTED = 12,
    SM_FINAL                    = 13
};

class PathPlannerNode : public rclcpp::Node
{
public:
    PathPlannerNode() : rclcpp::Node("path_planner")
    {
        declare_parameters();
        load_parameters();

        tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ---- Callback group for the main loop (reentrant, so blocking service
        //      calls don't deadlock with the executor) ----
        reentrant_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        // ---- Subscriptions (canonical + mapless mirror) ----
        auto cb_pose_goal = [this](geometry_msgs::msg::PoseStamped::SharedPtr m) { cb_goal_pose(m); };
        auto cb_point_goal = [this](geometry_msgs::msg::PointStamped::SharedPtr m) { cb_goal_point(m); };
        auto cb_precomputed_path_msg = [this](nav_msgs::msg::Path::SharedPtr m) { cb_precomputed_path(m); };
        auto cb_stop_msg = [this](std_msgs::msg::Empty::SharedPtr) { stop_ = true; };
        auto cb_risk = [this](std_msgs::msg::Bool::SharedPtr m) { collision_risk_ = m->data; seen_risk_ = true;};
        auto cb_status = [this](action_msgs::msg::GoalStatus::SharedPtr m) { cb_simple_move_status(m); };
        auto cb_patience = [this](std_msgs::msg::Bool::SharedPtr m) { patience_ = m->data; };

        sub_goal_pose_     = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/nav_control/goal", 10, cb_pose_goal);
        sub_goal_point_    = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "/mapless_nav/current_goal", 10, cb_point_goal);
        sub_precomputed_path_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mapless_nav/precomputed_path", 10, cb_precomputed_path_msg);
        sub_stop_          = this->create_subscription<std_msgs::msg::Empty>(
            "/stop", 10, cb_stop_msg);
        sub_nav_stop_      = this->create_subscription<std_msgs::msg::Empty>(
            "/navigation/stop", 10, cb_stop_msg);
        sub_stop_ml_       = this->create_subscription<std_msgs::msg::Empty>(
            "/mapless_nav/stop", 10, cb_stop_msg);
        sub_risk_classic_  = this->create_subscription<std_msgs::msg::Bool>(
            "/navigation/obs_detector/collision_risk", 10, cb_risk);
        sub_risk_ml_       = this->create_subscription<std_msgs::msg::Bool>(
            "/mapless_nav/collision_risk", 10, cb_risk);
        sub_sm_status_     = this->create_subscription<action_msgs::msg::GoalStatus>(
            "/simple_move/goal_reached", 10, cb_status);
        sub_sm_status_ml_  = this->create_subscription<action_msgs::msg::GoalStatus>(
            "/mapless_nav/simple_move_goal_reached", 10, cb_status);
        sub_set_patience_  = this->create_subscription<std_msgs::msg::Bool>(
            "/navigation/set_patience", 10, cb_patience);
        sub_set_patience_ml_ = this->create_subscription<std_msgs::msg::Bool>(
            "/mapless_nav/set_patience", 10, cb_patience);

        // ---- Publications (canonical + mapless mirror) ----
        pub_goal_path_classic_    = this->create_publisher<nav_msgs::msg::Path>(
            "/navigation/simple_move/goal_path", 1);
        pub_goal_dist_angle_classic_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/simple_move/goal_dist_angle", 1);
        pub_goal_dist_angle_ml_   = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/mapless_nav/goal_dist_angle", 1);
        pub_stop_classic_         = this->create_publisher<std_msgs::msg::Empty>(
            "/simple_move/stop", 1);
        pub_stop_ml_              = this->create_publisher<std_msgs::msg::Empty>(
            "/mapless_nav/stop", 1);
        pub_obs_enable_classic_   = this->create_publisher<std_msgs::msg::Bool>(
            "/navigation/obs_detector/enable", 1);
        pub_obs_enable_ml_        = this->create_publisher<std_msgs::msg::Bool>(
            "/mapless_nav/obs_detector_enable", 1);
        pub_status_classic_       = this->create_publisher<action_msgs::msg::GoalStatus>(
            "/navigation/status", 10);
        pub_status_ml_            = this->create_publisher<action_msgs::msg::GoalStatus>(
            "/mapless_nav/nav_status", 10);

        // ---- Service clients (map_augmenter) ----
        clt_aug_map_  = this->create_client<nav_msgs::srv::GetMap>(augmented_map_service_);
        clt_aug_cost_ = this->create_client<nav_msgs::srv::GetMap>(augmented_cost_map_service_);
        clt_are_obs_  = this->create_client<std_srvs::srv::Trigger>(are_there_obs_service_);
        clt_in_obs_   = this->create_client<std_srvs::srv::Trigger>(is_in_obstacles_service_);

        // ---- Services exposed for external consumers (compat) ----
        srv_plan_static_ = this->create_service<nav_msgs::srv::GetPlan>(
            "/path_planner/plan_path_with_static",
            [this](const std::shared_ptr<nav_msgs::srv::GetPlan::Request> req,
                   std::shared_ptr<nav_msgs::srv::GetPlan::Response> resp) {
                plan_with_static(req, resp);
            },
            rmw_qos_profile_services_default,
            reentrant_group_);

        srv_plan_augmented_ = this->create_service<nav_msgs::srv::GetPlan>(
            "/path_planner/plan_path_with_augmented",
            [this](const std::shared_ptr<nav_msgs::srv::GetPlan::Request> req,
                   std::shared_ptr<nav_msgs::srv::GetPlan::Response> resp) {
                plan_with_augmented(req, resp);
            },
            rmw_qos_profile_services_default,
            reentrant_group_);

        // ---- Main loop ----
        loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / STATE_RATE_HZ),
            [this]() { tick(); },
            reentrant_group_);

        // ---- Debug ----
        dbg_ = std::make_unique<debugger::DebugClient>(this, true);
        dbg_->send("planner ready", "NODE_STATUS", "info", -1.0f);
        heartbeat_timer_ = this->create_wall_timer(1s, [this]() {
            dbg_->send("Node initialized and ready.", "NODE_STATUS", "info", -1.0f);
        });

        RCLCPP_INFO(this->get_logger(),
            "path_planner: patience=%s proximity=%.2f base=%s",
            patience_ ? "true" : "false", proximity_criterion_, base_link_name_.c_str());
    }

    void wait_for_map_augmenter()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for map_augmenter services...");
        auto wait = [this](auto cli, const std::string& name, int timeout_s) {
            if (!cli->wait_for_service(std::chrono::seconds(timeout_s)))
                RCLCPP_WARN(this->get_logger(), "Timeout waiting for %s", name.c_str());
        };
        wait(clt_aug_map_,  augmented_map_service_, 30);
        wait(clt_aug_cost_, augmented_cost_map_service_, 30);
        wait(clt_are_obs_,  are_there_obs_service_, 30);
        wait(clt_in_obs_,   is_in_obstacles_service_, 30);
        RCLCPP_INFO(this->get_logger(), "map_augmenter ready.");
    }

private:
    // ================================================================
    //  Parameters
    // ================================================================
    void declare_parameters()
    {
        this->declare_parameter("diagonal_paths", false);
        this->declare_parameter("smooth_alpha", 0.1);
        this->declare_parameter("smooth_beta", 0.9);
        this->declare_parameter("patience", true);
        this->declare_parameter("proximity_criterion", 2.0);
        this->declare_parameter("base_link_name", std::string("base_link"));
        this->declare_parameter("map_frame", std::string("map"));
        this->declare_parameter("odom_frame", std::string("odom"));
        this->declare_parameter("augmented_map_service",
            std::string("/mapless_nav/map_augmenter/get_augmented_map"));
        this->declare_parameter("augmented_cost_map_service",
            std::string("/mapless_nav/map_augmenter/get_augmented_cost_map"));
        this->declare_parameter("are_there_obs_service",
            std::string("/mapless_nav/map_augmenter/are_there_obstacles"));
        this->declare_parameter("is_in_obstacles_service",
            std::string("/mapless_nav/map_augmenter/is_inside_obstacles"));
    }

    void load_parameters()
    {
        diagonal_paths_      = this->get_parameter("diagonal_paths").as_bool();
        smooth_alpha_        = (float)this->get_parameter("smooth_alpha").as_double();
        smooth_beta_         = (float)this->get_parameter("smooth_beta").as_double();
        patience_            = this->get_parameter("patience").as_bool();
        proximity_criterion_ = (float)this->get_parameter("proximity_criterion").as_double();
        base_link_name_      = this->get_parameter("base_link_name").as_string();
        map_frame_           = this->get_parameter("map_frame").as_string();
        odom_frame_          = this->get_parameter("odom_frame").as_string();
        augmented_map_service_       = this->get_parameter("augmented_map_service").as_string();
        augmented_cost_map_service_  = this->get_parameter("augmented_cost_map_service").as_string();
        are_there_obs_service_       = this->get_parameter("are_there_obs_service").as_string();
        is_in_obstacles_service_     = this->get_parameter("is_in_obstacles_service").as_string();
    }

    // ================================================================
    //  Callbacks
    // ================================================================
    void cb_goal_pose(const geometry_msgs::msg::PoseStamped::SharedPtr m)
    {
        precomputed_path_pending_ = false;
        active_precomputed_path_ = false;
        global_goal_ = m->pose;
        new_global_goal_ = true;
        goal_has_orientation_ = true;
    }

    void cb_goal_point(const geometry_msgs::msg::PointStamped::SharedPtr m)
    {
        precomputed_path_pending_ = false;
        active_precomputed_path_ = false;
        global_goal_.position = m->point;
        // PointStamped carries no orientation -> preserve current yaw.
        float rx, ry, ryaw;
        if (get_robot_pose(rx, ry, ryaw)) {
            global_goal_.orientation.x = 0.0;
            global_goal_.orientation.y = 0.0;
            global_goal_.orientation.z = std::sin(ryaw * 0.5f);
            global_goal_.orientation.w = std::cos(ryaw * 0.5f);
        } else {
            global_goal_.orientation.w = 1.0;
        }
        new_global_goal_ = true;
        goal_has_orientation_ = false;
    }

    void cb_precomputed_path(const nav_msgs::msg::Path::SharedPtr m)
    {
        if (!m || m->poses.size() < 2) {
            RCLCPP_WARN(this->get_logger(),
                        "Ignoring precomputed path: at least two poses are required.");
            return;
        }
        if (!m->header.frame_id.empty() && m->header.frame_id != map_frame_) {
            RCLCPP_WARN(this->get_logger(),
                        "Ignoring precomputed path in frame '%s' (expected '%s').",
                        m->header.frame_id.c_str(), map_frame_.c_str());
            return;
        }

        pending_path_ = *m;
        pending_path_.header.frame_id = map_frame_;
        pending_path_.header.stamp = this->get_clock()->now();
        global_goal_ = pending_path_.poses.back().pose;
        goal_has_orientation_ = false;
        precomputed_path_pending_ = true;
        new_global_goal_ = true;

        RCLCPP_INFO(this->get_logger(),
                    "Received precomputed path (%zu poses); bypassing A* for this goal.",
                    pending_path_.poses.size());
    }

    void cb_simple_move_status(const action_msgs::msg::GoalStatus::SharedPtr m)
    {
        simple_move_status_ = m->status;
    }

    // ================================================================
    //  Helpers
    // ================================================================
    bool get_robot_pose(float& x, float& y, float& yaw)
    {
        try {
            auto tf = tf_buffer_->lookupTransform(map_frame_, base_link_name_, tf2::TimePointZero);
            x = (float)tf.transform.translation.x;
            y = (float)tf.transform.translation.y;
            const auto& q = tf.transform.rotation;
            yaw = std::atan2((float)q.z, (float)q.w) * 2.0f;
            return true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "TF %s->%s: %s",
                                 map_frame_.c_str(), base_link_name_.c_str(), e.what());
            return false;
        }
    }

    template <typename FutureT>
    bool wait_future(FutureT& f, std::chrono::milliseconds timeout)
    {
        return f.wait_for(timeout) == std::future_status::ready;
    }

    bool call_get_map(rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr cli,
                      nav_msgs::msg::OccupancyGrid& out)
    {
        auto req = std::make_shared<nav_msgs::srv::GetMap::Request>();
        auto fut = cli->async_send_request(req);
        if (!wait_future(fut, 3000ms)) return false;
        out = fut.get()->map;
        return true;
    }

    bool call_trigger(rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cli, bool& out)
    {
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto fut = cli->async_send_request(req);
        if (!wait_future(fut, 3000ms)) return false;
        out = fut.get()->success;
        return true;
    }

    void publish_status(int status, const std::string& text)
    {
        action_msgs::msg::GoalStatus msg;
        msg.status = (int8_t)status;
        msg.goal_info.stamp = this->get_clock()->now();
        pub_status_classic_->publish(msg);
        pub_status_ml_->publish(msg);
        if (text != current_status_text_) {
            current_status_text_ = text;

            if (text != "Starting new movement task") {
                dbg_->send(text, "STATE", status == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED
                                         ? "info"
                                         : (status == action_msgs::msg::GoalStatus::STATUS_ABORTED ? "warn" : "info"),
                           1.5f);
            }
        }
    }

    void publish_goal_path(const nav_msgs::msg::Path& p)
    {
        nav_msgs::msg::Path out = p;
        out.header.stamp = this->get_clock()->now();
        pub_goal_path_classic_->publish(out);
    }

    void publish_goal_dist_angle(float d, float a)
    {
        std_msgs::msg::Float32MultiArray m;
        m.data = {d, a};
        pub_goal_dist_angle_classic_->publish(m);
        pub_goal_dist_angle_ml_->publish(m);
    }

    void publish_simple_move_stop()
    {
        std_msgs::msg::Empty e;
        pub_stop_classic_->publish(e);
        pub_stop_ml_->publish(e);
    }

    void publish_obs_enable(bool on)
    {
        std_msgs::msg::Bool b;
        b.data = on;
        pub_obs_enable_classic_->publish(b);
        pub_obs_enable_ml_->publish(b);
    }

    // ================================================================
    //  External service handlers (path_planner compat)
    // ================================================================
    void plan_with_static(const std::shared_ptr<nav_msgs::srv::GetPlan::Request> req,
                          std::shared_ptr<nav_msgs::srv::GetPlan::Response> resp)
    {
        nav_msgs::msg::OccupancyGrid map, cost;
        // Reuse augmented service as the source; in this stack the static
        // map is the same as the augmented one minus obstacles.
        if (!call_get_map(clt_aug_map_, map) || !call_get_map(clt_aug_cost_, cost)) return;
        nav_msgs::msg::Path raw;
        if (!navigation_planner::PathPlanner::AStar(map, cost,
                req->start.pose, req->goal.pose, diagonal_paths_, raw)) return;
        nav_msgs::msg::Path smoothed = navigation_planner::PathPlanner::SmoothPath(
            raw, smooth_alpha_, smooth_beta_);
        nav_msgs::msg::Path decimated = navigation_planner::PathPlanner::downsample(smoothed, 0.15f);
        resp->plan = navigation_planner::PathPlanner::ChaikinSmooth(decimated, 4);
    }

    void plan_with_augmented(const std::shared_ptr<nav_msgs::srv::GetPlan::Request> req,
                            std::shared_ptr<nav_msgs::srv::GetPlan::Response> resp)
    {
        nav_msgs::msg::OccupancyGrid map, cost;
        if (!call_get_map(clt_aug_map_, map) || !call_get_map(clt_aug_cost_, cost)) return;
        nav_msgs::msg::Path raw;
        if (!navigation_planner::PathPlanner::AStar(map, cost,
                req->start.pose, req->goal.pose, diagonal_paths_, raw)) return;
        nav_msgs::msg::Path smoothed = navigation_planner::PathPlanner::SmoothPath(
            raw, smooth_alpha_, smooth_beta_);
        nav_msgs::msg::Path decimated = navigation_planner::PathPlanner::downsample(smoothed, 0.15f);
        resp->plan = navigation_planner::PathPlanner::ChaikinSmooth(decimated, 4);
    }

    // ================================================================
    //  Main state machine
    // ================================================================
    void tick()
    {
        if (stop_) {
            stop_ = false;
            if (current_status_ == action_msgs::msg::GoalStatus::STATUS_EXECUTING)
                publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                               "Stop signal received. Task cancelled.");
            state_ = SM_INIT;
            precomputed_path_pending_ = false;
            active_precomputed_path_ = false;
        }
        if (new_global_goal_) state_ = SM_WAITING_FOR_TASK;

        slow_counter_++;

        switch (state_) {
        case SM_INIT:
            if (!announced_ready_) {
                RCLCPP_INFO(this->get_logger(), "planner ready. Waiting for goal.");
                announced_ready_ = true;
            }
            state_ = SM_WAITING_FOR_TASK;
            break;

        case SM_WAITING_FOR_TASK:
            if (new_global_goal_) {
                new_global_goal_ = false;
                if (current_status_ == action_msgs::msg::GoalStatus::STATUS_EXECUTING)
                    publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                                   "Cancelling current movement.");
                goal_id_++;
                RCLCPP_INFO(this->get_logger(), "New goal id=%d", goal_id_);
                publish_status(action_msgs::msg::GoalStatus::STATUS_EXECUTING,
                               "Starting new movement task");
                near_goal_sent_ = false;
                state_ = SM_CALCULATE_PATH;
            }
            break;

        case SM_CALCULATE_PATH: {
            if (precomputed_path_pending_) {
                precomputed_path_pending_ = false;
                if (pending_path_.poses.size() < 2) {
                    active_precomputed_path_ = false;
                    publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                                   "Precomputed path is invalid (fewer than two poses).");
                    state_ = SM_INIT;
                    break;
                }
                active_precomputed_path_ = true;
                pending_path_.header.stamp = this->get_clock()->now();
                RCLCPP_INFO(this->get_logger(),
                            "Executing supplied path directly (%zu poses); A* skipped.",
                            pending_path_.poses.size());
                state_ = SM_ENABLE_OBS_DETECT;
                break;
            }

            float rx, ry, ryaw;
            if (!get_robot_pose(rx, ry, ryaw)) break;

            nav_msgs::msg::OccupancyGrid map, cost;
            bool ok = call_get_map(clt_aug_map_, map) && call_get_map(clt_aug_cost_, cost);

            nav_msgs::msg::Path raw;
            if (ok) {
                geometry_msgs::msg::Pose start_pose, goal_pose;
                start_pose.position.x = rx;
                start_pose.position.y = ry;
                goal_pose = global_goal_;
                ok = navigation_planner::PathPlanner::AStar(
                    map, cost, start_pose, goal_pose, diagonal_paths_, raw);
            }

            if (!ok) {
                RCLCPP_WARN(this->get_logger(), "Cannot compute path to (%.2f, %.2f)",
                            global_goal_.position.x, global_goal_.position.y);
                publish_simple_move_stop();
                if (!patience_) {
                    state_ = SM_CHECK_IF_INSIDE_OBS;
                } else {
                    state_ = SM_CHECK_IF_OBSTACLES;
                }
            } else {
                nav_msgs::msg::Path smoothed = navigation_planner::PathPlanner::SmoothPath(
                    raw, smooth_alpha_, smooth_beta_);
                nav_msgs::msg::Path decimated = navigation_planner::PathPlanner::downsample(smoothed, 0.15f);
                pending_path_ = navigation_planner::PathPlanner::ChaikinSmooth(decimated, 4);
                state_ = SM_ENABLE_OBS_DETECT;
            }
            break;
        }

        case SM_CHECK_IF_INSIDE_OBS: {
            bool inside = false;
            if (call_trigger(clt_in_obs_, inside) && inside) {
                RCLCPP_INFO(this->get_logger(), "Robot inside obstacle. Moving backwards.");
                publish_goal_dist_angle(-0.25f, 0.0f);
                state_ = SM_WAIT_MOVE_BACKWARDS;
            } else {
                publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                               "Cannot calc path from start to goal");
                state_ = SM_INIT;
            }
            break;
        }

        case SM_WAIT_MOVE_BACKWARDS:
            if (simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED ||
                simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_ABORTED) {
                simple_move_status_ = 0;
                state_ = SM_CALCULATE_PATH;
            }
            break;

        case SM_CHECK_IF_OBSTACLES: {
            bool are_obs = false;
            bool ok = call_trigger(clt_are_obs_, are_obs);
            if (!ok || !are_obs) {
                RCLCPP_WARN(this->get_logger(), "No temporal obstacles. Announcing failure.");
                publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                               "Cannot calculate path from start to goal point");
                state_ = SM_INIT;
            } else {
                RCLCPP_INFO(this->get_logger(), "Temporal obstacles detected. Waiting.");
                publish_status(action_msgs::msg::GoalStatus::STATUS_EXECUTING,
                               "Waiting for temporal obstacles to move");
                state_ = SM_WAIT_FOR_NO_OBSTACLES;
                slow_counter_ = 0;
            }
            break;
        }

        case SM_WAIT_FOR_NO_OBSTACLES:
            if (slow_counter_ % SLOW_FACTOR != 0) break;
            {
                bool are_obs = false;
                bool ok = call_trigger(clt_are_obs_, are_obs);
                if (!ok) {
                    publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                                   "Cannot check temporal obstacles");
                    state_ = SM_INIT;
                } else if (!are_obs) {
                    RCLCPP_INFO(this->get_logger(), "Temporal obstacles removed.");
                    state_ = SM_CALCULATE_PATH;
                }
            }
            break;

        case SM_ENABLE_OBS_DETECT:
            seen_risk_ = false;
            publish_obs_enable(true);
            RCLCPP_INFO(this->get_logger(), "obs_detector enable=true, waiting for first risk msg...");
            state_ = SM_WAIT_FOR_OBS_DETECT;
            break;

        case SM_WAIT_FOR_OBS_DETECT:
            // The obs_detect node publishes collision_risk at 30 Hz once enabled.
            // We wait until we see at least one message, tracked in `seen_risk_`.
            if (seen_risk_) {
                RCLCPP_INFO(this->get_logger(), "obs_detector is live.");
                state_ = SM_START_MOVE_PATH;
            }
            break;           

        case SM_START_MOVE_PATH:
            RCLCPP_INFO(this->get_logger(), "Sending path (%zu pts).", pending_path_.poses.size());
            collision_risk_ = false;
            publish_goal_path(pending_path_);
            simple_move_status_ = 0;
            state_ = SM_WAIT_FOR_MOVE_FINISHED;
            break;

        case SM_WAIT_FOR_MOVE_FINISHED: {
            float rx, ry, ryaw;
            if (get_robot_pose(rx, ry, ryaw)) {
                float err = std::hypot(global_goal_.position.x - rx, global_goal_.position.y - ry);
                if (err < proximity_criterion_ && !near_goal_sent_) {
                    near_goal_sent_ = true;
                    RCLCPP_INFO(this->get_logger(), "Near goal point");
                    publish_status(action_msgs::msg::GoalStatus::STATUS_EXECUTING, "Near goal point");
                }
            }
            if (simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED) {
                simple_move_status_ = 0;
                publish_obs_enable(false);
                state_ = SM_CORRECT_FINAL_ANGLE;
            } else if (collision_risk_) {
                collision_risk_ = false;
                if (active_precomputed_path_) {
                    RCLCPP_WARN(this->get_logger(),
                                "Collision risk on precomputed path; aborting to preserve route geometry.");
                    publish_simple_move_stop();
                    publish_obs_enable(false);
                    publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                                   "Collision risk on precomputed path.");
                    active_precomputed_path_ = false;
                    state_ = SM_INIT;
                } else {
                    RCLCPP_WARN(this->get_logger(), "Collision risk before goal. Replanning.");
                    state_ = SM_CALCULATE_PATH;
                }
            } else if (simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_ABORTED) {
                simple_move_status_ = 0;
                if (active_precomputed_path_) {
                    RCLCPP_WARN(this->get_logger(),
                                "Controller aborted precomputed path; not replanning a different route.");
                    publish_obs_enable(false);
                    publish_status(action_msgs::msg::GoalStatus::STATUS_ABORTED,
                                   "Controller aborted precomputed path.");
                    active_precomputed_path_ = false;
                    state_ = SM_INIT;
                } else {
                    RCLCPP_WARN(this->get_logger(), "simple_move aborted. Retrying.");
                    state_ = SM_CALCULATE_PATH;
                }
            }
            break;
        }

        case SM_CORRECT_FINAL_ANGLE: {
            RCLCPP_INFO(this->get_logger(),
                "SM_CORRECT_FINAL_ANGLE: goal_has_orientation=%d",
                (int)goal_has_orientation_);
            if (!goal_has_orientation_) {
                state_ = SM_FINAL;
                break;
            }
            float rx, ry, ryaw;
            if (!get_robot_pose(rx, ry, ryaw)) break;
            float target_yaw = std::atan2(
                (float)global_goal_.orientation.z,
                (float)global_goal_.orientation.w) * 2.0f;
            float err = target_yaw - ryaw;
            while (err >  M_PI) err -= 2.0f * (float)M_PI;
            while (err <= -M_PI) err += 2.0f * (float)M_PI;
            publish_goal_dist_angle(0.0f, err);
            state_ = SM_WAIT_FOR_ANGLE_CORRECTED;
            break;
        }

        case SM_WAIT_FOR_ANGLE_CORRECTED:
            if (simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED) {
                simple_move_status_ = 0;
                state_ = SM_FINAL;
            } else if (simple_move_status_ == action_msgs::msg::GoalStatus::STATUS_ABORTED) {
                simple_move_status_ = 0;
                RCLCPP_WARN(this->get_logger(), "Angle correction aborted. Replanning.");
                state_ = SM_CALCULATE_PATH;
            }
            break;

        case SM_FINAL:
            RCLCPP_INFO(this->get_logger(), "TASK FINISHED.");
            publish_status(action_msgs::msg::GoalStatus::STATUS_SUCCEEDED,
                           "Global goal point reached");
            active_precomputed_path_ = false;
            state_ = SM_INIT;
            break;

        default:
            RCLCPP_ERROR(this->get_logger(), "Invalid state %d", state_);
            state_ = SM_INIT;
            break;
        }
    }

    // ================================================================
    //  Members
    // ================================================================
    // Params
    bool  diagonal_paths_ = false;
    float smooth_alpha_ = 0.1f;
    float smooth_beta_  = 0.9f;
    bool  patience_ = true;
    bool goal_has_orientation_ = false;
    float proximity_criterion_ = 2.0f;
    std::string base_link_name_ = "base_link";
    std::string map_frame_ = "map";
    std::string odom_frame_ = "odom";
    std::string augmented_map_service_, augmented_cost_map_service_;
    std::string are_there_obs_service_, is_in_obstacles_service_;

    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Callback group
    rclcpp::CallbackGroup::SharedPtr reentrant_group_;

    // Subs
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_goal_point_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_precomputed_path_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr sub_stop_, sub_nav_stop_, sub_stop_ml_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_risk_classic_, sub_risk_ml_;
    rclcpp::Subscription<action_msgs::msg::GoalStatus>::SharedPtr sub_sm_status_, sub_sm_status_ml_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_set_patience_, sub_set_patience_ml_;

    // Pubs
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_goal_path_classic_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_goal_dist_angle_classic_, pub_goal_dist_angle_ml_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub_stop_classic_, pub_stop_ml_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_obs_enable_classic_, pub_obs_enable_ml_;
    rclcpp::Publisher<action_msgs::msg::GoalStatus>::SharedPtr pub_status_classic_, pub_status_ml_;

    // Clients
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr clt_aug_map_, clt_aug_cost_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clt_are_obs_, clt_in_obs_;

    // Services
    rclcpp::Service<nav_msgs::srv::GetPlan>::SharedPtr srv_plan_static_, srv_plan_augmented_;

    // Timers
    rclcpp::TimerBase::SharedPtr loop_timer_, heartbeat_timer_;

    // Debug
    std::unique_ptr<debugger::DebugClient> dbg_;
    std::string current_status_text_;

    // State
    int state_ = SM_INIT;
    int goal_id_ = -1;
    int current_status_ = 0;
    int slow_counter_ = 0;
    bool announced_ready_ = false;
    bool stop_ = false;
    bool new_global_goal_ = false;
    bool precomputed_path_pending_ = false;
    bool active_precomputed_path_ = false;
    bool near_goal_sent_ = false;
    bool seen_risk_ = false;
    bool collision_risk_ = false;
    geometry_msgs::msg::Pose global_goal_;
    nav_msgs::msg::Path pending_path_;
    int8_t simple_move_status_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathPlannerNode>();
    node->wait_for_map_augmenter();

    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();

    rclcpp::shutdown();
    return 0;
}
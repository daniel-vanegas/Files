#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PointStamped, PoseStamped, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray, Bool, Int8, Float32MultiArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
import math
import numpy as np

from debugger import DebugClient

class PotFieldsNode(Node):
    def __init__(self):
        super().__init__('pot_fields')
        
        self.dbg = DebugClient(self, verbose=True)
        self.dbg.send("Node initialized and ready.", msg_id="NODE_STATUS")
        
        # Parameters
        self.epsilon = self.declare_parameter('epsilon', 0.5).value
        self.tol     = self.declare_parameter('tol', 0.5).value
        self.eta     = self.declare_parameter('eta', 1.5).value
        self.zeta    = self.declare_parameter('zeta', 8.5).value
        self.d0      = self.declare_parameter('d0', 0.5).value
        self.alpha   = self.declare_parameter('alpha', 0.8).value
        self.beta    = self.declare_parameter('beta', 0.2).value
        self.gamma   = self.declare_parameter('gamma', 0.4).value
        self.v_max   = self.declare_parameter('v_max', 0.4).value
        self.w_max   = self.declare_parameter('w_max', 0.8).value
        self.nav_mode = self.declare_parameter('nav_mode', 'mapless').value  # 'mapless' or 'map'

        # Internal state
        self.laser_readings = []
        self.rej_cloud = np.array([0.0, 0.0])
        self.robot_vel_x = 0.0
        self.target_goal = None          # absolute goal (used in map mode)
        self.relative_goal = None        # (distance, angle) in robot frame (used in mapless mode)
        self.nav_active = False
        self.nav_fails = 0
        self.stop_nav = False

        # Stuck detection (smarter)
        self.vel_buffer = []            # rolling buffer of linear velocity
        self.dist_buffer = []           # rolling buffer of distance to goal
        self.last_dist_to_goal = None
        self.stuck_start_time = None
        self.escape_active = False
        self.escape_stage = 0
        self.escape_start_time = None
        self.escape_turn_direction = 1

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/hardware/base/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/navigation/pot_field_markers', 10)
        self.obs_enable_pub = self.create_publisher(Bool, '/navigation/obs_detector/enable', 1)
        self.head_pub = self.create_publisher(Float64MultiArray, '/hardware/head/goal_pose', 1)
        self.goal_reached_pub = self.create_publisher(Int8, '/maples_nav/goal_reached', 1)
        self.odom_reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.relative_goal_marker_pub = self.create_publisher(Marker, '/navigation/relative_goal_marker', 1)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/hardware/base/scan', self.scan_callback, 10)
        self.move_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.move_goal_callback, 10)
        self.rej_cloud_sub = self.create_subscription(Vector3, '/navigation/obs_detector/pf_rejection_force', self.rej_cloud_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/hardware/base/odom', self.odom_callback, 10)
        self.clicked_point_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked_point_callback, 10)
        self.nn_goal_sub = self.create_subscription(Float32MultiArray, '/NN_goal', self.nn_goal_callback, 10)
        self.status_sub = self.create_subscription(Int8, '/maples_nav/goal_reached', self.status_callback, 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)

        head_cmd = Float64MultiArray()
        head_cmd.data = [0.0, -0.4]
        self.head_pub.publish(head_cmd)

        enable_msg = Bool()
        enable_msg.data = False
        self.obs_enable_pub.publish(enable_msg)

        self.get_logger().info(f"Pot Fields node started (mode: {self.nav_mode})")

    # ------------------- Callbacks -------------------
    def scan_callback(self, msg):
        self.laser_readings = [
            (msg.ranges[i], msg.angle_min + i * msg.angle_increment)
            for i in range(len(msg.ranges))
            if not math.isnan(msg.ranges[i]) and not math.isinf(msg.ranges[i])
        ]

    def move_goal_callback(self, msg):
        point_msg = PointStamped()
        point_msg.header = msg.header
        point_msg.point = msg.pose.position
        self.clicked_point_callback(point_msg)

    def rej_cloud_callback(self, msg):
        self.rej_cloud = np.array([-msg.x, -msg.y])

    def odom_callback(self, msg):
        self.robot_vel_x = msg.twist.twist.linear.x

    def clicked_point_callback(self, msg):
        # Only used in map mode (absolute goal)
        if self.nav_mode != 'map':
            self.nav_active = True
            return

        if msg.header.frame_id != "odom":
            self.get_logger().error("Goal frame must be 'odom' for map mode. Set RViz fixed frame to 'odom'.")
            return

        goal_x, goal_y = msg.point.x, msg.point.y

        reset_msg = PoseWithCovarianceStamped()
        reset_msg.header.frame_id = "odom"
        reset_msg.header.stamp = self.get_clock().now().to_msg()
        reset_msg.pose.pose.orientation.w = 1.0
        self.odom_reset_pub.publish(reset_msg)
        self.get_logger().info("Odometry reset – goal is relative to current robot position.")

        self.get_logger().info(f"New absolute goal: ({goal_x:.3f}, {goal_y:.3f})")
        self.target_goal = (goal_x, goal_y)
        self.relative_goal = None
        self.nav_active = True
        self.stop_nav = False
        self.nav_fails = 0

        # Reset stuck detection
        self.vel_buffer.clear()
        self.dist_buffer.clear()
        self.last_dist_to_goal = None
        self.stuck_start_time = None
        self.escape_active = False

        enable_msg = Bool()
        enable_msg.data = True
        self.obs_enable_pub.publish(enable_msg)

    def nn_goal_callback(self, msg):
        if self.nav_mode != 'mapless':
            return
        if len(msg.data) < 2:
            return
        distance = msg.data[0]
        angle = msg.data[1]
        if distance < 0.01:
            return

        self.dbg.send(f"Distance: {distance:.2f} m, Angle: {angle:.2f} rad", msg_id="NAV_GOAL")

        self.relative_goal = (distance, angle)
        # Publish the relative goal marker
        self.publish_relative_goal_marker(distance, angle)

        self.target_goal = None
        self.stop_nav = False
        self.nav_fails = 0
        self.vel_buffer.clear()
        self.dist_buffer.clear()
        self.last_dist_to_goal = None
        self.stuck_start_time = None
        self.escape_active = False

        enable_msg = Bool()
        enable_msg.data = True
        self.obs_enable_pub.publish(enable_msg)

    # Inside pot_fields.py, after setting self.relative_goal
    def publish_relative_goal_marker(self, distance, angle):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # Position relative to robot
        marker.pose.position.x = distance * math.cos(angle)
        marker.pose.position.y = distance * math.sin(angle)
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.relative_goal_marker_pub.publish(marker)       

    def status_callback(self, msg):
        if msg.data == 4:
            self.nav_fails += 1
            self.get_logger().warn(f"Navigation failure {self.nav_fails}")
            if self.nav_fails > 2:
                self.stop_nav = True
                self.nav_fails = 0

    # ------------------- Helpers -------------------
    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            yaw = 2 * math.atan2(q.z, q.w)
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return 0.0, 0.0, 0.0

    def get_goal_wrt_robot(self, goal_x, goal_y):
        rx, ry, ra = self.get_robot_pose()
        dx = goal_x - rx
        dy = goal_y - ry
        gx = dx * math.cos(ra) + dy * math.sin(ra)
        gy = -dx * math.sin(ra) + dy * math.cos(ra)
        return gx, gy

    def distance_to_goal(self):
        if self.nav_mode == 'mapless' and self.relative_goal is not None:
            return self.relative_goal[0]
        elif self.nav_mode == 'map' and self.target_goal is not None:
            gx, gy = self.get_goal_wrt_robot(*self.target_goal)
            return math.hypot(gx, gy)
        else:
            return float('inf')

    def attraction_force(self, goal_x, goal_y):
        gx, gy = self.get_goal_wrt_robot(goal_x, goal_y)
        dist = math.hypot(gx, gy)
        if dist < 1e-6:
            return np.array([0.0, 0.0])
        unit_vec = np.array([gx, gy]) / dist
        return -self.eta * unit_vec

    def attraction_force_relative(self, distance, angle):
        """Attractive force from relative goal (already in robot frame)."""
        if distance < 1e-6:
            return np.array([0.0, 0.0])
        gx = distance * math.cos(angle)
        gy = distance * math.sin(angle)
        unit_vec = np.array([gx, gy]) / distance
        return -self.eta * unit_vec

    def rejection_force(self):
        if not self.laser_readings:
            return np.array([0.0, 0.0])
        force = np.array([0.0, 0.0])
        count = 0
        for r, angle in self.laser_readings:
            if r < self.d0:
                f = self.zeta * math.sqrt(1.0/r - 1.0/self.d0)
                force[0] += f * math.cos(angle)
                force[1] += f * math.sin(angle)
                count += 1
        if count > 0:
            force /= count
        return force

    def calculate_control(self, gx, gy):
        error_angle = math.atan2(gy, gx)
        dist = math.hypot(gx, gy)
        v = self.v_max * math.exp(-(error_angle * error_angle) / self.alpha)
        w = self.w_max * (2.0 / (1.0 + math.exp(-error_angle / self.beta)) - 1.0)
        if dist < self.tol:
            v, w = 0.0, 0.0
        return v, w

    def publish_forces_marker(self, fa, fr, f_total, id_base=0):
        self._publish_arrow(fa, [0.0, 0.0, 1.0, 1.0], id_base)
        self._publish_arrow(fr, [1.0, 0.0, 0.0, 1.0], id_base+1)
        self._publish_arrow(f_total, [0.0, 0.6, 0.0, 1.0], id_base+2)

    def _publish_arrow(self, force, color, marker_id):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pot_fields"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.scale.x, marker.scale.y, marker.scale.z = (0.07, 0.1, 0.15)
        marker.points.append(Point(x=0.0, y=0.0))
        marker.points.append(Point(x=-float(force[0]), y=-float(force[1])))
        self.marker_pub.publish(marker)

    # ------------------- Smart Stuck Detection -------------------
    def is_truly_stuck(self):
        """Returns True if robot is not making progress toward goal and has negligible motion."""
        if not self.nav_active:
            return False

        dist_now = self.distance_to_goal()
        if math.isinf(dist_now):
            return False

        # Update rolling buffers
        self.vel_buffer.append(abs(self.robot_vel_x))
        if len(self.vel_buffer) > 20:
            self.vel_buffer.pop(0)

        if self.last_dist_to_goal is not None:
            self.dist_buffer.append(self.last_dist_to_goal - dist_now)
            if len(self.dist_buffer) > 20:
                self.dist_buffer.pop(0)
        self.last_dist_to_goal = dist_now

        if len(self.vel_buffer) < 20 or len(self.dist_buffer) < 20:
            return False

        avg_vel = sum(self.vel_buffer) / len(self.vel_buffer)
        avg_progress = sum(self.dist_buffer) / len(self.dist_buffer)

        if avg_vel < 0.005 and avg_progress < 0.002:
            if self.stuck_start_time is None:
                self.stuck_start_time = self.get_clock().now()
            else:
                elapsed = (self.get_clock().now() - self.stuck_start_time).nanoseconds / 1e9
                if elapsed >= 4.0:
                    return True
        else:
            self.stuck_start_time = None
        return False

    def choose_escape_turn_direction(self):
        if not self.laser_readings:
            return 1
        left_ranges = [r for r, a in self.laser_readings if a > 0.8 and a < 1.4]
        right_ranges = [r for r, a in self.laser_readings if a < -0.8 and a > -1.4]
        avg_left = np.mean(left_ranges) if left_ranges else 0.0
        avg_right = np.mean(right_ranges) if right_ranges else 0.0
        return 1 if avg_left > avg_right else -1

    # ------------------- Escape State Machine -------------------
    def run_escape_maneuver(self):
        if not self.escape_active:
            self.escape_active = True
            self.escape_stage = 0
            self.escape_start_time = self.get_clock().now()
            self.escape_turn_direction = self.choose_escape_turn_direction()
            self.get_logger().warn("Truly stuck – starting escape maneuver")
            self.dbg.send("Status: ESCAPING", msg_id="STATE")
            return

        now = self.get_clock().now()
        elapsed = (now - self.escape_start_time).nanoseconds / 1e9
        cmd = Twist()

        if self.escape_stage == 0:          # Reverse
            if elapsed < 1.0:
                cmd.linear.x = -0.15
            else:
                self.escape_stage = 1
                self.escape_start_time = now
        elif self.escape_stage == 1:        # Turn
            if elapsed < 1.5:
                cmd.angular.z = 0.8 * self.escape_turn_direction
            else:
                self.escape_stage = 2
                self.escape_start_time = now
        elif self.escape_stage == 2:        # Forward
            if elapsed < 2.5:
                cmd.linear.x = 0.2
            else:
                # Escape complete – reset odometry to current robot pose
                reset_msg = PoseWithCovarianceStamped()
                reset_msg.header.frame_id = "odom"
                reset_msg.header.stamp = self.get_clock().now().to_msg()
                reset_msg.pose.pose.orientation.w = 1.0
                self.odom_reset_pub.publish(reset_msg)
                self.get_logger().info("Odometry reset after escape – drift eliminated")

                self.escape_active = False
                self.stuck_start_time = None
                self.vel_buffer.clear()
                self.dist_buffer.clear()
                self.get_logger().info("Escape finished – resuming navigation")
                self.dbg.send("Status: NAVIGATING", msg_id="STATE")
                return

        self.cmd_vel_pub.publish(cmd)

    # ------------------- Main Control Loop -------------------
    def control_loop(self):
        try:
            # 1. Stuck detection
            self.dbg.send("Node initialized and ready.", msg_id="NODE_STATUS")
            if not self.escape_active and self.is_truly_stuck():
                self.escape_active = True
                self.escape_stage = 0
                self.escape_start_time = self.get_clock().now()
                self.escape_turn_direction = self.choose_escape_turn_direction()
                self.get_logger().warn("Truly stuck – starting escape maneuver")
                self.dbg.send("Status: ESCAPING", msg_id="STATE")

            # 2. Escape mode
            if self.escape_active:
                self.run_escape_maneuver()
                return

            # 3. Normal navigation
            if not self.nav_active:
                self.cmd_vel_pub.publish(Twist())
                self.dbg.send("IDLE - Waiting for goal", msg_id="STATE")
                return
            else:
                self.dbg.send("Status: NAVIGATING", msg_id="STATE")

            # ---------- Mapless mode (relative goal) ----------
            if self.nav_mode == 'mapless' and self.relative_goal is not None:
                distance, angle = self.relative_goal
                gx = distance * math.cos(angle)
                gy = distance * math.sin(angle)

                self.dbg.send(f"Distance: {distance:.2f} m, Angle: {angle:.3f} rad", msg_id="NAV_GOAL")

                if self.stop_nav:
                    self.get_logger().info("Navigation stopped due to repeated failures")
                    self.nav_active = False
                    self.cmd_vel_pub.publish(Twist())
                    return

                # Attractive force from relative goal
                fa = self.attraction_force_relative(distance, angle)
                fr_lidar = self.rejection_force()
                fr_cloud = self.gamma * self.rej_cloud
                fr_total = fr_lidar + fr_cloud
                ftotal = fa + fr_total

                dx = -ftotal[0] * self.epsilon
                dy = -ftotal[1] * self.epsilon
                v, w = self.calculate_control(dx, dy)

                if distance < self.tol:
                    self.dbg.send("Goal Reached", msg_id="DEBUG", ttl=5.0)
                    self.nav_active = False
                    self.relative_goal = None
                    self.cmd_vel_pub.publish(Twist())
                    status = Int8()
                    status.data = 3
                    self.goal_reached_pub.publish(status)
                    disable_msg = Bool()
                    disable_msg.data = False
                    self.obs_enable_pub.publish(disable_msg)
                    # Delete the marker by publishing an empty marker with DELETE action
                    delete_marker = Marker()
                    delete_marker.header.frame_id = "base_link"
                    delete_marker.action = Marker.DELETEALL
                    self.relative_goal_marker_pub.publish(delete_marker)
                    return

                cmd = Twist()
                cmd.linear.x = v
                cmd.angular.z = w
                cmd.linear.y = -self.gamma * self.rej_cloud[1]
                self.cmd_vel_pub.publish(cmd)
                self.publish_forces_marker(fa, fr_total, ftotal)
                return

            # ---------- Map mode (absolute goal) ----------
            elif self.nav_mode == 'map' and self.target_goal is not None:
                goal_x, goal_y = self.target_goal
                gx, gy = self.get_goal_wrt_robot(goal_x, goal_y)
                distance = math.hypot(gx, gy)
                angle_rad = math.atan2(gy, gx)

                self.dbg.send(f"Distance: {distance:.2f} m, Angle: {angle_rad:.3f} rad", msg_id="NAV_GOAL")

                if self.stop_nav:
                    self.get_logger().info("Navigation stopped due to repeated failures")
                    self.nav_active = False
                    self.cmd_vel_pub.publish(Twist())
                    return

                fa = self.attraction_force(goal_x, goal_y)
                fr_lidar = self.rejection_force()
                fr_cloud = self.gamma * self.rej_cloud
                fr_total = fr_lidar + fr_cloud
                ftotal = fa + fr_total

                dx = -ftotal[0] * self.epsilon
                dy = -ftotal[1] * self.epsilon
                v, w = self.calculate_control(dx, dy)

                if distance < self.tol:
                    self.dbg.send("SUCCESS - Goal Reached", msg_id="STATE")

                    self.nav_active = False
                    self.target_goal = None
                    self.cmd_vel_pub.publish(Twist())
                    status = Int8()
                    status.data = 3
                    self.goal_reached_pub.publish(status)
                    disable_msg = Bool()
                    disable_msg.data = False
                    self.obs_enable_pub.publish(disable_msg)
                    return

                cmd = Twist()
                cmd.linear.x = v
                cmd.angular.z = w
                cmd.linear.y = -self.gamma * self.rej_cloud[1]
                self.cmd_vel_pub.publish(cmd)
                self.publish_forces_marker(fa, fr_total, ftotal)
                return

            # No valid goal in selected mode
            self.cmd_vel_pub.publish(Twist())
            self.dbg.send("IDLE - No goal available", msg_id="STATE")

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.nav_active = False
            self.cmd_vel_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PotFieldsNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node terminated due to error: {e}")
    finally:
        if node is not None:
            node.cmd_vel_pub.publish(Twist())
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
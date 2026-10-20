#!/usr/bin/env python3
import math
import array
import os
import rclpy
from rclpy.node import Node
from controller import Supervisor
from geometry_msgs.msg import Twist, TransformStamped, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan, Image, CameraInfo
from std_msgs.msg import Float64, Float64MultiArray, Bool
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ('1', 'true', 'yes', 'on')


class BaseHardware:
    """Wheel motors, odometry and laser."""

    def __init__(self, robot, time_step):
        self.robot = robot
        self.time_step = time_step

        # Robot geometry
        self.wheel_radius = 0.06
        self.L = 0.165

        # Speed commands
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_w = 0.0
        self.actual_vx = 0.0
        self.actual_vy = 0.0
        self.actual_w = 0.0
        self.max_acc = [2.0, 2.0, 4.0]

        # Odometry
        self.x = 0.0
        self.y = 1.0
        self.theta = 0.0

        # Wheels
        self.wheels = []
        self.wheel_sensors = []
        wheel_names = ['wheel0_joint', 'wheel1_joint', 'wheel2_joint']
        for name in wheel_names:
            motor = robot.getDevice(name)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.wheels.append(motor)

            sensor = robot.getDevice(name + "_sensor")
            if sensor:
                sensor.enable(time_step)
                self.wheel_sensors.append(sensor)
            else:
                self.wheel_sensors.append(None)

        # Lidar
        self.lidar = robot.getDevice('URG-04LX')
        if self.lidar:
            self.lidar.enable(time_step)
            self.lidar.enablePointCloud()

    def set_target_velocity(self, vx, vy, w):
        self.target_vx = vx
        self.target_vy = vy
        self.target_w = w

    def accelerate(self, dt):
        # alpha 0.6 -> time constant ~28 ms at 60 Hz.
        alpha = 0.6
        self.actual_vx += alpha * (self.target_vx - self.actual_vx)
        self.actual_vy += alpha * (self.target_vy - self.actual_vy)
        self.actual_w += alpha * (self.target_w - self.actual_w)

    def update(self, dt):
        self.accelerate(dt)

        vx, vy, w_target = self.actual_vx, self.actual_vy, self.actual_w

        sin60 = math.sin(math.pi / 3)
        cos60 = math.cos(math.pi / 3)

        v_wheel0 = (-sin60 * vx + cos60 * vy + self.L *
                    w_target) / self.wheel_radius
        v_wheel1 = (-vy + self.L * w_target) / self.wheel_radius
        v_wheel2 = (sin60 * vx + cos60 * vy + self.L *
                    w_target) / self.wheel_radius

        # Limit wheel speeds. vx_max ~ (R / sin60) * MAX_WHEEL_SPEED.
        # MAX 18 allows vx up to ~1.25 m/s.
        MAX_WHEEL_SPEED = 10
        v_wheel0 = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_wheel0))
        v_wheel1 = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_wheel1))
        v_wheel2 = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_wheel2))

        self.wheels[0].setVelocity(v_wheel0)
        self.wheels[1].setVelocity(v_wheel1)
        self.wheels[2].setVelocity(v_wheel2)

        dx = (vx * math.cos(self.theta) - vy * math.sin(self.theta)) * dt
        dy = (vx * math.sin(self.theta) + vy * math.cos(self.theta)) * dt
        self.x += dx
        self.y += dy
        self.theta += w_target * dt

    def get_odom(self):
        return self.x, self.y, self.theta

    def reset_odom(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.actual_vx = 0.0
        self.actual_vy = 0.0
        self.actual_w = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_w = 0.0

    def get_wheel_positions(self):
        positions = []
        for i in range(3):
            if self.wheel_sensors[i]:
                positions.append(self.wheel_sensors[i].getValue())
            else:
                positions.append(0.0)
        return positions

    def get_laser_scan(self):
        if not self.lidar:
            return None
        ranges = self.lidar.getRangeImage()
        scan = LaserScan()
        scan.angle_min = -2.094395
        scan.angle_max = 2.094395
        scan.angle_increment = (
            scan.angle_max - scan.angle_min) / (len(ranges) - 1)
        scan.range_min = 0.06
        scan.range_max = 4.1
        scan.ranges = list(reversed(ranges))
        return scan


class HeadHardware:
    """Head pan/tilt and Kinect cameras."""

    def __init__(self, robot, time_step):
        self.robot = robot
        self.time_step = time_step

        self.pan_joint = robot.getDevice('head_pan_joint')
        self.tilt_joint = robot.getDevice('head_tilt_joint')
        self.pan_sensor = robot.getDevice('head_pan_joint_sensor')
        self.tilt_sensor = robot.getDevice('head_tilt_joint_sensor')

        if self.pan_joint:
            self.pan_joint.setPosition(0.0)
            max_vel = self.pan_joint.getMaxVelocity()
            self.pan_joint.setVelocity(max_vel if max_vel > 0 else 3)
        if self.tilt_joint:
            self.tilt_joint.setPosition(0.0)
            max_vel = self.tilt_joint.getMaxVelocity()
            self.tilt_joint.setVelocity(max_vel if max_vel > 0 else 3)
        if self.pan_sensor:
            self.pan_sensor.enable(time_step)
        if self.tilt_sensor:
            self.tilt_sensor.enable(time_step)

        self.kinect_color = robot.getDevice("camera_color")
        if self.kinect_color:
            self.kinect_color.enable(time_step)
            self.color_width = self.kinect_color.getWidth()
            self.color_height = self.kinect_color.getHeight()
            self.color_fov = self.kinect_color.getFov()
            self.color_encoding = "bgra8"
            self.color_bytes_per_pixel = 4
            self.color_step = self.color_width * self.color_bytes_per_pixel
        else:
            self.color_width = self.color_height = 0

        self.kinect_depth = robot.getDevice("camera_range")
        if self.kinect_depth:
            self.kinect_depth.enable(time_step)
            self.depth_width = self.kinect_depth.getWidth()
            self.depth_height = self.kinect_depth.getHeight()
            self.depth_fov = self.kinect_depth.getFov()
            self.depth_encoding = "32FC1"
            self.depth_bytes_per_pixel = 4
            self.depth_step = self.depth_width * self.depth_bytes_per_pixel
        else:
            self.depth_width = self.depth_height = 0

        self.target_pan = 0.0
        self.target_tilt = 0.0

    def set_target_pan(self, pan):
        self.target_pan = pan
        if self.pan_joint:
            self.pan_joint.setPosition(self.target_pan)

    def set_target_tilt(self, tilt):
        self.target_tilt = tilt
        if self.tilt_joint:
            self.tilt_joint.setPosition(self.target_tilt)

    def get_joint_positions(self):
        pan = self.pan_sensor.getValue() if self.pan_sensor else 0.0
        tilt = self.tilt_sensor.getValue() if self.tilt_sensor else 0.0
        return [pan, tilt]

    def get_color_image(self):
        if not self.kinect_color:
            return None
        data = self.kinect_color.getImage()
        if data and len(data) == self.color_width * self.color_height * self.color_bytes_per_pixel:
            img = Image()
            img.height = self.color_height
            img.width = self.color_width
            img.encoding = self.color_encoding
            img.is_bigendian = False
            img.step = self.color_step
            img.data = data
            return img
        return None

    def get_depth_image(self):
        if not self.kinect_depth:
            return None
        depth_floats = self.kinect_depth.getRangeImage()
        if depth_floats and len(depth_floats) == self.depth_width * self.depth_height:
            depth_bytes = array.array('f', depth_floats).tobytes()
            img = Image()
            img.height = self.depth_height
            img.width = self.depth_width
            img.encoding = self.depth_encoding
            img.is_bigendian = False
            img.step = self.depth_step
            img.data = depth_bytes
            return img
        return None

    def get_camera_info(self, width, height, fov, frame_id):
        if width == 0 or height == 0:
            return None
        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        fx = width / (2.0 * math.tan(fov / 2.0))
        fy = fx
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.header.frame_id = frame_id
        return info


class TorsoHardware:
    """Torso lift joint."""

    def __init__(self, robot, time_step):
        self.robot = robot
        self.time_step = time_step

        self.torso_motor = robot.getDevice('torso_joint')
        self.torso_sensor = robot.getDevice('torso_joint_sensor')
        if self.torso_motor:
            self.torso_motor.setPosition(0.3014)
            self.torso_motor.setVelocity(10)
        if self.torso_sensor:
            self.torso_sensor.enable(time_step)

        self.target_position = 0.301

    def set_target_position(self, pos):
        self.target_position = pos
        if self.torso_motor:
            self.torso_motor.setPosition(self.target_position)

    def get_joint_position(self):
        return self.torso_sensor.getValue() if self.torso_sensor else 0.0


class JustinaController(Node):
    def __init__(self, robot: Supervisor):
        super().__init__(
            'justina_controller',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Throttle parameters. Env vars allow the launch file to set
        # them via the Webots process, and they are also visible as ROS
        # parameters under /justina_controller.
        self.declare_parameter(
            'camera_publish_hz',
            _env_float('JUSTINA_CAMERA_HZ', 10.0))
        self.declare_parameter(
            'camera_enable',
            _env_bool('JUSTINA_CAMERA_ENABLE', True))
        self.declare_parameter(
            'scan_publish_divider',
            _env_int('JUSTINA_SCAN_DIVIDER', 2))

        self.camera_publish_hz = float(self.get_parameter(
            'camera_publish_hz').value)
        self.camera_enable = bool(self.get_parameter(
            'camera_enable').value)
        self.scan_publish_divider = int(self.get_parameter(
            'scan_publish_divider').value)

        self.last_camera_pub_time = 0.0
        self._scan_counter = 0

        # Hardware modules
        self.base = BaseHardware(robot, self.time_step)
        self.head = HeadHardware(robot, self.time_step)
        self.torso = TorsoHardware(robot, self.time_step)
        gripper_poses = {
            'right_gripper_L_joint': 0.7,
            'right_gripper_R_joint': 0.7,
            'left_gripper_L_joint':  0.7,
            'left_gripper_R_joint':  0.7,
        }
        self.gripper_poses = gripper_poses
        self.gripper_joints = []
        self.gripper_names = []
        for name, pos in gripper_poses.items():
            dev = robot.getDevice(name)
            if dev:
                dev.setPosition(pos)
                dev.setVelocity(1.0)
                self.gripper_joints.append(dev)
                self.gripper_names.append(name)
        self.set_initial_head_pose()   

        # Collision detector. Optional: older worlds may not have it.
        self.collision_sensor = robot.getDevice('collision_detector')
        if self.collision_sensor:
            self.collision_sensor.enable(self.time_step)
            self.get_logger().info("collision_detector enabled")
        else:
            self.get_logger().warn(
                "collision_detector device not found in world. "
                "Collision flag will stay 0.")

        # Subscriptions
        self.create_subscription(Twist, '/hardware/base/cmd_vel',
                                 self.cmd_vel_callback, 10)
        self.create_subscription(Float64, '/head/pan_cmd',
                                 self.head_pan_callback, 10)
        self.create_subscription(Float64, '/head/tilt_cmd',
                                 self.head_tilt_callback, 10)
        self.create_subscription(Float64, '/torso/cmd_pos',
                                 self.torso_callback, 10)
        self.create_subscription(Float64MultiArray, '/hardware/head/goal_pose',
                                 self.head_goal_callback, 10)

        # Teleport
        self.create_subscription(Pose2D, '/webots/teleport',
                                 self.teleport_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry, '/hardware/base/odom', 10)
        self.laser_pub = self.create_publisher(
            LaserScan, '/hardware/base/scan', 10)
        self.joint_pub = self.create_publisher(
            JointState, '/hardware/joint_states', 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.color_pub = self.create_publisher(
            Image, '/hardware/camera/image_raw', 10)
        self.color_info_pub = self.create_publisher(
            CameraInfo, '/hardware/camera/camera_info', 10)
        self.depth_pub = self.create_publisher(
            Image, '/hardware/camera/depth/image_raw', 10)
        self.depth_info_pub = self.create_publisher(
            CameraInfo, '/hardware/camera/depth/camera_info', 10)
        self.teleport_done_pub = self.create_publisher(
            Bool, '/webots/teleport_done', 10)
        self.collision_pub = self.create_publisher(
            Bool, '/navigation/collision', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_sim_time = None
        self.get_logger().info(
            f"Justina controller initialized "
            f"(camera_hz={self.camera_publish_hz}, "
            f"camera_enable={self.camera_enable}, "
            f"scan_divider={self.scan_publish_divider})")

    def set_initial_head_pose(self):
        pan_target = 0.0
        tilt_target = -0.5

        if self.head.pan_joint:
            pan_min = self.head.pan_joint.getMinPosition()
            pan_max = self.head.pan_joint.getMaxPosition()
            if pan_min is not None and pan_max is not None:
                pan_target = max(pan_min, min(pan_max, pan_target))
        if self.head.tilt_joint:
            tilt_min = self.head.tilt_joint.getMinPosition()
            tilt_max = self.head.tilt_joint.getMaxPosition()
            if tilt_min is not None and tilt_max is not None:
                tilt_target = max(tilt_min, min(tilt_max, tilt_target))

        self.head.set_target_pan(pan_target)
        self.head.set_target_tilt(tilt_target)
        self.get_logger().info(
            f"Initial head pose set: pan={pan_target:.3f}, tilt={tilt_target:.3f}")

    def cmd_vel_callback(self, msg):
        self.base.set_target_velocity(
            msg.linear.x, msg.linear.y, msg.angular.z)
        if abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01:
            for name, dev in zip(self.gripper_names, self.gripper_joints):
                dev.setPosition(self.gripper_poses[name])

    def set_grippers_closed(self):
        joints = {
            'right_gripper_L_joint': 0.0,
            'right_gripper_R_joint': 0.0,
            'left_gripper_L_joint':  0.0,
            'left_gripper_R_joint':  0.0,
        }
        for name, pos in joints.items():
            joint = self.robot.getDevice(name)
            if joint:
                joint.setPosition(pos)
                joint.setVelocity(1.0)

    def teleport_callback(self, msg: Pose2D):
        try:
            robot_node = self.robot.getSelf()
            if robot_node is None:
                self.get_logger().error("getSelf() returned None.")
                self.teleport_done_pub.publish(Bool(data=False))
                return

            trans_field = robot_node.getField('translation')
            rot_field = robot_node.getField('rotation')

            if trans_field is None or rot_field is None:
                self.get_logger().error(
                    "Robot fields 'translation' or 'rotation' not found.")
                self.teleport_done_pub.publish(Bool(data=False))
                return

            cur = trans_field.getSFVec3f()
            new_z = cur[2] if cur is not None else 0.06

            trans_field.setSFVec3f([float(msg.x), float(msg.y), float(new_z)])
            rot_field.setSFRotation([0.0, 0.0, 1.0, float(msg.theta)])
            robot_node.resetPhysics()

            self.base.reset_odom(0.0, 0.0, 0.0)

            self.get_logger().info(
                f"Teleport -> ({msg.x:.3f}, {msg.y:.3f}, {msg.theta:.3f})")

            self.teleport_done_pub.publish(Bool(data=True))

        except Exception as e:
            self.get_logger().error(f"Teleport failed: {e}")
            self.teleport_done_pub.publish(Bool(data=False))

    def head_goal_callback(self, msg):
        if len(msg.data) >= 2:
            pan = msg.data[0]
            tilt = msg.data[1]

            if self.head.pan_joint:
                pan_min = self.head.pan_joint.getMinPosition()
                pan_max = self.head.pan_joint.getMaxPosition()
                if pan_min is not None and pan_max is not None:
                    pan = max(pan_min, min(pan_max, pan))
            if self.head.tilt_joint:
                tilt_min = self.head.tilt_joint.getMinPosition()
                tilt_max = self.head.tilt_joint.getMaxPosition()
                if tilt_min is not None and tilt_max is not None:
                    tilt = max(tilt_min, min(tilt_max, tilt))

            self.head.set_target_pan(pan)
            self.head.set_target_tilt(tilt)
            self.get_logger().info(
                f"Head goal received: pan={pan:.3f}, tilt={tilt:.3f}")
        else:
            self.get_logger().warn(
                "Received head goal pose with insufficient data")

    def head_pan_callback(self, msg):
        self.head.set_target_pan(msg.data)

    def head_tilt_callback(self, msg):
        self.head.set_target_tilt(msg.data)

    def torso_callback(self, msg):
        self.torso.set_target_position(msg.data)

    def step(self):
        if self.robot.step(self.time_step) == -1:
            return False

        sim_time = self.robot.getTime()
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        stamp = Time(sec=sec, nanosec=nanosec)

        self.clock_pub.publish(Clock(clock=stamp))

        if self.last_sim_time is None:
            dt = 0.0
        else:
            dt = sim_time - self.last_sim_time
        self.last_sim_time = sim_time

        self.base.update(dt)

        self.publish_odometry(stamp)
        self.publish_joint_states(stamp)
        self.publish_laser_scan(stamp)
        self.publish_camera_images(stamp)
        self.publish_collision()

        rclpy.spin_once(self, timeout_sec=0)

        return True

    def publish_collision(self):
        if not self.collision_sensor:
            return
        coll = self.collision_sensor.getValue() > 0.5
        self.collision_pub.publish(Bool(data=bool(coll)))

    def publish_odometry(self, stamp):
        x, y, theta = self.base.get_odom()
        qz = math.sin(theta / 2)
        qw = math.cos(theta / 2)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def publish_joint_states(self, stamp):
        js = JointState()
        js.header.stamp = stamp
        js.name = [
            'wheel0_joint', 'wheel1_joint', 'wheel2_joint',
            'torso_joint',
            'head_pan_joint', 'head_tilt_joint',
            'lf_shoulder_joint', 'ls_shoulder_joint',
            'l_arm_twist_joint', 'l_elbow_joint',
            'l_wrist_joint', 'l_hand_joint',
            'left_gripper_R_joint', 'left_gripper_L_joint',
            'rf_shoulder_joint', 'rs_shoulder_joint',
            'r_arm_twist_joint', 'r_elbow_joint',
            'r_wrist_joint', 'r_hand_joint',
            'right_gripper_R_joint', 'right_gripper_L_joint'
        ]
        positions = []
        positions.extend(self.base.get_wheel_positions())
        positions.append(self.torso.get_joint_position())
        positions.extend(self.head.get_joint_positions())
        positions.extend([0.0] * (len(js.name) - (3 + 1 + 2)))
        js.position = positions
        self.joint_pub.publish(js)

    def publish_laser_scan(self, stamp):
        if self.scan_publish_divider <= 1:
            scan = self.base.get_laser_scan()
            if scan:
                scan.header.stamp = stamp
                scan.header.frame_id = "laser_link"
                self.laser_pub.publish(scan)
            return

        self._scan_counter += 1
        if self._scan_counter % self.scan_publish_divider != 0:
            return

        scan = self.base.get_laser_scan()
        if scan:
            scan.header.stamp = stamp
            scan.header.frame_id = "laser_link"
            self.laser_pub.publish(scan)

    def publish_camera_images(self, stamp):
        if not self.camera_enable:
            return

        sim_time = self.robot.getTime()
        if (sim_time - self.last_camera_pub_time
                < 1.0 / self.camera_publish_hz):
            return
        self.last_camera_pub_time = sim_time

        # Color image
        color_img = self.head.get_color_image()
        if color_img:
            color_img.header.stamp = stamp
            color_img.header.frame_id = "camera_color_frame"
            self.color_pub.publish(color_img)

            info = self.head.get_camera_info(
                self.head.color_width, self.head.color_height,
                self.head.color_fov, "camera_color_frame")
            if info:
                info.header.stamp = stamp
                self.color_info_pub.publish(info)

        # Depth image
        depth_img = self.head.get_depth_image()
        if depth_img:
            depth_img.header.stamp = stamp
            depth_img.header.frame_id = "camera_depth_frame"
            self.depth_pub.publish(depth_img)

            info = self.head.get_camera_info(
                self.head.depth_width, self.head.depth_height,
                self.head.depth_fov, "camera_depth_frame")
            if info:
                info.header.stamp = stamp
                self.depth_info_pub.publish(info)


def main():
    rclpy.init()
    robot = Supervisor()
    controller = JustinaController(robot)

    while controller.step():
        pass

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

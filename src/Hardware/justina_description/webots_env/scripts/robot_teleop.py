#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray, Float64, Empty, Bool
from geometry_msgs.msg import Twist


class JoystickTeleop(Node):
    def __init__(self):
        super().__init__('joystick_teleop')

        # Declare parameters
        self.declare_parameter('timer_freq', 20.0)
        self.declare_parameter('forward_scale', 0.5)
        self.declare_parameter('rotate_scale', 0.8)
        self.declare_parameter('lateral_scale', 0.5)
        self.declare_parameter('trigger_deadzone', 0.1)
        self.declare_parameter('trigger_activation', 0.2)
        self.declare_parameter('stick_deadzone', 0.2)

        self.declare_parameter('lb_button_index', 4)
        self.declare_parameter('rb_button_index', 5)
        self.declare_parameter('rb_speed_multiplier', 2.0)

        self.declare_parameter('debug', False)
        self.declare_parameter('control_always_enable', False)
        self.debug = self.get_parameter('debug').value
        self.control_always_enable = self.get_parameter(
            'control_always_enable').value

        # Timer
        freq_hz = self.get_parameter('timer_freq').value
        timer_period = 1.0 / freq_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Head control
        self.head_pan_target = 0.0
        self.head_tilt_target = 0.0
        self.head_speed = 0.01
        self.pan_max = 1.57
        self.tilt_max = 0.5

        # Torso control
        self.torso_target = 0.3014
        self.torso_min = 0.0
        self.torso_max = 0.3014
        self.torso_step = 0.01

        # Base movement state
        self.speedX = 0.0
        self.speedY = 0.0
        self.yaw = 0.0
        self.stop = False
        self.skip_state = False

        # Button states
        self.prev_left_stick_button = False
        self.pad_vert = 0.0
        self.lb_pressed = False
        self.lb_was_pressed = False      # for detecting LB release
        self.rb_pressed = False

        self.paused = False

        # Zero‑publishing timer (after LB release)
        self.zero_until = self.get_clock().now()

        # Subscriber
        self.sub_joy = self.create_subscription(Joy, '/hardware/joy', self.joy_callback, 10)
        self.sub_paused = self.create_subscription(Bool, '/teleop/paused', self.paused_callback, 10)
        self.pub_twist = self.create_publisher(Twist, '/hardware/base/cmd_vel', 10)
        self.pan_pub = self.create_publisher(Float64, '/head/pan_cmd', 10)
        self.tilt_pub = self.create_publisher(Float64, '/head/tilt_cmd', 10)
        self.pub_head = self.create_publisher(Float64MultiArray, '/hardware/head/goal_pose', 10)
        self.pub_torso = self.create_publisher(Float64, '/torso/cmd_pos', 10)
        self.pub_stop = self.create_publisher(Empty, '/stop', 10)
        self.pub_skip = self.create_publisher(Empty, '/hardware/robot_state/skip_state', 10)

        self.get_logger().info(
            f"Joystick teleop started – LB enable (button {self.get_parameter('lb_button_index').value}), "
            f"RB multiplier {self.get_parameter('rb_speed_multiplier').value} (button {self.get_parameter('rb_button_index').value})"
        )

    def joy_callback(self, msg: Joy):
        # Buttons
        self.stop = (msg.buttons[1] == 1)
        self.skip_state = (msg.buttons[3] == 1)

        # Left stick button (index 9) for head centering
        left_stick_button = msg.buttons[9] if len(msg.buttons) > 9 else 0
        if left_stick_button == 1 and not self.prev_left_stick_button:
            self.head_pan_target = 0.0
            self.head_tilt_target = 0.0
        self.prev_left_stick_button = left_stick_button

        # LB – enable switch
        lb_idx = self.get_parameter('lb_button_index').value
        new_lb = (msg.buttons[lb_idx] == 1) if lb_idx < len(msg.buttons) else False

        # Detect LB release
        if self.lb_was_pressed and not new_lb:
            # Just released: start 1‑second zero‑publishing window
            self.zero_until = self.get_clock().now() + rclpy.duration.Duration(seconds=1.0)

        self.lb_pressed = new_lb
        self.lb_was_pressed = new_lb   # update for next callback

        # RB – speed multiplier
        rb_idx = self.get_parameter('rb_button_index').value
        new_rb = (msg.buttons[rb_idx] == 1) if rb_idx < len(msg.buttons) else False
        self.rb_pressed = new_rb

        # Axes
        left_x = msg.axes[0]
        left_y = msg.axes[1]
        right_x = msg.axes[3]
        right_y = msg.axes[4]
        left_trigger = msg.axes[2]
        right_trigger = msg.axes[5]
        self.pad_vert = msg.axes[7] if len(msg.axes) > 7 else 0.0

        # Triggers (lateral speed)
        trig_dead = self.get_parameter('trigger_deadzone').value
        if left_trigger < trig_dead:
            left_trigger = 0.0
        if right_trigger < trig_dead:
            right_trigger = 0.0

        magnitude_trigger = math.hypot(left_trigger, right_trigger)
        if magnitude_trigger > self.get_parameter('trigger_activation').value:
            lateral_scale = self.get_parameter('lateral_scale').value
            new_speedY = (right_trigger - left_trigger) * lateral_scale
        else:
            new_speedY = 0.0

        self.speedY = new_speedY

        # Right stick (base movement) with RB multiplier
        stick_dead = self.get_parameter('stick_deadzone').value
        magnitude_right = math.hypot(right_x, right_y)
        if magnitude_right > stick_dead:
            forward_scale = self.get_parameter('forward_scale').value
            rotate_scale = self.get_parameter('rotate_scale').value
            if self.rb_pressed:
                multiplier = self.get_parameter('rb_speed_multiplier').value
                forward_scale *= multiplier
                rotate_scale *= multiplier
            new_speedX = forward_scale * right_y
            new_yaw = rotate_scale * right_x
        else:
            new_speedX = 0.0
            new_yaw = 0.0

        self.speedX = new_speedX
        self.yaw = new_yaw

        # Left stick (head control)
        magnitude_left = math.hypot(left_x, left_y)
        if magnitude_left > stick_dead:
            norm_x = left_x / magnitude_left
            norm_y = left_y / magnitude_left
            increment = self.head_speed * magnitude_left
            self.head_pan_target += norm_x * increment
            self.head_tilt_target += norm_y * increment
            self.head_pan_target = max(min(self.head_pan_target, self.pan_max), -self.pan_max)
            self.head_tilt_target = max(min(self.head_tilt_target, self.tilt_max), -self.tilt_max)

    def paused_callback(self, msg):
        self.paused = msg.data
        if self.paused:
            t = Twist()
            self.pub_twist.publish(t)

    def timer_callback(self):
        now = self.get_clock().now()

        if self.paused:
            twist = Twist()
            self.pub_twist.publish(twist)
            return

        if self.control_always_enable or self.lb_pressed:
            # Base twist
            twist = Twist()
            twist.linear.x = self.speedX
            twist.linear.y = self.speedY
            twist.angular.z = self.yaw
            self.pub_twist.publish(twist)

            # Head commands
            pan_msg = Float64()
            pan_msg.data = self.head_pan_target
            tilt_msg = Float64()
            tilt_msg.data = self.head_tilt_target
            self.pan_pub.publish(pan_msg)
            self.tilt_pub.publish(tilt_msg)

            head_msg = Float64MultiArray()
            head_msg.data = [self.head_pan_target, self.head_tilt_target]
            self.pub_head.publish(head_msg)

            # Torso control
            if self.pad_vert > 0.5:
                self.torso_target += self.torso_step
            elif self.pad_vert < -0.5:
                self.torso_target -= self.torso_step
            self.torso_target = max(self.torso_min, min(self.torso_max, self.torso_target))
            torso_msg = Float64()
            torso_msg.data = self.torso_target
            self.pub_torso.publish(torso_msg)

            # Stop / skip commands
            if self.stop:
                self.pub_stop.publish(Empty())
            if self.skip_state:
                self.pub_skip.publish(Empty())

        elif now < self.zero_until:
            twist = Twist()
            self.pub_twist.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = JoystickTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Bool
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import torch
import os
import sys


# This is the "Nuclear Option" - it tells PyTorch not to use cuDNN
# but still use the GPU (CUDA). It fixes the CUDNN_STATUS error 100% of the time.
torch.backends.cudnn.enabled = False 

# Keep these for stability
torch.backends.cuda.matmul.allow_tf32 = True
# --- Fix for cuDNN CUDNN_STATUS_NOT_INITIALIZED ---
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Force an early context creation to verify the GPU handle
    torch.cuda.set_device(0)
    torch.empty(1).cuda()
# --------------------------------------------------

sys.path.append(os.path.dirname(__file__))
from TorchModels.utils import models

class RNNMapless(Node):
    def __init__(self):
        super().__init__('rnn_mapless')

        # ---------- Model parameters ----------
        self.n_channels = 2
        self.occ_grid_meters = 4
        self.img_size = int(20 * self.occ_grid_meters)   # 80 x 80 occupancy grid  
        self.hidden_size = 350
        self.num_layers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---------- ROS2 parameters ----------
        self.robot_name = self.declare_parameter('robot_name', 'justina').value
        self.turn_help = self.declare_parameter('turn_help', True).value

        # ---------- Load model ----------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'TorchModels', 'RNN.pth')
        self.get_logger().info(f'Loading model from {model_path}')

        self.model = models.RNN(
            channels=self.n_channels,
            img_size=self.img_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            output_size=3,
            activation='tanh'
        )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'Model loaded on {self.device}')

        # ---------- ROS2 interfaces ----------
        self.goal_sub = self.create_subscription(Float32MultiArray, '/NN_goal', self.goal_callback, 10)
        self.point_sub = self.create_subscription(PointStamped, '/clicked_point', self.point_callback, 10)
        self.occ_sub = self.create_subscription(OccupancyGrid, '/mapless/local_occ_grid', self.occ_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/hardware/base/cmd_vel', 10)
        self.target_reached_pub = self.create_publisher(Bool, '/maples_nav/goal_reached', 10)
        self.goal_active_sub = self.create_subscription(Bool, '/mapless_nav/goal_active', self.goal_active_callback, 10)

        # ---------- State variables (all Python floats) ----------
        self.last_goal = [0.0, 0.0]   # distance, theta
        self.target_reached = True
        self.linx = 0.0
        self.liny = 0.0
        self.angz = 0.0

        self.publish_target_reached(True)

        # ---------- Robot-specific setup ----------
        if self.robot_name == 'justina':
            # Use Float64MultiArray as expected by the real robot
            self.head_pub = self.create_publisher(Float64MultiArray, '/hardware/head/goal_pose', 10)
            head_msg = Float64MultiArray()
            head_msg.data = [0.0, -0.4]   # [pan, tilt]
            self.head_pub.publish(head_msg)
            self.get_logger().info("Justina initial head position sent (pan=0.0, tilt=-0.4)")
        elif self.robot_name == 'takeshi':
            self.get_logger().info("Takeshi: using default RNN controller")
        else:
            self.get_logger().info(f"No specific init for robot {self.robot_name}")

        self.timer = self.create_timer(0.2, self.control_loop)
        self.get_logger().info('RNN Controller ready')

    def goal_active_callback(self, msg):
        if msg.data:
            self.target_reached = False
            self.publish_target_reached(False)
            self.get_logger().info("Mapless goal activated")
        else:
            self.target_reached = True
            self.publish_target_reached(True)
            self.linx = 0.0
            self.liny = 0.0
            self.angz = 0.0
            self.get_logger().info("Mapless goal deactivated")

    def goal_callback(self, msg):
        if self.target_reached:
            self.last_goal = [0.0, 0.0]
        else:
            self.last_goal = list(msg.data)

    def point_callback(self, msg):
        self.get_logger().info(f'New goal: ({msg.point.x:.3f}, {msg.point.y:.3f})')
        self.target_reached = False
        self.publish_target_reached(False)

    def occ_callback(self, msg):
        rows = msg.info.height
        data = np.array(msg.data, dtype=np.float32).reshape(rows, rows)
        data = np.rot90(np.flip(data, axis=0))

        d, th = self.last_goal
        dist_row = np.ones(rows, dtype=np.float32) * round(d, 2)
        theta_row = np.ones(rows, dtype=np.float32) * round(th, 2)
        data_X = np.vstack((data, dist_row, theta_row))   # (82,80)

        batch_input = [data_X for _ in range(self.n_channels)]
        nn_input = np.expand_dims(np.array(batch_input), axis=0)   # (1,2,82,80)
        x_input = torch.tensor(nn_input, device=self.device, dtype=torch.float32)

        if abs(d) > 0.6:
            with torch.no_grad():
                output = self.model(x_input).cpu()
                # Ensure output is (batch,3)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                y_pred = output.numpy()[0]   # shape (3,)

            # Convert numpy scalars to Python float
            pred_linx = float(y_pred[0])
            pred_liny = float(y_pred[1])
            pred_angz = float(y_pred[2])

            if self.turn_help:
                if abs(th) < 0.5:
                    self.linx = pred_linx
                    self.liny = pred_liny
                    self.angz = pred_angz * 0.4
                else:
                    self.linx = pred_linx / (10.0 * abs(th))
                    self.liny = 0.8 * pred_liny / (10.0 * abs(th))
                    self.angz = pred_angz * 2.0
            else:
                self.linx, self.liny, self.angz = pred_linx, pred_liny, pred_angz
        else:
            if not self.target_reached:
                self.linx = 0.05
                self.liny = 0.0
                self.angz = 0.0
            else:
                self.linx = self.liny = self.angz = 0.0

            if d != 0.0 and th != 0.0:
                self.target_reached = True
                self.publish_target_reached(True)

        # clamp velocities to reasonable ranges (safety)
        self.linx = max(-0.5, min(0.5, self.linx))
        self.liny = max(-0.5, min(0.5, self.liny))
        self.angz = max(-1.0, min(1.0, self.angz))

    def control_loop(self):
        twist = Twist()
        twist.linear.x = self.linx
        twist.linear.y = self.liny
        twist.angular.z = self.angz
        self.cmd_pub.publish(twist)

        d, th = self.last_goal
        status = "> target reached" if self.target_reached else ""
        self.get_logger().debug(
            f"trg({d:.4f}, {th:.4f}) || vel({self.linx:.3f}, {self.liny:.3f}, {self.angz:.3f}) {status}",
            throttle_duration_sec=1.0
        )

    def publish_target_reached(self, reached):
        msg = Bool()
        msg.data = reached
        self.target_reached_pub.publish(msg)

    def set_initial_head_pose(self):
        """Send a head pose command to set pan=0.0, tilt=-0.5 rad (look slightly down)."""
        head_msg = Float64MultiArray()
        head_msg.data = [0.0, -0.5]   # [pan, tilt]
        self.head_pub.publish(head_msg)
        self.get_logger().info("Head pose updated: pan=0.0, tilt=-0.5")


def main(args=None):
    rclpy.init(args=args)
    node = RNNMapless()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import os
import sys
from collections import deque

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray

# Directivas para estabilidad CUDA/PyTorch
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    torch.empty(1).cuda()

sys.path.append(os.path.dirname(__file__))
from TorchModels.utils import models


def process_raw_velocities(y_pred, vmax, reverse_enable=False, denormalize=True):
    vx_raw, vy_raw, wz_raw = y_pred[0], y_pred[1], y_pred[2]

    if denormalize:
        if reverse_enable:
            vx = vx_raw * vmax[0]
        else:
            vx = max(0.0, vx_raw) * vmax[0]

        vy = vy_raw * vmax[1]
        wz = wz_raw * vmax[2]
    else:
        vx = vx_raw if reverse_enable else max(0.0, vx_raw)
        vy = vy_raw
        wz = wz_raw

    return float(vx), float(vy), float(wz)


class RNNMapless(Node):
    def __init__(self):
        super().__init__("rnn_mapless")

        # ---------- Parámetros ROS 2 ----------
        self.robot_name = self.declare_parameter("robot_name", "justina").value
        self.turn_help = self.declare_parameter("turn_help", False).value  # False por defecto para no atenuar ganancias
        self.checkpoint_name = self.declare_parameter("checkpoint", "best_model.pth").value

        # Parámetros de amplificación / ganancia
        self.enable_denormalization = self.declare_parameter("enable_denormalization", True).value
        self.gain_linx = self.declare_parameter("gain_linx", 1.0).value
        self.gain_liny = self.declare_parameter("gain_liny", 1.0).value
        self.gain_angz = self.declare_parameter("gain_angz", 1.0).value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------- Cargar Checkpoint y Configuración ----------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "TorchModels", "experiments", self.checkpoint_name)

        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, "TorchModels", self.checkpoint_name)

        self.get_logger().info(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config", {})
        self.sequence_length = config.get("sequence_length", 4)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 1)
        self.vmax = config.get("vmax", (0.5, 0.5, 1.0))
        self.reverse_enable = config.get("reverse_enable", False)
        self.distance_max = config.get("distance_max", None)

        self.get_logger().info(
            f"Settings -> Denormalize: {self.enable_denormalization} | "
            f"Gains (vx, vy, wz): ({self.gain_linx}, {self.gain_liny}, {self.gain_angz}) | "
            f"Turn Help: {self.turn_help}"
        )

        # ---------- Inicialización del Modelo ----------
        self.model = models.RNN(
            channels=self.sequence_length,
            img_size=80,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Búfer circular para la secuencia temporal
        self.frame_buffer = deque(maxlen=self.sequence_length)

        # ---------- Interfaces ROS 2 ----------
        self.goal_sub = self.create_subscription(
            Float32MultiArray, "/NN_goal", self.goal_callback, 10
        )
        self.point_sub = self.create_subscription(
            PointStamped, "/clicked_point", self.point_callback, 10
        )
        self.occ_sub = self.create_subscription(
            OccupancyGrid, "/mapless/local_occ_grid", self.occ_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/hardware/base/cmd_vel", 10)
        self.target_reached_pub = self.create_publisher(
            Bool, "/maples_nav/goal_reached", 10
        )
        self.goal_active_sub = self.create_subscription(
            Bool, "/mapless_nav/goal_active", self.goal_callback_active, 10
        )

        # Variables de estado
        self.last_goal = [0.0, 0.0]
        self.target_reached = True
        self.linx = 0.0
        self.liny = 0.0
        self.angz = 0.0

        self.publish_target_reached(True)

        if self.robot_name == "justina":
            self.head_pub = self.create_publisher(
                Float64MultiArray, "/hardware/head/goal_pose", 10
            )
            head_msg = Float64MultiArray()
            head_msg.data = [0.0, -0.4]
            self.head_pub.publish(head_msg)

        self.timer = self.create_timer(0.2, self.control_loop)
        self.get_logger().info("RNN Mapless Controller ready")

    def goal_callback_active(self, msg):
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
            self.frame_buffer.clear()
            self.get_logger().info("Mapless goal deactivated")

    def goal_callback(self, msg):
        if self.target_reached:
            self.last_goal = [0.0, 0.0]
        else:
            self.last_goal = list(msg.data)

    def point_callback(self, msg):
        self.get_logger().info(f"New goal received: ({msg.point.x:.3f}, {msg.point.y:.3f})")
        self.target_reached = False
        self.publish_target_reached(False)

    def occ_callback(self, msg):
        rows = msg.info.height
        data = np.array(msg.data, dtype=np.float32).reshape(rows, rows)
        data = np.rot90(np.flip(data, axis=0))
        data = np.clip(data / 100.0, 0.0, 1.0)

        d, th = self.last_goal

        norm_d = d
        if self.distance_max is not None and self.distance_max > 0:
            norm_d = np.clip(d / self.distance_max, 0.0, 1.0)

        dist_row = np.ones((1, rows), dtype=np.float32) * float(norm_d)
        theta_row = np.ones((1, rows), dtype=np.float32) * float(th)

        data_frame = np.vstack((data, dist_row, theta_row))

        self.frame_buffer.append(data_frame)
        while len(self.frame_buffer) < self.sequence_length:
            self.frame_buffer.appendleft(data_frame)

        sequence = np.stack(list(self.frame_buffer), axis=0)
        nn_input = np.expand_dims(sequence, axis=0)
        x_input = torch.tensor(nn_input, device=self.device, dtype=torch.float32)

        if abs(d) > 0.6:
            with torch.no_grad():
                output = self.model(x_input).cpu()
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() == 1:
                    output = output.unsqueeze(0)

                y_pred = output.numpy()[0]

            # Inferencia base
            pred_linx, pred_liny, pred_angz = process_raw_velocities(
                y_pred, self.vmax, self.reverse_enable, denormalize=self.enable_denormalization
            )

            # Aplicar potenciadores (ganancias pasadas por launch)
            pred_linx *= self.gain_linx
            pred_liny *= self.gain_liny
            pred_angz *= self.gain_angz

            # Asignación directa de velocidades
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

        # Límites de seguridad
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
            throttle_duration_sec=1.0,
        )

    def publish_target_reached(self, reached):
        msg = Bool()
        msg.data = reached
        self.target_reached_pub.publish(msg)


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


if __name__ == "__main__":
    main()
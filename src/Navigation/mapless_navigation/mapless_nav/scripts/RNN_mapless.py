#!/usr/bin/env python3
"""Single ROS 2 inference/controller for ROS 1 and ROS 2 mapless RNN checkpoints.

Both model layouts live in TorchModels/models.py. Checkpoints are identified by
metadata when available or by their tensor shapes for historical plain state_dicts.
For Webots, set use_sim_time=true in the launch file. Motion can be disabled with
`enable_motion:=false`; initial command limits are intentionally conservative.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import rclpy
    from geometry_msgs.msg import PointStamped, Twist
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray
except ModuleNotFoundError as exc:
    if (exc.name or "").split('.')[0] not in ("rclpy", "geometry_msgs", "nav_msgs", "std_msgs"):
        raise
    rclpy, Node = None, object


# LOM255_REEVALUATION_FIX_V1
def scale_occupancy_grid(occupancy, normalize_occupancy: bool, occupancy_max: float = 100.0):
    """Convert the incoming ROS 0..100 grid to the checkpoint's input scale."""
    maximum = float(occupancy_max)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"occupancy_max must be positive and finite; got {maximum}")
    values = np.asarray(occupancy, dtype=np.float32)
    if normalize_occupancy:
        return np.clip(values / np.float32(100.0), 0.0, 1.0)
    return values * np.float32(maximum / 100.0)


def prepare_observation(grid, distance: float, theta: float, settings: dict) -> np.ndarray:
    if settings.get("preprocessing") is not None:
        from experiment_control import policy_frame
        return policy_frame(grid, [distance, theta], source="ros_occupancy_grid")
    values = np.rot90(np.flip(np.asarray(grid, dtype=np.float32), axis=0))
    occ = scale_occupancy_grid(values, settings["normalize_occupancy"], settings["occupancy_max"])
    maximum = settings.get("distance_max")
    target_distance = float(np.clip(distance / float(maximum), 0, 1)) if maximum is not None and float(maximum) > 0 else distance
    rows = np.empty((2, values.shape[1]), dtype=np.float32)
    rows[0, :], rows[1, :] = target_distance, theta
    return np.vstack((occ, rows)).astype(np.float32, copy=False)


def decode_action(output, settings: dict) -> np.ndarray:
    if settings.get("preprocessing") is not None:
        from experiment_control import normalize_policy_actions
        return normalize_policy_actions(output, settings["preprocessing"], inverse=True)
    values = np.asarray(output, dtype=float)[:3].copy()
    values[:2] *= np.asarray(settings["vmax"][:2], dtype=float)
    if settings["wz_normalized"]:
        values[2] *= settings["vmax"][2]
    return values


def get_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError("Checkpoint is not a supported mapless RNN state_dict: conv1.weight is missing")
    return state


def detect_architecture(state: dict, checkpoint: Any) -> str:
    if isinstance(checkpoint, dict):
        claimed = checkpoint.get("architecture")
        if claimed in ("ros1_legacy", "ros2"):
            return claimed
    shape = tuple(state["conv1.weight"].shape)
    if len(shape) != 4:
        raise ValueError(f"Unexpected conv1.weight shape: {shape}")
    # Legacy Conv2d takes the stacked sequence as its input channels (C -> C),
    # and its first RNN layer historically expects 736 features for 80x80 input.
    # The native ROS 2 network applies a one-channel conv with four feature maps
    # and its first RNN layer expects 2464 features.
    first_rnn = state.get("rnn_cell_list.0.x2h.weight")
    recurrent_input_size = int(first_rnn.shape[1]) if first_rnn is not None else None
    if shape[1] > 1 or recurrent_input_size == 736:
        return "ros1_legacy"
    if shape[1] == 1 and (recurrent_input_size is None or recurrent_input_size != 736):
        return "ros2"
    raise ValueError(f"Cannot infer checkpoint architecture from conv1.weight shape {shape}")


def find_checkpoint(models_dir: Path, checkpoint_name: str) -> Path:
    requested = Path(checkpoint_name).expanduser()
    candidates = [requested] if requested.is_absolute() else [
        models_dir / requested,
        models_dir / "experiments" / requested,
        models_dir / "ROS1_train" / requested,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint {checkpoint_name!r} was not found. Tried:\n  - {attempted}")


def get_nested_config(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return {}


def count_legacy_layers(state: dict) -> int:
    layers = set()
    for key in state:
        if key.startswith("rnn_cell_list.") and ".x2h.weight" in key:
            try:
                layers.add(int(key.split(".")[1]))
            except (ValueError, IndexError):
                pass
    return len(layers) or 4


def make_model(architecture: str, state: dict, config: dict, sequence_length_override: int = 0):
    scripts_dir = Path(__file__).resolve().parent
    models_dir = scripts_dir / "TorchModels"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from models import RNN

    first_rnn = state.get("rnn_cell_list.0.x2h.weight")
    inferred_hidden = int(first_rnn.shape[0]) if first_rnn is not None else 350
    if architecture == "ros1_legacy":
        inferred_channels = int(state["conv1.weight"].shape[0])
        default_channels = inferred_channels
    elif architecture == "ros2":
        default_channels = sequence_length_override or 4
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if architecture == "ros1_legacy":
        configured_channels = (
            config.get("stack_n_channels")
            or config.get("sequence_length")
            or config.get("channels")
            or default_channels
        )
    else:
        configured_channels = config.get("sequence_length") or config.get("channels") or default_channels
    channels = int(configured_channels)
    if sequence_length_override > 0 and not config:
        if architecture == "ros1_legacy" and sequence_length_override != default_channels:
            raise ValueError(
                "For a plain ROS 1 legacy checkpoint, sequence length is encoded in conv1 weights; "
                f"the checkpoint requires C={default_channels}, not C={sequence_length_override}."
            )
        channels = int(sequence_length_override)
    hidden_size = int(config.get("hidden_size", inferred_hidden))
    num_layers = int(config.get("num_layers", count_legacy_layers(state)))
    dropout_1 = float(config.get("dropout_1", 0.20))
    dropout_2 = float(config.get("dropout_2", 0.15))

    model = RNN(
        channels=channels,
        img_size=80,
        hidden_size=hidden_size,
        num_layers=num_layers,
        architecture=architecture,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
    )

    data_format = config.get("data_format")
    if "normalize_occupancy" in config:
        normalize_occupancy = bool(config["normalize_occupancy"])
    elif "normalize_data" in config:
        normalize_occupancy = bool(config["normalize_data"])
    else:
        # Old plain ROS 1 state_dicts used raw 0..100 occupancy values;
        # ROS 2 model checkpoints normally used occupancy normalized to [0,1].
        normalize_occupancy = architecture == "ros2" or data_format == "ros2_episode"

    label_normalization = config.get("label_normalization", "legacy" if architecture == "ros1_legacy" else "vmax")
    wz_normalized = not (
        data_format == "ros1_legacy" and label_normalization == "legacy"
    )
    if not config and architecture == "ros1_legacy":
        # Historical checkpoint has no saved preprocessing metadata; wz was raw.
        wz_normalized = False

    settings = {
        "sequence_length": channels,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "normalize_occupancy": normalize_occupancy,
        "occupancy_max": float(config.get("occupancy_max", 100.0)),
        "distance_max": config.get("distance_max", None),
        "vmax": config.get("vmax", (0.5, 0.5, 1.0)),
        "reverse_enable": bool(config.get("reverse_enable", False)),
        "wz_normalized": wz_normalized,
        "data_format": data_format or ("ros1_legacy" if architecture == "ros1_legacy" else "ros2_episode"),
    }
    preprocessing = config.get("preprocessing")
    if int(config.get("format_version", 0)) >= 3 or preprocessing is not None:
        from experiment_control import validate_policy_spec
        if not isinstance(preprocessing, dict):
            raise ValueError("Checkpoint preprocessing metadata is missing")
        validate_policy_spec(preprocessing)
        if ((architecture, channels, hidden_size, num_layers) != ("ros1_legacy", 2, 350, 4)
                or normalize_occupancy or config.get("distance_max") is not None
                or not config.get("normalize_label", False)
                or config.get("label_normalization") != "legacy_training_pool_minmax"
                or float(config.get("occupancy_max", 100.0)) != 100.0):
            raise ValueError("Checkpoint settings disagree with the reference preprocessing specification")
        settings["preprocessing"] = preprocessing
        settings["wz_normalized"] = False
    model.load_state_dict(state, strict=True)
    return model, settings


class RNNMapless(Node):
    def __init__(self):
        super().__init__("rnn_mapless")

        # checkpoint is canonical; model_filename is retained for old launch files.
        self.declare_parameter("checkpoint", "RNN.pth")
        self.declare_parameter("model_filename", "")
        self.declare_parameter("enable_motion", True)
        self.declare_parameter("robot_name", "justina")
        self.declare_parameter("turn_help", True)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("sequence_length", 0)
        self.declare_parameter("goal_topic", "/NN_goal")
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("local_grid_topic", "/mapless/local_occ_grid")
        self.declare_parameter("cmd_vel_topic", "/hardware/base/cmd_vel")
        self.declare_parameter("max_vx", 0.08)
        self.declare_parameter("max_vy", 0.05)
        self.declare_parameter("max_wz", 0.20)
        self.declare_parameter("vmax_x", 0.5)
        self.declare_parameter("vmax_y", 0.5)
        self.declare_parameter("vmax_w", 1.0)
        self.declare_parameter("gain_linx", 1.0)
        self.declare_parameter("gain_liny", 1.0)
        self.declare_parameter("gain_angz", 1.0)
        self.declare_parameter("log_every_n_maps", 10)
        self.declare_parameter("stale_timeout_sec", 1.0)

        requested = str(self.get_parameter("model_filename").value).strip()
        if not requested:
            requested = str(self.get_parameter("checkpoint").value).strip()
        scripts_dir = Path(__file__).resolve().parent
        models_dir = scripts_dir / "TorchModels"
        checkpoint_path = find_checkpoint(models_dir, requested)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # compatibility with older torch versions
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = get_state_dict(checkpoint)
        self.architecture = detect_architecture(state, checkpoint)
        config = get_nested_config(checkpoint)
        self.model, self.model_settings = make_model(
            self.architecture,
            state,
            config,
            sequence_length_override=int(self.get_parameter("sequence_length").value),
        )
        self.model.to(self.device)
        self.model.eval()

        self.sequence_length = int(self.model_settings["sequence_length"])
        vmax_config = self.model_settings.get("vmax", (0.5, 0.5, 1.0))
        if not isinstance(vmax_config, (tuple, list)) or len(vmax_config) != 3:
            vmax_config = (
                self.get_parameter("vmax_x").value,
                self.get_parameter("vmax_y").value,
                self.get_parameter("vmax_w").value,
            )
        self.vmax = tuple(float(value) for value in vmax_config)
        self.distance_max = self.model_settings["distance_max"]
        self.normalize_occupancy = bool(self.model_settings["normalize_occupancy"])
        self.occupancy_max = float(self.model_settings["occupancy_max"])
        if not math.isfinite(self.occupancy_max) or self.occupancy_max <= 0.0:
            raise ValueError("Checkpoint occupancy_max must be positive and finite")
        self.turn_help = bool(self.get_parameter("turn_help").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.max_command = (
            float(self.get_parameter("max_vx").value),
            float(self.get_parameter("max_vy").value),
            float(self.get_parameter("max_wz").value),
        )
        self.gains = (
            float(self.get_parameter("gain_linx").value),
            float(self.get_parameter("gain_liny").value),
            float(self.get_parameter("gain_angz").value),
        )
        self.enable_motion = bool(self.get_parameter("enable_motion").value)
        self.stale_timeout = max(0.2, float(self.get_parameter("stale_timeout_sec").value))
        self.log_every = max(1, int(self.get_parameter("log_every_n_maps").value))
        if self.goal_tolerance <= 0 or min(*self.vmax, *self.max_command) <= 0:
            raise ValueError("goal tolerance and velocity limits must be positive")

        self.frames = deque(maxlen=self.sequence_length)
        self.last_goal = (0.0, 0.0)
        self.last_goal_walltime: float | None = None
        self.last_grid_walltime: float | None = None
        self.goal_active = False
        self.command = (0.0, 0.0, 0.0)
        self.map_count = 0

        goal_topic = str(self.get_parameter("goal_topic").value)
        point_topic = str(self.get_parameter("clicked_point_topic").value)
        grid_topic = str(self.get_parameter("local_grid_topic").value)
        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.goal_sub = self.create_subscription(Float32MultiArray, goal_topic, self.goal_callback, 10)
        self.point_sub = self.create_subscription(PointStamped, point_topic, self.point_callback, 10)
        self.occ_sub = self.create_subscription(OccupancyGrid, grid_topic, self.occ_callback, 10)
        self.goal_active_sub = self.create_subscription(Bool, "/mapless_nav/goal_active", self.goal_active_callback, 10)
        self.reached_pub = self.create_publisher(Bool, "/maples_nav/goal_reached", 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10) if self.enable_motion else None
        self.timer = self.create_timer(0.1, self.publish_command) if self.enable_motion else None

        if str(self.get_parameter("robot_name").value) == "justina":
            self.head_pub = self.create_publisher(Float64MultiArray, "/hardware/head/goal_pose", 10)
            head = Float64MultiArray()
            head.data = [0.0, -0.4]
            self.head_pub.publish(head)

        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.get_logger().info(
            f"Unified RNN controller ready | architecture={self.architecture} | checkpoint={checkpoint_path.name} | "
            f"sequence={self.sequence_length} | hidden={self.model_settings['hidden_size']} | "
            f"layers={self.model_settings['num_layers']} | device={self.device} | use_sim_time={use_sim_time}"
        )
        self.get_logger().info(
            f"data_format={self.model_settings['data_format']} | LOM normalized={self.normalize_occupancy}; "
            f"occupancy_max={self.occupancy_max:g}; incoming_grid_max=100; "
            f"motion={'ENABLED' if self.enable_motion else 'DISABLED'}; "
            f"command caps vx={self.max_command[0]:.3f}, vy={self.max_command[1]:.3f}, wz={self.max_command[2]:.3f}"
        )
        if not use_sim_time:
            self.get_logger().warning("use_sim_time=false. Set true for Webots launches.")

    def point_callback(self, msg: PointStamped):
        self.goal_active = True
        # Don't let commands from a previous goal leak into the new goal while TF updates.
        self.last_goal = (0.0, 0.0)
        self.last_goal_walltime = None
        self.command = (0.0, 0.0, 0.0)
        self.frames.clear()
        self.get_logger().info(f"Clicked goal received in frame {msg.header.frame_id!r}; waiting for /NN_goal transform")
        self.publish_reached(False)

    def goal_active_callback(self, msg: Bool):
        self.goal_active = bool(msg.data)
        if not self.goal_active:
            self.last_goal = (0.0, 0.0)
            self.stop_command()
            self.frames.clear()
        else:
            self.last_goal_walltime = time.monotonic()
        self.publish_reached(not self.goal_active)

    def goal_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warning("Ignoring /NN_goal with fewer than two values")
            return
        distance, theta = float(msg.data[0]), float(msg.data[1])
        if not (math.isfinite(distance) and math.isfinite(theta)):
            self.get_logger().warning("Ignoring /NN_goal containing non-finite values")
            return
        if self.goal_active:
            self.last_goal = (max(0.0, distance), theta)
            self.last_goal_walltime = time.monotonic()

    def stop_command(self):
        self.command = (0.0, 0.0, 0.0)

    def occ_callback(self, msg: OccupancyGrid):
        height, width = int(msg.info.height), int(msg.info.width)
        if (height, width) != (80, 80):
            self.stop_command()
            self.get_logger().warning(f"Expected 80x80 local grid; received {height}x{width}; stopping")
            return
        self.last_grid_walltime = time.monotonic()
        if not self.goal_active:
            self.stop_command()
            return

        distance, theta = self.last_goal
        if self.last_goal_walltime is None or time.monotonic() - self.last_goal_walltime > self.stale_timeout:
            self.stop_command()
            self.frames.clear()
            return
        if distance <= self.goal_tolerance:
            self.stop_command()
            self.goal_active = False
            self.frames.clear()
            self.publish_reached(True)
            return

        occ_source = np.asarray(msg.data, dtype=np.float32).reshape(height, width)
        occ_source_max = float(np.max(occ_source))
        frame = prepare_observation(occ_source, distance, theta, self.model_settings)
        occ = frame[:-2]
        self.frames.append(frame)
        if self.model_settings.get("preprocessing") is not None and len(self.frames) < self.sequence_length:
            self.stop_command()
            return
        while len(self.frames) < self.sequence_length:
            self.frames.appendleft(frame)
        sequence = np.stack(tuple(self.frames), axis=0)[None, ...]
        x = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)

        try:
            with torch.inference_mode():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                y = output.detach().cpu().reshape(-1).numpy()
            if y.size < 3 or not np.all(np.isfinite(y[:3])):
                raise ValueError(f"Invalid output from RNN: {y}")

            vx, vy, wz = (float(value) for value in decode_action(y, self.model_settings))
            if not self.model_settings["reverse_enable"]:
                vx = max(0.0, vx)

            if self.turn_help:
                if abs(theta) < 0.5:
                    wz *= 0.4
                else:
                    scale = 10.0 * abs(theta)
                    vx /= scale
                    vy = 0.8 * vy / scale
                    wz *= 2.0

            vx *= self.gains[0]
            vy *= self.gains[1]
            wz *= self.gains[2]
            vx = float(np.clip(vx, -self.max_command[0], self.max_command[0]))
            vy = float(np.clip(vy, -self.max_command[1], self.max_command[1]))
            wz = float(np.clip(wz, -self.max_command[2], self.max_command[2]))
            self.command = (vx, vy, wz)
            self.map_count += 1
            if self.map_count == 1 or self.map_count % self.log_every == 0:
                self.get_logger().info(
                    f"arch={self.architecture} goal d={distance:.3f} theta={theta:.3f} | "
                    f"raw=({y[0]:+.4f},{y[1]:+.4f},{y[2]:+.4f}) | cmd=({vx:+.3f},{vy:+.3f},{wz:+.3f})"
                    f" | lom_src_max={occ_source_max:.3f} lom_net_max={float(np.max(occ)):.3f}"
                    f" occupancy_max={self.occupancy_max:g}"
                )
        except Exception as exc:
            self.stop_command()
            self.get_logger().error(f"RNN inference failed; command stopped: {exc}")

    def publish_command(self):
        if self.cmd_pub is None:
            return
        now = time.monotonic()
        stale_grid = self.last_grid_walltime is None or now - self.last_grid_walltime > self.stale_timeout
        stale_goal = self.last_goal_walltime is None or now - self.last_goal_walltime > self.stale_timeout
        cmd = (0.0, 0.0, 0.0) if (stale_grid or stale_goal or not self.goal_active) else self.command
        if stale_grid or stale_goal:
            self.command = (0.0, 0.0, 0.0)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = cmd
        self.cmd_pub.publish(msg)

    def publish_reached(self, reached: bool):
        msg = Bool()
        msg.data = bool(reached)
        self.reached_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.cmd_pub is not None and rclpy.ok():
                self.cmd_pub.publish(Twist())
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    if rclpy is None:
        raise RuntimeError("ROS 2 is unavailable. Source the ROS 2 workspace before starting navigation.")
    rclpy.init(args=args)
    node = None
    try:
        node = RNNMapless()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Some ROS 2 versions can raise this while SIGINT is tearing down the context.
        # Only suppress it if shutdown has already begun; preserve live runtime errors.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

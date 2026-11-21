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

from m1b_pair_live import ros_depth_to_network, stamp_to_ns, best_past_depth, pack_m1b_input

try:
    import rclpy
    from geometry_msgs.msg import PointStamped, Twist
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray
except ModuleNotFoundError as exc:
    if (exc.name or "").split('.')[0] not in ("rclpy", "geometry_msgs", "nav_msgs", "std_msgs", "sensor_msgs"):
        raise
    rclpy, Node = None, object


EXPERIMENT_TOOLS_DIR = Path(__file__).resolve().parent / "experiment_tools"

if str(EXPERIMENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_TOOLS_DIR))


# Slices and shapes for the three LOM variants. Same values as dataset_utils.
VARIANT_SLICES = {
    "4x4": (slice(40, 120), slice(20, 100)),
    "5x4": (slice(40, 140), slice(20, 100)),
    "7x6": (slice(0, 140), slice(0, 120)),
}

VARIANT_SHAPES = {
    "4x4": (80, 80),
    "5x4": (100, 80),
    "7x6": (140, 120),
}

VARIANT_ORIGIN = {
    "4x4": (0.0, -2.0),
    "5x4": (-1.0, -2.0),
    "7x6": (-1.0, -3.0),
}


# LOM255_REEVALUATION_FIX_V1
def scale_occupancy_grid(
    occupancy,
    normalize_occupancy: bool,
    occupancy_max: float = 100.0,
    representation: str = "lom",
):
    """Convert the incoming ROS 0..100 grid to the checkpoint's input scale."""
    maximum = float(occupancy_max)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"occupancy_max must be positive and finite; got {maximum}")
    values = np.asarray(occupancy, dtype=np.float32)
    if representation == "tlom":
        # The recorded T-LOM was binarized to 0/100 before saving. The runtime
        # topic publishes 0/50/100, so collapse anything >= 50 to 100 to match
        # the scale the model saw during training.
        values = np.where(values >= 50.0, 100.0, 0.0).astype(np.float32)
    if normalize_occupancy:
        return np.clip(values / np.float32(100.0), 0.0, 1.0)
    return values * np.float32(maximum / 100.0)


def source_grid_variant(shape, settings: dict):
    variant = settings.get("variant", "4x4")
    if shape == VARIANT_SHAPES[variant][::-1]:
        return variant
    if (shape == VARIANT_SHAPES["7x6"][::-1]
            and settings.get("data_format") == "ros2_episode"
            and settings.get("preprocessing") is None):
        return "7x6"
    return None


def prepare_observation(grid, distance: float, theta: float, settings: dict) -> np.ndarray:
    if settings.get("preprocessing") is not None:
        from experiment_control import policy_frame
        return policy_frame(grid, [distance, theta], source="ros_occupancy_grid")
    variant = settings.get("variant", "4x4")
    if variant not in VARIANT_SHAPES:
        raise ValueError(f"Unknown variant {variant!r}")
    _, expected_w = VARIANT_SHAPES[variant]
    source = np.asarray(grid, dtype=np.float32)
    grid_variant = source_grid_variant(source.shape, settings)
    if grid_variant is None or not np.isfinite(source).all():
        raise ValueError(
            f"Unexpected ROS grid {source.shape} for variant {variant}"
        )
    rotated = np.fliplr(np.rot90(source))
    if grid_variant == "7x6":
        rows, cols = VARIANT_SLICES[variant]
        rotated = rotated[rows, cols]
    occ = scale_occupancy_grid(
        rotated,
        settings["normalize_occupancy"],
        settings["occupancy_max"],
        representation=settings.get("representation", "lom"),
    )
    maximum = settings.get("distance_max")
    target_distance = float(np.clip(distance / float(maximum), 0, 1)) if maximum is not None and float(maximum) > 0 else distance
    rows = np.empty((2, expected_w), dtype=np.float32)
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
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict is invalid")
    if "conv1.weight" not in state and "encoder.0.weight" not in state:
        raise ValueError("Checkpoint is not a known mapless model")
    return state


def detect_architecture(state: dict, checkpoint: Any) -> str:
    # Check the dedicated family before the generic legacy conv-weight heuristic.
    config = get_nested_config(checkpoint)
    if config.get('model_family') == 'spatiotemporal_m1b_depth_pair_v1':
        architecture = config.get('architecture')
        if architecture not in ('m1b_mega', 'm1b_depth_direct_v2'):
            raise ValueError('Unsupported M1_B pair checkpoint architecture')
        if config.get('model_type') != ('M1_B' if architecture == 'm1b_mega' else 'M1_B_Depth'):
            raise ValueError('M1_B pair checkpoint model_type mismatch')
        return architecture
    if isinstance(checkpoint, dict):
        claimed = checkpoint.get("architecture")
        if claimed in ("ros1_legacy", "ros2"):
            return claimed
    if isinstance(checkpoint, dict):
        config = get_nested_config(checkpoint)
        if (config.get("model_family") == "spatiotemporal_m1refine_v1"
                and config.get("model_type") in ("M1_control", "M1_B", "M1retro_vel")
                and "conv1.weight" in state):
            return "m1refine_" + config["model_type"].lower()
    if isinstance(checkpoint, dict):
        config = get_nested_config(checkpoint)
        if (config.get("model_family") == "spatiotemporal_m123_v1"
                and config.get("model_type") in ("M1", "M2", "M3")
                and "conv1.weight" in state):
            return "m123_" + config["model_type"].lower()
    if isinstance(checkpoint, dict):
        config = get_nested_config(checkpoint)
        if (config.get("model_family") == "spatiotemporal_v1"
                and config.get("model_type") == "convlstm"
                and "encoder.0.weight" in state):
            return "temporal_convlstm"
    if isinstance(checkpoint, dict):
        config = get_nested_config(checkpoint)
        model_type = config.get("model_type")
        if (config.get("model_family") == "temporal_v1"
                and model_type in ("rnn", "gru", "lstm")
                and "encoder.0.weight" in state):
            return "temporal_" + model_type
    if "conv1.weight" not in state:
        raise ValueError("Temporal checkpoints require explicit model_family metadata")
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
    workspace_root = models_dir.parents[5]

    candidates = [requested] if requested.is_absolute() else [
        models_dir / requested,
        workspace_root / "experiments" / requested,
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
    if architecture in ('m1b_mega', 'm1b_depth_direct_v2'):
        if (config.get('model_family') != 'spatiotemporal_m1b_depth_pair_v1'
                or config.get('architecture') != architecture
                or config.get('model_variant') != ('baseline' if architecture == 'm1b_mega' else 'depth')
                or config.get('variant') != '5x4'
                or config.get('goal_fusion') != 'mixed_rows_not_in_conv1'
                or not config.get('normalize_occupancy')):
            raise ValueError('Invalid M1_B pair checkpoint configuration')
        history = int(config['history_pairs'])
        hidden = int(config['spatial_hidden_channels'])
        if history != 2 or hidden != 16:
            raise ValueError('This live adapter only supports trained history_pairs=2, hidden=16')
        kwargs = {
            'history_pairs': history,
            'hidden_channels': hidden,
            'img_h': 100,
            'img_w': 80,
            'conv_dropouts': config.get('conv_dropouts', (.20,.15)),
            'head_dropouts': config.get('head_dropouts', (.20,.15)),
        }
        if architecture == 'm1b_mega':
            from m1b_mega_models import M1BMegaPolicy
            model = M1BMegaPolicy(**kwargs)
        else:
            from m1b_depth_verified import M1BMegaDepthPolicy
            if config.get('depth_encoding') != '16UC1_PNG_normalized_near_high':
                raise ValueError('Unexpected Depth encoding in checkpoint')
            if float(config.get('depth_range_max_m', 0)) != 3.5:
                raise ValueError('Depth range does not match training')
            model = M1BMegaDepthPolicy(**kwargs)
        model.load_state_dict(state, strict=True)
        settings = {
            'sequence_length': int(config['sequence_length']),
            'hidden_size': hidden, 'num_layers': 1,
            'normalize_occupancy': True, 'occupancy_max': 100.0,
            'distance_max': None,
            'vmax': (0.5, 0.5, 1.0),
            'reverse_enable': False,
            'wz_normalized': True,
            'data_format': 'ros2_episode',
            'variant': '5x4', 'representation': 'tlom',
            'require_full_sequence': False,
            'temporal_mode': 'stateful_spatial',
            'm123_dual_source': True,
            'm1b_pair_variant': 'baseline' if architecture == 'm1b_mega' else 'depth',
            'm1b_requires_depth': architecture == 'm1b_depth_direct_v2',
            'depth_range_max_m': 3.5,
        }
        return model, settings

    from models import RNN, TemporalPolicy, GoalConditionedConvLSTM

    if architecture.startswith("m1refine_"):
        from m1_refine_models import M1RefinePolicy,SPECS
        variant_name=config.get('model_type')
        if (config.get('model_family')!='spatiotemporal_m1refine_v1'
                or variant_name not in SPECS
                or architecture!='m1refine_'+variant_name.lower()
                or config.get('architecture')!=architecture
                or config.get('temporal_mode')!='stateful'
                or int(config.get('input_channels',0))!=SPECS[variant_name]['history']*2
                or config.get('velocity_feedback')!=('previous_expert_cmd_and_previous_nn'
                    if SPECS[variant_name]['feedback'] else 'none')
                or config.get('goal_fusion')!='mixed_rows'
                or tuple(config.get('layer_bptt',()))!=(16,)
                or config.get('variant')!='5x4'
                or int(config.get('img_h',0))!=100 or int(config.get('img_w',0))!=80
                or not config.get('episode_plan_sha256') or not config.get('source_hashes')
                or not config.get('normalize_occupancy',False)):
            raise ValueError('Invalid M1 refinement checkpoint metadata')
        model=M1RefinePolicy(variant_name,100,80,
               conv_dropouts=config.get('conv_dropouts',(0.20,0.15)),
               head_dropouts=config.get('head_dropouts',(0.20,0.15)))
        model.load_state_dict(state,strict=True)
        settings={'sequence_length':16,'hidden_size':16,'num_layers':1,
                  'normalize_occupancy':True,'occupancy_max':100.0,
                  'distance_max':config.get('distance_max'),
                  'vmax':config.get('vmax',(0.5,0.5,1.0)),
                  'reverse_enable':bool(config.get('reverse_enable',False)),
                  'wz_normalized':True,'data_format':'ros2_episode',
                  'variant':'5x4','representation':'tlom',
                  'require_full_sequence':False,'temporal_mode':'stateful_spatial',
                  'm1_refine_variant':variant_name,'m123_dual_source':True}
        return model,settings

    if architecture.startswith("m123_"):
        from m123_models import M123Policy
        variant_name=architecture.removeprefix("m123_").upper()
        if (variant_name not in M123Policy.SPECS
                or config.get("model_family") != "spatiotemporal_m123_v1"
                or config.get("model_type") != variant_name
                or config.get("architecture") != architecture
                or config.get("temporal_mode") != "stateful"
                or int(config.get("input_channels",0)) != 2
                or config.get("goal_fusion") != ("mixed_rows" if variant_name=="M1" else "mixed_columns")
                or tuple(config.get("layer_bptt",())) != M123Policy.SPECS[variant_name]["bptt"]
                or int(config.get("spatial_hidden_channels",0)) != 16):
            raise ValueError("Invalid M123 architecture metadata")
        grid_variant=config.get("variant")
        if (grid_variant not in VARIANT_SHAPES
                or tuple(config.get("img_h",()) if isinstance(config.get("img_h"),tuple) else
                         (config.get("img_h"),config.get("img_w"))) != VARIANT_SHAPES[grid_variant]):
            raise ValueError("M123 grid variant metadata mismatch")
        if not bool(config.get("normalize_occupancy")):
            raise ValueError("M123 expects normalized occupancy values")
        model=M123Policy(variant_name, int(config["img_h"]), int(config["img_w"]), hidden_channels=16)
        model.load_state_dict(state, strict=True)
        settings={"sequence_length":16,"hidden_size":16,
                  "num_layers":M123Policy.SPECS[variant_name]['layers'],
                  "normalize_occupancy":True,"occupancy_max":100.0,
                  "distance_max":config.get('distance_max'),
                  "vmax":config.get('vmax',(0.5,0.5,1.0)),
                  "reverse_enable":bool(config.get('reverse_enable',False)),
                  "wz_normalized":True,"data_format":"ros2_episode",
                  "variant":grid_variant,"representation":"tlom",
                  "require_full_sequence":False,
                  "temporal_mode":"stateful_spatial",
                  "m123_variant":variant_name,
                  "m123_dual_source":variant_name=='M1'}
        return model,settings

    if architecture == "temporal_convlstm":
        if (config.get("model_family") != "spatiotemporal_v1"
                or config.get("model_type") != "convlstm"
                or config.get("temporal_mode") != "stateful"
                or config.get("target_fusion") not in ("spatial_before_convlstm", "spatial_mixed_half_map")):
            raise ValueError("Invalid goal-conditioned ConvLSTM checkpoint metadata")
        goal_fusion = config.get("goal_fusion", "dual")
        expected_fusion = ("spatial_mixed_half_map" if goal_fusion == "mixed"
                           else "spatial_before_convlstm")
        if goal_fusion not in ("dual", "mixed") or config["target_fusion"] != expected_fusion:
            raise ValueError("ConvLSTM goal fusion metadata mismatch")
        variant = config.get("variant", "5x4")
        if (variant not in VARIANT_SHAPES
                or (int(config["img_h"]), int(config["img_w"])) != VARIANT_SHAPES[variant]):
            raise ValueError("ConvLSTM grid variant does not match training dimensions")
        if config.get("representation") not in ("lom", "tlom"):
            raise ValueError("Invalid ConvLSTM representation")
        if not bool(config.get("normalize_occupancy", False)):
            raise ValueError("ConvLSTM requires the recorded normalized binary occupancy")
        if config.get("preprocessing") is not None:
            raise ValueError("ConvLSTM uses direct ROS 2 frame preprocessing, not frozen legacy pairs")
        model = GoalConditionedConvLSTM(
            img_h=int(config["img_h"]), img_w=int(config["img_w"]),
            spatial_hidden_channels=int(config["spatial_hidden_channels"]),
            encoder_channels=tuple(config.get("encoder_channels") or (8, 16)),
            goal_fusion=goal_fusion,
            num_layers=int(config["num_layers"]),
            dropout_1=float(config.get("dropout_1", 0.20)),
            dropout_2=float(config.get("dropout_2", 0.15)),
        )
        model.load_state_dict(state, strict=True)
        settings = {
            "sequence_length": int(config["sequence_length"]),
            "hidden_size": int(config["spatial_hidden_channels"]),
            "num_layers": int(config["num_layers"]),
            "normalize_occupancy": True,
            "occupancy_max": 100.0,
            "distance_max": config.get("distance_max"),
            "vmax": config.get("vmax", (0.5, 0.5, 1.0)),
            "reverse_enable": bool(config.get("reverse_enable", False)),
            "wz_normalized": True,
            "data_format": "ros2_episode",
            "variant": variant,
            "representation": config["representation"],
            "require_full_sequence": False,
            "temporal_mode": "stateful_spatial",
        }
        return model, settings

    if architecture.startswith("temporal_"):
        model_type = architecture.removeprefix("temporal_")
        if config.get("model_family") != "temporal_v1" or config.get("model_type") != model_type:
            raise ValueError("Temporal checkpoint model metadata mismatch")
        model = TemporalPolicy(
            model_type=model_type,
            img_h=int(config["img_h"]), img_w=int(config["img_w"]),
            hidden_size=int(config["hidden_size"]), num_layers=int(config["num_layers"]),
            dropout_1=float(config.get("dropout_1", 0.20)),
            dropout_2=float(config.get("dropout_2", 0.15)),
        )
        model.load_state_dict(state, strict=True)
        if config.get("temporal_mode") != "independent":
            raise ValueError("This patch supports independent temporal checkpoints only")
        if config.get("representation") not in ("lom", "tlom"):
            raise ValueError("Invalid temporal observation representation")
        variant = config.get("variant", "5x4")
        if variant not in VARIANT_SHAPES:
            raise ValueError("Unknown grid variant")
        if (int(config["img_h"]), int(config["img_w"])) != VARIANT_SHAPES[variant]:
            raise ValueError("Checkpoint grid dimensions disagree with variant")
        settings = {
            "sequence_length": int(config["sequence_length"]),
            "hidden_size": int(config["hidden_size"]),
            "num_layers": int(config["num_layers"]),
            "normalize_occupancy": bool(config.get("normalize_occupancy", True)),
            "occupancy_max": float(config.get("occupancy_max", 100.0)),
            "distance_max": config.get("distance_max"),
            "vmax": config.get("vmax", (0.5, 0.5, 1.0)),
            "reverse_enable": bool(config.get("reverse_enable", False)),
            "wz_normalized": True,
            "data_format": "ros2_episode",
            "variant": variant,
            "representation": config["representation"],
            "require_full_sequence": True,
            "temporal_mode": "independent",
        }
        return model, settings

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

    img_h = int(config.get("img_h", 80))
    img_w = int(config.get("img_w", 80))
    model = RNN(
        channels=channels,
        img_h=img_h,
        img_w=img_w,
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

    variant = config.get("variant", "4x4")
    representation = config.get("representation", "lom")
    if variant not in VARIANT_SLICES:
        raise ValueError(f"Unknown variant in checkpoint: {variant!r}")
    if representation not in ("lom", "tlom"):
        raise ValueError(f"Unknown representation in checkpoint: {representation!r}")
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
        "variant": variant,
        "representation": representation,
        "require_full_sequence": bool(config.get("require_full_sequence", False)),
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
        self.declare_parameter("depth_topic", "/hardware/camera/depth/image_raw")
        self.declare_parameter("depth_max_age_sec", 0.25)

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
        self.variant = self.model_settings.get("variant", "4x4")
        self.representation = self.model_settings.get("representation", "lom")
        if self.variant not in VARIANT_SHAPES:
            raise ValueError(f"Unknown variant in checkpoint: {self.variant!r}")
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
        self.hx = None
        self._m123_previous = None
        self._m123_lom_cache = {}
        self._m123_tlom_cache = {}
        self._m1b_pair = bool(self.model_settings.get('m1b_pair_variant'))
        self._m1b_depth_required = bool(self.model_settings.get('m1b_requires_depth'))
        self._m1b_depth_cache = []  # (sim timestamp ns, normalized depth, wall time)
        self._m1b_pending_pairs = {}  # (lom, tlom, insertion wall time)
        self._m1b_previous_depth = None
        self._m1b_last_step_stamp_ns = None
        self._m1b_last_depth_walltime = None
        self._m1b_depth_max_age_ns = int(float(self.get_parameter('depth_max_age_sec').value) * 1e9)
        if not 0 < self._m1b_depth_max_age_ns <= int(0.25 * 1e9):
            raise ValueError('Depth age must be in (0, 0.25] s to match training')
        self.temporal_mode = self.model_settings.get("temporal_mode", "historical")
        self.last_goal = (0.0, 0.0)
        self.last_goal_walltime: float | None = None
        self.last_grid_walltime: float | None = None
        self.goal_active = False
        self.command = (0.0, 0.0, 0.0)
        self.map_count = 0

        goal_topic = str(self.get_parameter("goal_topic").value)
        point_topic = str(self.get_parameter("clicked_point_topic").value)
        if self.representation == "tlom":
            grid_topic = "/mapless/temporal_occ_grid"
        else:
            grid_topic = str(self.get_parameter("local_grid_topic").value)
        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.goal_sub = self.create_subscription(Float32MultiArray, goal_topic, self.goal_callback, 10)
        self.point_sub = self.create_subscription(PointStamped, point_topic, self.point_callback, 10)
        self.occ_sub = self.create_subscription(OccupancyGrid, grid_topic, self.occ_callback, 10)
        self.lom_sub = None
        if self.model_settings.get("m123_dual_source"):
            self.lom_sub = self.create_subscription(
                OccupancyGrid, "/mapless/local_occ_grid", self.lom_callback, 10)
        self.depth_sub = None
        if self._m1b_depth_required:
            self.depth_sub = self.create_subscription(
                Image, str(self.get_parameter('depth_topic').value),
                self.m1b_depth_callback, qos_profile_sensor_data)
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
        self.hx = None
        self._m123_previous = None
        self._m123_lom_cache.clear()
        self._m123_tlom_cache.clear()
        if self._m1b_pair:
            self._m1b_reset_temporal()
            self._m1b_depth_cache.clear()
        self.get_logger().info(f"Clicked goal received in frame {msg.header.frame_id!r}; waiting for /NN_goal transform")
        self.publish_reached(False)

    def goal_active_callback(self, msg: Bool):
        was_active = self.goal_active
        self.goal_active = bool(msg.data)
        if self._m1b_pair and (not self.goal_active or not was_active):
            self._m1b_reset_temporal()
            self._m1b_depth_cache.clear()
        if self.goal_active and not was_active:
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            self._m123_lom_cache.clear()
            self._m123_tlom_cache.clear()
        if not self.goal_active:
            self._m123_lom_cache.clear()
            self._m123_tlom_cache.clear()
            self.last_goal = (0.0, 0.0)
            self.stop_command()
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
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

    @staticmethod
    def _grid_stamp(msg):
        return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

    def _m1b_reset_temporal(self):
        # Keep only causal state within the currently active navigation episode.
        self.hx = None
        self._m123_previous = None
        self._m1b_previous_depth = None
        self._m1b_last_step_stamp_ns = None
        self._m1b_last_depth_walltime = None
        self._m1b_pending_pairs.clear()

    def m1b_depth_callback(self, msg: Image):
        # The camera streams metric float32; training consumed a quantized PNG.
        try:
            frame = ros_depth_to_network(
                msg, max_range_m=self.model_settings['depth_range_max_m'])
            if not np.any(frame > 0):
                raise ValueError('Depth has no valid pixels')
            stamp_ns = stamp_to_ns(msg.header.stamp)
            if stamp_ns <= 0:
                raise ValueError('Depth timestamp missing')
            self._m1b_depth_cache.append((stamp_ns, frame, time.monotonic()))
            self._m1b_depth_cache.sort(key=lambda entry: entry[0])
            self._m1b_depth_cache = self._m1b_depth_cache[-12:]
            self._m1b_try_depth_pairs()
        except Exception as exc:
            self.stop_command()
            self.get_logger().warning(f'M1_B Depth conversion failed; stopped: {exc}')

    def _m1b_try_depth_pairs(self):
        # Process pending paired grids chronologically, with causal Depth only.
        if not self._m1b_depth_cache:
            return
        latest_ns = self._m1b_depth_cache[-1][0]
        for stamp in sorted(tuple(self._m1b_pending_pairs)):
            if stamp not in self._m1b_pending_pairs:
                break
            grid_ns = int(stamp[0]) * 1_000_000_000 + int(stamp[1])
            lom, tlom, queued_wall = self._m1b_pending_pairs[stamp]
            found = best_past_depth(
                grid_ns, [(ts, image) for ts,image,_ in self._m1b_depth_cache],
                self._m1b_depth_max_age_ns)
            if found is None:
                # Stop while waiting. Drop unrecoverable or timed-out samples.
                self.stop_command()
                if (latest_ns > grid_ns + self._m1b_depth_max_age_ns or
                        time.monotonic() - queued_wall > 0.5):
                    self._m1b_pending_pairs.pop(stamp, None)
                    self._m1b_reset_temporal()
                    break
                else:
                    break
            self._m1b_pending_pairs.pop(stamp, None)
            self.occ_callback(tlom, matched_lom=lom, matched_depth=found[1])

    def lom_callback(self, msg: OccupancyGrid):
        # Only M1 uses simultaneous local and temporal occupancy topics.
        if not self.model_settings.get('m123_dual_source'):
            return
        stamp=self._grid_stamp(msg)
        self._m123_lom_cache[stamp]=msg
        self._m123_match_pair(stamp)

    def _m123_match_pair(self, stamp):
        if stamp in self._m123_lom_cache and stamp in self._m123_tlom_cache:
            lom=self._m123_lom_cache.pop(stamp)
            tlom=self._m123_tlom_cache.pop(stamp)
            if self._m1b_depth_required:
                self._m1b_pending_pairs[stamp] = (lom, tlom, time.monotonic())
                # Avoid using an old velocity while waiting for an aligned image.
                self.stop_command()
                while len(self._m1b_pending_pairs) > 8:
                    self._m1b_pending_pairs.pop(min(self._m1b_pending_pairs))
                    self._m1b_reset_temporal()
                self._m1b_try_depth_pairs()
            else:
                self.occ_callback(tlom, matched_lom=lom)
        # Keep bounded buffers; never fuse sensor grids with different stamps.
        for cache in (self._m123_lom_cache,self._m123_tlom_cache):
            while len(cache)>8:
                del cache[min(cache)]

    def occ_callback(self, msg: OccupancyGrid, matched_lom=None, matched_depth=None):
        if self.model_settings.get('m123_dual_source') and matched_lom is None:
            stamp=self._grid_stamp(msg)
            self._m123_tlom_cache[stamp]=msg
            self._m123_match_pair(stamp)
            return
        if matched_lom is not None:
            a,b=matched_lom.info,msg.info
            if (a.height!=b.height or a.width!=b.width or
                    abs(a.resolution-b.resolution)>1e-5 or
                    abs(a.origin.position.x-b.origin.position.x)>1e-4 or
                    abs(a.origin.position.y-b.origin.position.y)>1e-4 or
                    len(matched_lom.data)!=len(msg.data)):
                self.stop_command()
                self.frames.clear()
                self.hx=None
                self._m123_previous=None
                if self._m1b_pair:
                    self._m1b_reset_temporal()
                self.get_logger().warning('M1 LOM/T-LOM geometry mismatch; stopping')
                return
        height, width = int(msg.info.height), int(msg.info.width)
        grid_variant = source_grid_variant((height, width), self.model_settings)
        if grid_variant is None:
            self.stop_command()
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            self.get_logger().warning(
                f"Unexpected ROS grid {height}x{width} for variant {self.variant}; stopping"
            )
            return
        expected_x, expected_y = VARIANT_ORIGIN[grid_variant]
        if (abs(float(msg.info.resolution) - 0.05) > 1e-5
                or abs(float(msg.info.origin.position.x) - expected_x) > 1e-4
                or abs(float(msg.info.origin.position.y) - expected_y) > 1e-4
                or len(msg.data) != height * width):
            self.stop_command()
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            self.get_logger().warning("Unexpected grid geometry or data length; stopping")
            return
        now = time.monotonic()
        if (self.last_grid_walltime is not None
                and now - self.last_grid_walltime > self.stale_timeout):
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
        self.last_grid_walltime = now
        if not self.goal_active:
            self.stop_command()
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            return

        distance, theta = self.last_goal
        if self.last_goal_walltime is None or time.monotonic() - self.last_goal_walltime > self.stale_timeout:
            self.stop_command()
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            return
        if distance <= self.goal_tolerance:
            self.stop_command()
            self.goal_active = False
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            self.publish_reached(True)
            return

        if self._m1b_depth_required and matched_depth is None:
            self.stop_command()
            return
        if self._m1b_pair:
            grid_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            last_ns = self._m1b_last_step_stamp_ns
            if last_ns is not None:
                if grid_ns <= last_ns:
                    self.stop_command()
                    self._m1b_reset_temporal()
                    return
                if grid_ns-last_ns > 1_000_000_000:
                    self._m1b_reset_temporal()
            self._m1b_last_step_stamp_ns = grid_ns
        occ_source = np.asarray(msg.data, dtype=np.float32).reshape(height, width)
        occ_source_max = float(np.max(occ_source))
        frame = prepare_observation(occ_source, distance, theta, self.model_settings)
        occ = frame[:-2]
        if self._m1b_pair:
            if matched_lom is None:
                raise RuntimeError('M1_B requires same-stamp LOM and T-LOM')
            local_source = np.asarray(matched_lom.data, dtype=np.float32).reshape(height,width)
            local_settings = {**self.model_settings, 'representation': 'lom'}
            local_frame = prepare_observation(local_source, distance, theta, local_settings)
            current_pair = np.stack((local_frame, frame), axis=0)
            sequence, depth_sequence = pack_m1b_input(
                current_pair, self._m123_previous,
                depth_current=matched_depth if self._m1b_depth_required else None,
                depth_previous=self._m1b_previous_depth)
            self._m123_previous = current_pair.copy()
            if self._m1b_depth_required:
                self._m1b_previous_depth = matched_depth.copy()
                self._m1b_last_depth_walltime = time.monotonic()
        elif self.model_settings.get('m1_refine_variant'):
            variant_name=self.model_settings['m1_refine_variant']
            if matched_lom is None:
                raise RuntimeError('M1 refinement requires synchronized LOM and T-LOM')
            local_source=np.asarray(matched_lom.data,dtype=np.float32).reshape(height,width)
            local_settings={**self.model_settings,'representation':'lom'}
            local_frame=prepare_observation(local_source,distance,theta,local_settings)
            current_pair=np.stack((local_frame,frame))
            if variant_name=='M1_B':
                previous=self._m123_previous if self._m123_previous is not None else current_pair
                packed=np.concatenate((previous,current_pair),axis=0)
            else:
                packed=current_pair
            self._m123_previous=current_pair.copy()
            sequence=packed[None,None,...]
            last_command=np.asarray(self.command,dtype=np.float32)
            vmax=np.asarray(self.vmax,dtype=np.float32)
            prior_cmd=np.clip(last_command/vmax,-1.0,1.0)
        elif self.model_settings.get('m123_variant'):
            variant_name=self.model_settings['m123_variant']
            if variant_name == 'M1':
                if matched_lom is None:
                    raise RuntimeError('M1 expected synchronized LOM/T-LOM')
                local_source=np.asarray(matched_lom.data,dtype=np.float32).reshape(height,width)
                local_settings={**self.model_settings,'representation':'lom'}
                local_frame=prepare_observation(local_source,distance,theta,local_settings)
                pair=np.stack((local_frame,frame))
            else:
                previous=self._m123_previous if self._m123_previous is not None else frame
                pair=np.stack((previous,frame))
            self._m123_previous=frame.copy()
            sequence=pair[None,None,...]
        elif self.temporal_mode == "stateful_spatial":
            # A single fresh observation updates the persistent spatial memory.
            # Never replay earlier frames with a non-zero recurrent state.
            sequence = frame[None, None, ...]
        else:
            self.frames.append(frame)
            if ((self.model_settings.get("preprocessing") is not None or self.model_settings.get("require_full_sequence"))
                    and len(self.frames) < self.sequence_length):
                self.stop_command()
                return
            while len(self.frames) < self.sequence_length:
                self.frames.appendleft(frame)
            sequence = np.stack(tuple(self.frames), axis=0)[None, ...]
        x = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)

        try:
            with torch.inference_mode():
                # New independent policies never carry hidden state over sliding windows.
                hx = None if self.temporal_mode == "independent" else self.hx
                if self._m1b_depth_required:
                    depth_tensor = torch.as_tensor(
                        depth_sequence, dtype=torch.float32, device=self.device)
                    output, new_hx = self.model(x, depth_tensor, hx=hx, return_state=True)
                elif self.model_settings.get('m1_refine_variant')=='M1retro_vel':
                    previous_command=torch.as_tensor(prior_cmd[None,None,:],
                        dtype=torch.float32,device=self.device)
                    output,new_hx=self.model(x,hx,return_state=True,cmd_history=previous_command)
                else:
                    output, new_hx = self.model(x, hx, return_state=True)
                if self.temporal_mode != "independent" and new_hx is not None:
                    if isinstance(new_hx, dict) and set(new_hx)=={'cells','prev_nn'}:
                        self.hx={'cells':[(h.detach(),c.detach()) for h,c in new_hx['cells']],
                                 'prev_nn':new_hx['prev_nn'].detach()}
                    elif isinstance(new_hx, tuple):
                        self.hx = tuple(h.detach() for h in new_hx)
                    elif isinstance(new_hx, torch.Tensor):
                        self.hx = new_hx.detach()
                    elif self.temporal_mode == "stateful_spatial":
                        self.hx = [(h.detach(), c.detach()) for h, c in new_hx]
                    else:
                        self.hx = [h.detach() for h in new_hx]
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
            self.frames.clear()
            self.hx = None
            self._m123_previous = None
            if self._m1b_pair:
                self._m1b_reset_temporal()
            self.get_logger().error(f"RNN inference failed; command stopped: {exc}")

    def publish_command(self):
        if self.cmd_pub is None:
            return
        now = time.monotonic()
        stale_grid = self.last_grid_walltime is None or now - self.last_grid_walltime > self.stale_timeout
        stale_goal = self.last_goal_walltime is None or now - self.last_goal_walltime > self.stale_timeout
        stale_depth = (self._m1b_depth_required and (
            self._m1b_last_depth_walltime is None or
            now - self._m1b_last_depth_walltime > self.stale_timeout))
        cmd = (0.0, 0.0, 0.0) if (stale_grid or stale_goal or stale_depth or not self.goal_active) else self.command
        if stale_grid or stale_goal or stale_depth:
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

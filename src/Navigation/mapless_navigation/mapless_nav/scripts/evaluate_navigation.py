#!/usr/bin/env python3
"""Run reproducible Webots navigation trials with independent pose observations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
HARDWARE = ROOT / "src/Hardware/justina_description/webots_env"
sys.path.insert(0, str(SCRIPT_DIR / "TorchModels"))
import sys
from pathlib import Path

EXPERIMENT_TOOLS_DIR = Path(__file__).resolve().parent / "experiment_tools"

if str(EXPERIMENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_TOOLS_DIR))

import experiment_control as control

ROUTE_CLASSES = ("straight", "semi_complex", "complex", "v_complex")


def rotate(xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def to_world(xy, offsets):
    return rotate(xy, offsets[2]) + np.asarray(offsets[:2])


def to_map(xy, offsets):
    return rotate(np.asarray(xy) - offsets[:2], -offsets[2])


def candidate(checkpoint, history_sha256):
    if history_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", history_sha256):
            raise ValueError("The training history SHA256 must contain 64 hexadecimal characters")
        matches = [p for p in (SCRIPT_DIR / "TorchModels/experiments").glob("train_*/training/history.json")
                   if control.file_hash(p) == history_sha256]
        if len(matches) != 1:
            raise ValueError(f"Expected one training run with this history hash; found {len(matches)}. Use --checkpoint explicitly.")
        directory = matches[0].parent
        checked = json.loads((directory / "checkpoint_check.json").read_text())
        history = json.loads(matches[0].read_text())
        best = min(history, key=lambda r: r["validation_loss"])
        if checked.get("status") != "passed" or checked["epoch"] != best["epoch"]:
            raise ValueError("The selected training checkpoint did not pass its reload check")
        checkpoint = directory / checked["checkpoint"]
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("Checkpoint must be a file")
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return {"path": str(checkpoint), "sha256": control.file_hash(checkpoint),
            "training_config": config, "history_sha256": history_sha256}


def launch_settings():
    source = SCRIPT_DIR.parent / "launch/mapless.launch.xml"
    tree = ET.parse(source).getroot()
    defaults = {a.attrib["name"]: a.attrib.get("default", "") for a in tree.findall("arg")}
    defaults.update(use_sim_time="true", enable_motion="true")

    def value(text):
        for key, item in defaults.items():
            text = text.replace("$(var " + key + ")", item)
        if "$(" in text:
            raise ValueError(f"Unsupported launch substitution: {text}")
        if text in ("true", "false"):
            return text == "true"
        try:
            return float(text) if any(c in text for c in ".eE") else int(text)
        except ValueError:
            return text

    nodes = []
    policy = None
    for n in tree.findall("node"):
        item = {"package": n.attrib["pkg"], "executable": n.attrib["exec"], "name": n.attrib["name"],
                "parameters": {p.attrib["name"]: value(p.attrib["value"]) for p in n.findall("param")},
                "remappings": [[r.attrib["from"], value(r.attrib["to"])] for r in n.findall("remap")]}
        if item["executable"] == "RNN_mapless.py":
            policy = item["parameters"]
        else:
            nodes.append(item)
    small = next(n for n in nodes if n["name"] == "local_occ_grid_nav")["parameters"]
    required = {"frame_id": "base_link", "min_x": 0.0, "max_x": 4.0, "min_y": -2.0, "max_y": 2.0,
                "resolution": 0.05, "min_z": 0.06, "max_z": 2.0, "use_lidar": True, "use_cloud": True,
                "lidar_downsample": 2, "cloud_downsample": 2, "map_topic": "/mapless/local_occ_grid"}
    if any(small.get(k) != v for k, v in required.items()) or policy is None:
        raise ValueError("Navigation LOM launch settings differ from the stage one reference")
    if policy.get("cmd_vel_topic") != "/hardware/base/cmd_vel":
        raise ValueError("The pilot requires the existing Justina velocity command topic")
    hardware_launch = HARDWARE / "launch/justina_mapless_wbt.launch.xml"
    cloud = next(n for n in ET.parse(hardware_launch).getroot().findall("node")
                 if n.attrib.get("exec") == "depth_to_pointcloud")
    cloud_params = {p.attrib["name"]: value(p.attrib["value"]) for p in cloud.findall("param")}
    return {"grid_nodes": nodes, "policy_parameters": policy, "cloud_parameters": cloud_params,
            "camera_environment": {k: os.environ.get(k, default) for k, default in
                                   [("JUSTINA_CAMERA_HZ", "10.0"), ("JUSTINA_CAMERA_ENABLE", "true"),
                                    ("JUSTINA_SCAN_DIVIDER", "2")]},
            "source_hashes": {str(p.relative_to(ROOT)): control.file_hash(p) for p in
                              [source, hardware_launch, HARDWARE / "controllers/justina_controller/justina_controller.py"]}}


# Grid extent for each variant. Values match mapless.launch.xml and the
# temporal_occ_grid_node configuration used during recording.
VARIANT_GRID_EXTENT = {
    "4x4": {"min_x": 0.0, "max_x": 4.0, "min_y": -2.0, "max_y": 2.0},
    "5x4": {"min_x": -1.0, "max_x": 4.0, "min_y": -2.0, "max_y": 2.0},
    "7x6": {"min_x": -1.0, "max_x": 6.0, "min_y": -3.0, "max_y": 3.0},
}


def evaluation_grid_variant(training_config):
    variant = training_config.get("variant", "4x4")
    if variant not in VARIANT_GRID_EXTENT:
        raise ValueError(f"Unknown variant in checkpoint: {variant!r}")
    if (training_config.get("data_format") == "ros2_episode"
            and training_config.get("preprocessing") is None
            and "variant" in training_config):
        return "7x6"
    return variant


def evaluation_settings(grid_variant):
    settings = launch_settings()
    grid_extent = VARIANT_GRID_EXTENT[grid_variant]
    new_grid_nodes = []
    replaced = False
    for node in settings["grid_nodes"]:
        if node["name"] == "local_occ_grid_nav":
            new_grid_nodes.append({
                "package": "mapless_nav",
                "executable": "temporal_occ_grid_node",
                "name": "temporal_occ_grid_nav",
                "parameters": {
                    "use_sim_time": True,
                    "base_frame": "base_link",
                    "odom_frame": "odom",
                    "map_topic": "/mapless/local_occ_grid",
                    "temporal_map_topic": "/mapless/temporal_occ_grid",
                    **grid_extent,
                    "min_z": 0.20,
                    "max_z": 2.0,
                    "resolution": 0.05,
                    "lidar_downsample": 1,
                    "cloud_downsample": 2,
                    "use_lidar": True,
                    "use_cloud": True,
                },
                "remappings": [],
            })
            replaced = True
        elif node["name"] == "local_occ_grid_rviz":
            # Skip the preview grid; the temporal node publishes both topics.
            continue
        else:
            new_grid_nodes.append(node)
    if not replaced:
        raise ValueError("launch_settings did not return a local_occ_grid_nav node")
    settings["grid_nodes"] = new_grid_nodes
    return settings


def prepare_suite(args, out, seed, metadata, model):
    import yaml

    if args.scenario not in control.ALL_SCENARIOS:
        raise ValueError("Scenario must be E01 through E30")
    if args.purpose == "development" and args.scenario in metadata["config"]["final_scenarios"]:
        raise ValueError("This scenario is reserved for final evaluation. Use --purpose final for the frozen final experiment.")
    world = HARDWARE / "worlds" / (args.scenario + ".wbt")
    if not world.is_file():
        raise ValueError(f"Webots world not found: {world}")
    background = control.load_review_map(HARDWARE / "environments/maps", args.scenario, "prohibition")
    map_metadata = yaml.safe_load(Path(background["source"]["yaml"]).read_text())
    offsets = [float(map_metadata.get("world_offset_" + axis, 0)) for axis in ("x", "y", "yaw")]
    if not np.isfinite(offsets).all():
        raise ValueError("Map world offsets must be finite")
    grid_variant = evaluation_grid_variant(model["training_config"])
    settings = evaluation_settings(grid_variant)
    policy_overrides = {}
    if args.turn_help != "from-launch":
        policy_overrides["turn_help"] = args.turn_help
    for name, value in (("max_vx", args.max_vx), ("max_vy", args.max_vy), ("max_wz", args.max_wz)):
        if value is not None:
            if not (value > 0.0 and math.isfinite(value)):
                raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
            policy_overrides[name] = str(value)
    sources = {"world_sha256": control.file_hash(world), "map_yaml_sha256": background["source"]["yaml_sha256"],
               "map_image_sha256": background["source"]["image_sha256"], "runtime": settings}
    if args.suite:
        suite = json.loads(args.suite.expanduser().read_text())
        stored = {k: v for k, v in suite["sources"].items() if k != "policy_overrides"}
        stored_overrides = suite["sources"].get("policy_overrides", {})
        if (suite.get("version") != 1 or suite["scenario"] != args.scenario or suite["purpose"] != args.purpose
                or stored != sources):
            raise ValueError("Evaluation suite and current scenario or runtime settings do not match")
        if stored_overrides and stored_overrides != policy_overrides:
            raise ValueError(
                "The reused suite was recorded with different policy overrides: "
                f"stored={stored_overrides}, requested={policy_overrides}. "
                "Create a new suite without --suite to run with different overrides."
            )
    else:
        generator = ROOT / "src/Navigation/mapless_navigation/dataset/scripts/data_analysis/generate_path_pool.py"
        command = [sys.executable, "-u", str(generator), "--scenario", args.scenario,
                   "--output-dir", str(out / "generated_pool"), "--maps-root", str(HARDWARE / "environments/maps"),
                   "--per-class", "1", "--seed", str(seed), "--timeout-seconds", "120",
                   "--v-max-goal-distance-m", "3.0"]
        print("Generating an independent evaluation task pool.", flush=True)
        result = subprocess.run(command, check=False)
        pools = list((out / "generated_pool" / args.scenario).rglob("path_pool.json"))
        if len(pools) != 1:
            raise ValueError(f"Task generation returned {result.returncode} and produced {len(pools)} pool files")
        pool_path = pools[0]
        pool = json.loads(pool_path.read_text())
        if (result.returncode != 0 or pool.get("pool_version") != 5 or pool.get("scenario") != args.scenario
                or pool.get("seed") != seed or not pool.get("generation_complete")
                or sorted(p.get("class", "") for p in pool.get("paths", [])) != sorted(ROUTE_CLASSES)):
            raise ValueError("The task generator did not complete the requested scenario, seed and route classes")
        tasks = []
        for p in sorted(pool["paths"], key=lambda p: (ROUTE_CLASSES.index(p["class"]), p["id"])):
            start, goal = p["start"], p["goal"]
            yaw = math.atan2(goal[1] - start[1], goal[0] - start[0]) if (args.heading or "toward_goal") == "toward_goal" else p["start_yaw_rad"]
            tasks.append({"id": p["id"], "start": start, "goal": goal, "start_yaw_rad": yaw,
                          "generator_start_yaw_rad": p["start_yaw_rad"], "route_class": p["class"],
                          "task_type": p.get("task_type"), "direct_line_safe": p["direct_line_safe"],
                          "behavior_tags": []})
        suite = {"version": 1, "scenario": args.scenario, "purpose": args.purpose, "seed": seed,
                 "sources": sources, "offsets": offsets, "map_source": background["source"],
                 "heading_policy": args.heading or "toward_goal", "generator_sha256": control.file_hash(generator),
                 "pool_complete": pool.get("generation_complete", False), "tasks": tasks,
                 "protocol": {"timeout_s": args.timeout or 60.0, "goal_tolerance_m": 0.25, "settle_s": 1.5,
                              "start_tolerance_m": 0.08, "start_yaw_tolerance_rad": 0.12,
                              "collision_threshold": 0.5,
                              "collision_source": "world collision_detector TouchSensor",
                              "pose_source": "Webots Supervisor getSelf position and orientation",
                              "input_monitoring": "diagnostic_only", "input_diagnostic_age_wall_s": 2.0,
                              "world_random_seed": max(1, int(seed % 2147483647)), "simulation_mode": "realtime"}}
    if args.suite and ((args.timeout is not None and args.timeout != suite["protocol"]["timeout_s"])
                       or (args.heading is not None and args.heading != suite["heading_policy"])):
        raise ValueError("A reused suite keeps its original time limit and initial headings")
    if not suite.get("tasks") or len({p["id"] for p in suite["tasks"]}) != len(suite["tasks"]):
        raise ValueError("The suite must contain distinct task identifiers")
    for task in suite["tasks"]:
        values = [*task["start"], *task["goal"], task["start_yaw_rad"]]
        if len(task["start"]) != 2 or len(task["goal"]) != 2 or not np.isfinite(values).all() or math.dist(task["start"], task["goal"]) < 0.5:
            raise ValueError("Invalid task coordinates")
    if policy_overrides:
        suite["sources"]["policy_overrides"] = policy_overrides
    control.write_json(out / "suite.json", suite)
    return suite


def node_end(text, opening):
    depth, quoted, escaped, comment = 0, False, False, False
    for index in range(opening, len(text)):
        c = text[index]
        if comment:
            comment = c != "\n"
            continue
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
            continue
        if c == '#':
            comment = True
        elif c == '"':
            quoted = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("Unbalanced Webots node braces")


def runtime_world(source, target, seed):
    text = source.read_text()
    if len(re.findall(r'\bcontroller\s+"justina_controller"', text)) != 1:
        raise ValueError("Expected exactly one Justina controller in the source world")
    robot = re.search(r'\bDEF\s+Justina\s+Robot\s*\{', text)
    if not robot or not re.search(r'\bsupervisor\s+TRUE', text[robot.start():node_end(text, robot.end() - 1)]):
        raise ValueError("Expected an existing Supervisor-enabled DEF Justina Robot")
    text = re.sub(r'\bcontroller\s+"justina_controller"', 'controller "mapless_evaluation_bridge"', text)
    match = re.search(r'\bWorldInfo\s*\{', text)
    if not match:
        raise ValueError("WorldInfo is missing")
    end = node_end(text, match.end() - 1)
    section = text[match.start():end]
    if re.search(r'\brandomSeed\s+-?\d+', section):
        section = re.sub(r'\brandomSeed\s+-?\d+', f'randomSeed {seed}', section)
    else:
        section = section[:match.end() - match.start()] + f"\n  randomSeed {seed}" + section[match.end() - match.start():]
    text = text[:match.start()] + section + text[end:]

    def absolute_url(match):
        name = match.group(1)
        if name.startswith(("./", "../")):
            return json.dumps(str((source.parent / name).resolve()))
        return match.group(0)

    text = re.sub(r'"([^"\n]+)"', absolute_url, text)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def prepare_rviz(out):
    import yaml

    share = HARDWARE.parent
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("justina_description"))
    except ImportError:
        pass
    source_launch = share / "webots_env/launch/justina_mapless_wbt.launch.xml"
    launch_tree = ET.parse(source_launch).getroot()
    rviz_node = next((n for n in launch_tree.findall("node") if n.attrib.get("exec") == "rviz2"), None)
    if rviz_node is None:
        raise ValueError(f"RViz configuration is missing from {source_launch}")
    arguments = shlex.split(rviz_node.attrib["args"].replace("$(var package_share)", shlex.quote(str(share))))
    source = Path(arguments[arguments.index("-d") + 1]).resolve(strict=True)
    document = yaml.safe_load(source.read_text())
    displays = document["Visualization Manager"]["Displays"]
    if not any(d.get("Topic", {}).get("Value") == "/evaluation/goal_marker" for d in displays):
        displays.append({"Class": "rviz_default_plugins/Marker", "Enabled": True,
                         "Name": "Evaluation goals and measured path", "Value": True, "Namespaces": {},
                         "Topic": {"Depth": 1, "Durability Policy": "Volatile", "Filter size": 10,
                                   "History Policy": "Keep Last", "Reliability Policy": "Reliable",
                                   "Value": "/evaluation/goal_marker"}})
    target = out / "evaluation.rviz"
    target.write_text(yaml.safe_dump(document, sort_keys=False))
    return {"source": str(source), "source_sha256": control.file_hash(source),
            "launch": str(source_launch), "path": str(target), "sha256": control.file_hash(target)}


def input_monitoring(suite):
    protocol = suite["protocol"]
    return {"mode": "diagnostic_only", "abort_on_input_delay": False,
            "diagnostic_age_wall_s": protocol.get("input_diagnostic_age_wall_s", protocol.get("input_watchdog_wall_s", 2.0))}


def prepare_runtime(out, suite, model, limit, domain, route_class=None, rviz=False, preview_seconds=5.0, close_after_trial=False):
    suite = json.loads(json.dumps(suite))
    if route_class:
        suite["tasks"] = [task for task in suite["tasks"] if task["route_class"] == route_class]
        if not suite["tasks"]:
            raise ValueError("No tasks match the requested route class")
    project = out / "webots_project"
    controllers = project / "controllers"
    controllers.mkdir(parents=True)
    bridge = controllers / "mapless_evaluation_bridge"
    bridge.mkdir()
    config_path = out / "runtime.json"
    config = {"version": 1, "run_id": out.name, "suite": suite, "model": model,
              "input_monitoring": input_monitoring(suite),
              "rviz": rviz, "preview_seconds": preview_seconds if rviz else 0.0,
              "keep_open_after_trial": rviz and not close_after_trial,
              "rviz_config": prepare_rviz(out) if rviz else None,
              "domain_id": domain, "limit": min(limit, len(suite["tasks"])),
              "source_script": str(Path(__file__).resolve()),
              "evaluator_sha256": control.file_hash(Path(__file__)),
              "policy_source_sha256": control.file_hash(SCRIPT_DIR / "RNN_mapless.py"),
              "model_definition_sha256": control.file_hash(SCRIPT_DIR / "TorchModels/models.py"),
              "controller_source": str(HARDWARE / "controllers/justina_controller/justina_controller.py"),
              "urdf": str(HARDWARE.parent / "urdf/justina.urdf"),
              "world": str(project / "worlds" / (suite["scenario"] + ".wbt"))}
    if not Path(config["urdf"]).is_file():
        raise ValueError(f"Robot description not found: {config['urdf']}")
    (bridge / "mapless_evaluation_bridge.py").write_text(
        "import runpy\nnamespace = runpy.run_path(" + repr(str(Path(__file__).resolve())) + ")\n"
        "namespace['run_bridge'](" + repr(str(config_path)) + ")\n")
    runtime_world(HARDWARE / "worlds" / (suite["scenario"] + ".wbt"), Path(config["world"]),
                  suite["protocol"]["world_random_seed"])
    control.write_json(config_path, config)
    launch = out / "evaluation.launch.py"
    launch.write_text("import runpy\ndef generate_launch_description():\n    return runpy.run_path("
                      + repr(str(Path(__file__).resolve())) + ")[\"make_launch\"](" + repr(str(config_path)) + ")\n")
    return config, launch


def make_launch(config_path):
    from launch import LaunchDescription
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue

    cfg = json.loads(Path(config_path).read_text())
    settings = cfg["suite"]["sources"]["runtime"]
    overrides = cfg["suite"]["sources"].get("policy_overrides", {})
    webots = ExecuteProcess(cmd=[shutil.which("webots") or "webots", "--mode=realtime", "--batch", cfg["world"]],
                           additional_env={"WEBOTS_CONTROLLER_PATH": str(Path(cfg["world"]).parents[1] / "controllers"),
                                           "PYTHONDONTWRITEBYTECODE": "1",
                                           **settings["camera_environment"]}, output="screen")
    policy_params = dict(settings["policy_parameters"])
    for key, value in overrides.items():
        if key == "turn_help":
            policy_params[key] = (value == "true") if isinstance(value, str) else bool(value)
        elif key in ("max_vx", "max_vy", "max_wz"):
            policy_params[key] = float(value)
    params = {**policy_params, "checkpoint": cfg["model"]["path"], "model_filename": "",
              "use_sim_time": True, "enable_motion": True, "goal_tolerance": cfg["suite"]["protocol"]["goal_tolerance_m"]}
    command = [sys.executable, "-u", str(SCRIPT_DIR / "RNN_mapless.py"), "--ros-args", "-r", "__node:=rnn_controller"]
    for key, value in params.items():
        command.extend(["-p", f"{key}:={json.dumps(value)}"])
    policy = ExecuteProcess(cmd=command, output="screen")
    nodes = [webots, Node(package="robot_state_publisher", executable="robot_state_publisher",
                          parameters=[{"robot_description": ParameterValue(Path(cfg["urdf"]).read_text(), value_type=str),
                                       "use_sim_time": True}],
                          remappings=[("joint_states", "/hardware/joint_states"), ("robot_description", "/hardware/robot_description")]),
             Node(package="justina_description", executable="depth_to_pointcloud", name="depth_to_pointcloud",
                  parameters=[settings["cloud_parameters"]], output="screen")]
    for n in settings["grid_nodes"]:
        nodes.append(Node(package=n["package"], executable=n["executable"], name=n["name"],
                          parameters=[n["parameters"]], remappings=[tuple(rule) for rule in n["remappings"]], output="screen"))
    nodes.extend([Node(package="mapless_nav", executable="get_goal_point", name="evaluation_goal_projection",
                       parameters=[{"use_sim_time": True, "goal_topic": "/evaluation/policy_goal"}], output="screen"), policy])
    for process in (webots, policy):
        nodes.append(RegisterEventHandler(OnProcessExit(target_action=process,
                     on_exit=[EmitEvent(event=Shutdown(reason="Evaluation process exited"))])))
    if cfg.get("rviz"):
        nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2",
                          parameters=[{"use_sim_time": True}], arguments=["-d", cfg["rviz_config"]["path"]], output="screen"))
    return LaunchDescription(nodes)


def load_controller_source(path):
    module = types.ModuleType("evaluation_original_controller")
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def run_bridge(config_path):
    import rclpy
    from controller import Supervisor
    from std_msgs.msg import String

    cfg = json.loads(Path(config_path).read_text())
    original = load_controller_source(cfg["controller_source"])
    rclpy.init()
    robot = Supervisor()
    node = original.JustinaController(robot)
    publisher = node.create_publisher(String, "/evaluation/telemetry", 20)
    body = robot.getSelf()
    sequence = 0
    try:
        while node.step():
            position, orientation = body.getPosition(), body.getOrientation()
            present = node.collision_sensor is not None
            value = float(node.collision_sensor.getValue()) if present else None
            pose = [float(position[0]), float(position[1]), math.atan2(orientation[3], orientation[0])]
            packet = {"run_id": cfg["run_id"], "sequence": sequence, "time": robot.getTime(),
                      "pose": pose if np.isfinite(pose).all() else None,
                      "collision_sensor_present": present, "collision_value": value,
                      "odom": list(node.base.get_odom()),
                      "command": [node.base.target_vx, node.base.target_vy, node.base.target_w]}
            publisher.publish(String(data=json.dumps(packet, allow_nan=False)))
            sequence += 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


class TrialMetrics:
    def __init__(self, task, protocol, offsets):
        self.task, self.protocol, self.offsets = task, protocol, offsets
        self.samples = []
        self.collisions = 0
        self.previous_sign = [0, 0]
        self.sign_changes = [0, 0]

    def append(self, packet):
        if self.samples and packet["time"] <= self.samples[-1]["time"]:
            raise ValueError("Simulation telemetry must advance monotonically")
        self.samples.append(packet)
        self.collisions += int(packet["collision_value"] > self.protocol["collision_threshold"])
        for index, (component, deadband) in enumerate([(1, 0.02), (2, 0.05)]):
            value = packet["command"][component]
            sign_value = 1 if value > deadband else -1 if value < -deadband else 0
            if sign_value:
                self.sign_changes[index] += int(self.previous_sign[index] not in (0, sign_value))
                self.previous_sign[index] = sign_value

    def distance(self, packet):
        return float(np.linalg.norm(to_map(packet["pose"][:2], self.offsets) - self.task["goal"]))

    def finish(self, reason):
        positions = np.array([to_map(p["pose"][:2], self.offsets) for p in self.samples])
        length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0
        start, goal = np.asarray(self.task["start"]), np.asarray(self.task["goal"])
        direction = goal - start
        direct = float(np.linalg.norm(direction))
        signed = (direction[0] * (positions[:, 1] - start[1]) - direction[1] * (positions[:, 0] - start[0])) / direct
        final = self.distance(self.samples[-1])
        elapsed = self.samples[-1]["time"] - self.samples[0]["time"]
        arrived = final <= self.protocol["goal_tolerance_m"]
        return {"task_id": self.task["id"], "route_class": self.task["route_class"], "reason": reason,
                "success": bool(reason == "goal" and arrived and not self.collisions), "arrived": arrived,
                "collision": bool(self.collisions), "collision_samples": self.collisions,
                "elapsed_sim_s": elapsed, "path_length_m": length, "direct_distance_m": direct,
                "final_goal_distance_m": final,
                "progress_efficiency": min(1.0, max(0.0, direct - final) / length) if length > 0 else 0.0,
                "max_left_deviation_from_goal_line_m": max(0.0, float(signed.max())),
                "max_right_deviation_from_goal_line_m": max(0.0, float(-signed.min())),
                "lateral_command_sign_changes": self.sign_changes[0], "angular_command_sign_changes": self.sign_changes[1],
                "behavior_tags": self.task.get("behavior_tags", []),
                "minimum_robot_clearance_m": None,
                "clearance_status": "not measured by this contact and pose pilot"}


def summarize(results, errors, requested):
    count = len(results)
    return {"status": "invalid" if errors or count != requested else "completed", "requested_trials": requested,
            "completed_trials": count, "infrastructure_errors": errors,
            "successes": sum(r["success"] for r in results),
            "success_rate": sum(r["success"] for r in results) / count if count else None,
            "collision_rate": sum(r["collision"] for r in results) / count if count else None,
            "not_run": max(0, requested - count), "results": results}


def parse_packet(text, run_id):
    packet = json.loads(text)
    if not isinstance(packet, dict):
        raise ValueError("Simulator telemetry must be a JSON object")
    if packet.get("run_id") != run_id:
        return None
    if (packet.get("collision_sensor_present") is not True or type(packet.get("sequence")) is not int
            or packet["sequence"] < 0 or not isinstance(packet.get("pose"), list)
            or len(packet["pose"]) != 3 or len(packet.get("odom", [])) != 3
            or len(packet.get("command", [])) != 3):
        raise ValueError("Independent pose or collision sensor is unavailable")
    values = [packet.get("time"), packet.get("collision_value"), *packet["pose"], *packet["odom"], *packet["command"]]
    try:
        valid = np.isfinite(np.asarray(values, dtype=float)).all()
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Non-finite simulator telemetry")
    return packet


def task_markers(task, metrics, offsets, Marker, Point):
    markers = []
    for identifier, xy, color in [(0, task["goal"], (1.0, 0.1, 0.1)),
                                   (1, task["start"], (0.1, 1.0, 0.1))]:
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.ns, marker.id = "evaluation", identifier
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        target = rotate(np.asarray(xy) - task["start"], -task["start_yaw_rad"])
        marker.pose.position.x, marker.pose.position.y = map(float, target)
        marker.pose.position.z, marker.pose.orientation.w = 0.2, 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = *color, 1.0
        markers.append(marker)
    line = Marker()
    line.header.frame_id = "odom"
    line.ns, line.id = "evaluation", 2
    line.type, line.action = Marker.LINE_STRIP, Marker.ADD
    line.pose.orientation.w, line.scale.x = 1.0, 0.035
    line.color.b, line.color.g, line.color.a = 1.0, 0.5, 1.0
    for packet in metrics.samples if metrics else []:
        target = rotate(to_map(packet["pose"][:2], offsets) - task["start"], -task["start_yaw_rad"])
        line.points.append(Point(x=float(target[0]), y=float(target[1]), z=0.08))
    if len(line.points) < 2:
        line.action = Marker.DELETE
    markers.append(line)
    label = Marker()
    label.header.frame_id = "odom"
    label.ns, label.id = "evaluation", 3
    label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
    label.pose.position.z, label.pose.orientation.w = 0.8, 1.0
    label.scale.z = 0.2
    label.color.r = label.color.g = label.color.b = label.color.a = 1.0
    label.text = f"Task {task['id']} | {task['route_class']}"
    markers.append(label)
    return markers


def save_session_result(node, cfg, out):
    result = summarize(node.results, node.errors, cfg["limit"])
    result.update(model=cfg["model"], scenario=cfg["suite"]["scenario"], purpose=cfg["suite"]["purpose"],
                  suite_sha256=control.file_hash(out / "suite.json"), protocol=cfg["suite"]["protocol"],
                  selected_task_ids=[task["id"] for task in cfg["suite"]["tasks"][:cfg["limit"]]],
                  input_monitoring=input_monitoring(cfg["suite"]), rviz_config=cfg.get("rviz_config"),
                  keep_open_after_trial=cfg.get("keep_open_after_trial", False))
    result = json.loads(json.dumps(result))
    control.write_json(out / "summary.json", result)
    return result


def keep_open_for_review(node, process, ros):
    print("Trial saved. Webots and RViz remain open for inspection. Press Ctrl+C to close.", flush=True)
    try:
        while ros.ok() and process.poll() is None:
            node.stop()
            for marker in node.last_markers:
                node.marker_pub.publish(marker)
            ros.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        print("Closing the inspection session.", flush=True)
    except Exception:
        if ros.ok():
            raise


def run_session(cfg, launch_path, out, startup_timeout):
    import rclpy
    from geometry_msgs.msg import Point, PointStamped, Pose2D, Twist
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan, PointCloud2
    from std_msgs.msg import Bool, Float32MultiArray, String
    from visualization_msgs.msg import Marker

    class Evaluator(Node):
        def __init__(self):
            super().__init__("mapless_evaluation")
            self.packet, self.packet_wall, self.previous_sequence = None, 0.0, -1
            self.phase, self.phase_wall, self.phase_sim = "startup", time.monotonic(), 0.0
            self.index, self.results, self.errors, self.metrics = 0, [], [], None
            self.goal_valid, self.ack, self.reached = False, False, False
            self.last_theta = None
            self.can_command = False
            self.grid_count, self.grid_wall, self.scan_wall, self.cloud_wall = 0, 0.0, 0.0, 0.0
            training_config = cfg["model"]["training_config"]
            self.grid_variant = evaluation_grid_variant(training_config)
            self.grid_topic = ("/mapless/temporal_occ_grid" if training_config.get("representation") == "tlom"
                               else "/mapless/local_occ_grid")
            self.stream_messages = {name: {"messages": 0, "last_size": None, "stamp": None}
                                    for name in ("grid", "scan", "cloud")}
            self.last_marker_wall = 0.0
            self.last_markers = []
            self.last_input_diagnostic_wall = 0.0
            self.input_health_log = (out / "input_health.jsonl").open("a")
            self.marker_pub = self.create_publisher(Marker, "/evaluation/goal_marker", 10)
            self.teleport_pub = self.create_publisher(Pose2D, "/webots/teleport", 10)
            self.active_pub = self.create_publisher(Bool, "/mapless_nav/goal_active", 10)
            self.stop_pub = self.create_publisher(Twist, "/hardware/base/cmd_vel", 10)
            self.project_pub = self.create_publisher(PointStamped, "/evaluation/policy_goal", 10)
            self.start_pub = self.create_publisher(PointStamped, "/clicked_point", 10)
            self.create_subscription(String, "/evaluation/telemetry", self.on_packet, 20)
            self.create_subscription(Bool, "/maples_nav/goal_reached", self.on_reached, 10)
            self.create_subscription(Float32MultiArray, "/NN_goal", self.on_goal, 10)
            self.create_subscription(OccupancyGrid, self.grid_topic, self.on_grid, qos_profile_sensor_data)
            self.create_subscription(LaserScan, "/hardware/base/scan", self.on_scan, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, "/hardware/camera/points", self.on_cloud, qos_profile_sensor_data)
            self.trajectory = (out / "trajectory.csv").open("w", newline="")
            self.writer = csv.writer(self.trajectory)
            self.writer.writerow(["task_id", "simulation_time", "world_x", "world_y", "world_yaw",
                                  "odom_x", "odom_y", "odom_yaw", "cmd_vx", "cmd_vy", "cmd_wz", "collision_value",
                                  "theta_input", "age_grid_s", "age_scan_s", "age_cloud_s"])

        @property
        def task(self):
            return cfg["suite"]["tasks"][self.index]

        def fail(self, reason, details=None):
            error = {"phase": self.phase, "reason": reason, "trial_index": self.index}
            if details is not None:
                error["inputs"] = details
            self.errors.append(error)
            self.phase = "done"
            self.stop()

        def receive(self, name, msg, size):
            now = time.monotonic()
            setattr(self, name + "_wall", now)
            entry = self.stream_messages[name]
            entry.update(messages=entry["messages"] + 1, last_size=int(size))
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            if stamp is not None:
                entry["stamp"] = float(stamp.sec) + float(stamp.nanosec) / 1e9

        def input_health(self, wall):
            topics = {"grid": self.grid_topic, "scan": "/hardware/base/scan",
                      "cloud": "/hardware/camera/points"}
            health = {}
            for name, topic in topics.items():
                received = getattr(self, name + "_wall")
                age = max(0.0, wall - received) if received else None
                health[topic] = {**self.stream_messages[name], "age_wall_s": age,
                                 "publishers": self.count_publishers(topic),
                                 "stale": age is None or age > input_monitoring(cfg["suite"])["diagnostic_age_wall_s"]}
            return health

        def save_input_diagnostics(self, wall, force=False):
            if force or wall - self.last_input_diagnostic_wall >= 1.0:
                health = self.input_health(wall)
                control.write_json(out / "input_health.json", health)
                record = {"time_wall_s": wall,
                          "sim_time_s": self.packet["time"] if self.packet else None,
                          "inputs": health}
                self.input_health_log.write(json.dumps(record, allow_nan=False) + "\n")
                self.input_health_log.flush()
                self.last_input_diagnostic_wall = wall

        def stop(self):
            if self.can_command:
                self.active_pub.publish(Bool(data=False))
                self.stop_pub.publish(Twist())

        def on_goal(self, msg):
            if len(msg.data) >= 2:
                self.last_theta = float(msg.data[1])
            if self.phase == "project" and len(msg.data) >= 2:
                expected = math.dist(self.task["start"], self.task["goal"])
                delta = rotate(np.asarray(self.task["goal"]) - self.task["start"], -self.task["start_yaw_rad"])
                expected_angle = math.atan2(delta[1], delta[0])
                angle_error = float(msg.data[1]) - expected_angle
                self.goal_valid = (abs(float(msg.data[0]) - expected) < 0.15
                                   and abs(math.atan2(math.sin(angle_error), math.cos(angle_error))) < 0.10)

        def on_grid(self, msg):
            if self.phase == "done":
                return
            extent = VARIANT_GRID_EXTENT[self.grid_variant]
            expected_w = int(round((extent["max_x"] - extent["min_x"]) / 0.05))
            expected_h = int(round((extent["max_y"] - extent["min_y"]) / 0.05))
            expected_cells = expected_w * expected_h
            if ((msg.info.height, msg.info.width) != (expected_h, expected_w)
                    or len(msg.data) != expected_cells
                    or not math.isclose(msg.info.resolution, 0.05, abs_tol=1e-6)
                    or msg.header.frame_id != "base_link"
                    or not math.isclose(msg.info.origin.position.x, extent["min_x"], abs_tol=1e-6)
                    or not math.isclose(msg.info.origin.position.y, extent["min_y"], abs_tol=1e-6)
                    or min(msg.data) < 0 or max(msg.data) > 100):
                self.fail(
                    f"The live navigation LOM does not match the recorded geometry "
                    f"and scale on {self.grid_topic}; expected {expected_w}x{expected_h} "
                    f"at origin ({extent['min_x']}, {extent['min_y']}), received "
                    f"{msg.info.width}x{msg.info.height} at origin "
                    f"({msg.info.origin.position.x:.3f}, {msg.info.origin.position.y:.3f})"
                )
                return
            self.grid_count += 1
            self.receive("grid", msg, len(msg.data))

        def on_scan(self, msg):
            self.receive("scan", msg, len(msg.ranges))

        def on_cloud(self, msg):
            self.receive("cloud", msg, msg.width * msg.height)

        def on_reached(self, msg):
            if self.phase == "arming" and not msg.data:
                self.ack = True
            elif self.phase == "active" and msg.data:
                self.reached = True

        def on_packet(self, msg):
            if self.phase == "done":
                return
            try:
                packet = parse_packet(msg.data, cfg["run_id"])
            except (ValueError, TypeError) as exc:
                self.fail(str(exc))
                return
            if packet is None:
                return
            if packet["sequence"] <= self.previous_sequence:
                self.fail("Telemetry sequence restarted or moved backwards")
                return
            self.previous_sequence = packet["sequence"]
            self.packet, self.packet_wall = packet, time.monotonic()
            if self.phase == "active":
                self.record(packet)
                if packet["collision_value"] > cfg["suite"]["protocol"]["collision_threshold"]:
                    self.finish("collision")
                elif self.metrics.distance(packet) <= cfg["suite"]["protocol"]["goal_tolerance_m"]:
                    self.finish("goal")
                elif packet["time"] - self.phase_sim >= cfg["suite"]["protocol"]["timeout_s"]:
                    self.finish("timeout")

        def record(self, packet):
            self.metrics.append(packet)
            health = self.input_health(time.monotonic())
            ages = {name: health[topic].get("age_wall_s") for name, topic in (
                ("grid", self.grid_topic),
                ("scan", "/hardware/base/scan"),
                ("cloud", "/hardware/camera/points"))}
            self.writer.writerow([self.task["id"], packet["time"], *packet["pose"], *packet["odom"],
                                  *packet["command"], packet["collision_value"],
                                  self.last_theta, ages["grid"], ages["scan"], ages["cloud"]])

        def change(self, phase):
            self.phase, self.phase_wall = phase, time.monotonic()
            self.phase_sim = self.packet["time"] if self.packet else 0.0
            print(f"Evaluation phase: {phase}; trial {self.index + 1}/{cfg['limit']}", flush=True)

        def goal_message(self):
            target = rotate(np.asarray(self.task["goal"]) - self.task["start"], -self.task["start_yaw_rad"])
            message = PointStamped()
            message.header.frame_id = "odom"
            message.point.x, message.point.y = float(target[0]), float(target[1])
            return message

        def finish(self, reason):
            self.stop()
            result = self.metrics.finish(reason)
            self.save_input_diagnostics(time.monotonic(), force=True)
            self.publish_markers()
            self.results.append(result)
            self.trajectory.flush()
            control.write_json(out / f"trial_{self.index:03d}.json", result)
            print("Trial result: " + json.dumps(result), flush=True)
            self.index += 1
            if self.index >= cfg["limit"]:
                self.phase = "done"
            else:
                self.metrics = None
                self.change("between")

        def publish_markers(self):
            self.last_markers = task_markers(self.task, self.metrics, cfg["suite"]["offsets"], Marker, Point)
            for marker in self.last_markers:
                self.marker_pub.publish(marker)
            self.last_marker_wall = time.monotonic()

        def tick(self):
            if self.phase == "done":
                return
            wall = time.monotonic()
            if self.can_command:
                self.save_input_diagnostics(wall)
            if self.can_command and self.index < cfg["limit"] and wall - self.last_marker_wall >= 0.5:
                self.publish_markers()
            if self.phase == "startup":
                if wall - self.phase_wall > startup_timeout:
                    self.fail("Startup timed out waiting for Webots, policy and goal projection")
                    return
                if (self.packet and self.grid_count >= 2 and self.scan_wall and self.cloud_wall
                        and self.project_pub.get_subscription_count() and self.start_pub.get_subscription_count()
                        and self.teleport_pub.get_subscription_count()):
                    counts = [self.count_publishers(topic) for topic in ["/clock", "/hardware/base/odom", "/NN_goal"]]
                    command_count = self.count_publishers("/hardware/base/cmd_vel")
                    if any(count > 1 for count in counts) or command_count > 2:
                        self.fail("Multiple simulator or goal publishers are present in the selected domain")
                    elif all(count == 1 for count in counts) and command_count == 2:
                        self.can_command = True
                        self.change("between")
                return
            if wall - self.packet_wall > 15:
                self.fail("Webots telemetry stopped or the simulation was paused")
                return
            if self.phase == "between":
                self.stop()
                if wall - self.phase_wall < 0.5:
                    return
                start = to_world(self.task["start"], cfg["suite"]["offsets"])
                message = Pose2D(x=float(start[0]), y=float(start[1]),
                                 theta=float(self.task["start_yaw_rad"] + cfg["suite"]["offsets"][2]))
                self.teleport_pub.publish(message)
                self.change("settle")
            elif self.phase == "settle":
                self.stop()
                protocol = cfg["suite"]["protocol"]
                target = to_world(self.task["start"], cfg["suite"]["offsets"])
                yaw = self.task["start_yaw_rad"] + cfg["suite"]["offsets"][2]
                error_yaw = abs(math.atan2(math.sin(self.packet["pose"][2] - yaw), math.cos(self.packet["pose"][2] - yaw)))
                settled = self.packet["time"] - self.phase_sim >= protocol["settle_s"]
                near = np.linalg.norm(np.asarray(self.packet["pose"][:2]) - target) <= protocol["start_tolerance_m"]
                if settled and near and error_yaw <= protocol["start_yaw_tolerance_rad"]:
                    if self.packet["collision_value"] > protocol["collision_threshold"]:
                        self.fail("The initial pose is already in contact")
                        return
                    if np.linalg.norm(self.packet["odom"][:2]) > 0.05 or abs(self.packet["odom"][2]) > 0.05:
                        self.fail("Odometry was not reset after teleport")
                        return
                    self.goal_valid = False
                    self.project_pub.publish(self.goal_message())
                    self.change("project")
                elif wall - self.phase_wall > 15:
                    self.fail("Teleport or pose settling failed")
            elif self.phase == "project":
                self.stop()
                if self.goal_valid and wall - self.phase_wall > 0.5:
                    self.change("preview" if cfg.get("rviz") else "quiet")
                elif wall - self.phase_wall > 10:
                    self.fail("The policy goal vector did not match the requested odometry goal")
            elif self.phase == "preview":
                self.stop()
                if wall - self.phase_wall >= cfg.get("preview_seconds", 5.0) and self.marker_pub.get_subscription_count():
                    self.change("quiet")
                elif wall - self.phase_wall > max(20.0, cfg.get("preview_seconds", 5.0) + 10):
                    self.fail("RViz did not subscribe to the evaluation markers")
            elif self.phase == "quiet" and wall - self.phase_wall >= 0.5:
                self.ack = self.reached = False
                self.start_pub.publish(self.goal_message())
                self.change("arming")
            elif self.phase == "arming":
                if self.ack:
                    self.metrics = TrialMetrics(self.task, cfg["suite"]["protocol"], cfg["suite"]["offsets"])
                    self.change("active")
                    self.record(self.packet)
                elif wall - self.phase_wall > 5:
                    self.fail("The policy did not acknowledge the new goal")
            elif self.phase == "active" and self.reached:
                if self.packet["time"] - self.phase_sim < 0.25:
                    self.fail("Premature goal status during activation")
                else:
                    self.finish("goal" if self.metrics.distance(self.packet) <= cfg["suite"]["protocol"]["goal_tolerance_m"] else "false_arrival")

    os.environ["ROS_DOMAIN_ID"] = str(cfg["domain_id"])
    rclpy.init(args=[])
    node = Evaluator()
    environment = dict(os.environ, ROS_DOMAIN_ID=str(cfg["domain_id"]))
    process = None
    result = None
    print(f"Starting an isolated ROS domain: {cfg['domain_id']}", flush=True)
    try:
        discovery_deadline = time.monotonic() + 2.0
        while time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if (any(node.count_publishers(topic) for topic in ["/clock", "/hardware/base/odom", "/NN_goal", "/evaluation/telemetry"])
                or node.count_publishers("/hardware/base/cmd_vel") > 1):
            node.fail("The selected ROS domain is already in use; choose another --domain-id")
        with (out / "simulation.log").open("w") as log:
            if node.phase != "done":
                process = subprocess.Popen(["ros2", "launch", str(launch_path)], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while rclpy.ok() and node.phase != "done":
                rclpy.spin_once(node, timeout_sec=0.05)
                node.tick()
                if process is not None and process.poll() is not None and node.phase != "done":
                    node.fail(f"The launch process exited with code {process.returncode}; inspect simulation.log")
            node.stop()
            node.trajectory.flush()
            result = save_session_result(node, cfg, out)
            if cfg.get("keep_open_after_trial") and node.can_command and process is not None and process.poll() is None:
                print(f"Result saved: {out / 'summary.json'}", flush=True)
                keep_open_for_review(node, process, rclpy)
    except KeyboardInterrupt:
        node.fail("Evaluation interrupted")
    except Exception as exc:
        node.fail("Evaluation runtime error: " + str(exc))
        raise
    finally:
        if rclpy.ok():
            node.stop()
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0.05)
        node.trajectory.close()
        node.input_health_log.close()
        if process is not None and process.poll() is None:
            for sig, timeout in [(signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)]:
                try:
                    os.killpg(process.pid, sig)
                    process.wait(timeout=timeout)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        if result is None:
            result = save_session_result(node, cfg, out)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--history-sha256")
    parser.add_argument("--scenario", default="E01")
    parser.add_argument("--purpose", choices=("development", "final"), default="development")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--route-class", choices=ROUTE_CLASSES)
    parser.add_argument("--ignore-dataset-hash", action="store_true", help="Skip the datasets_sha256 check between checkpoint and datasets.json (use with caution: manifests and catalog are still verified)")
    parser.add_argument("--rviz", action=argparse.BooleanOptionalAction, default=True, help="Launch RViz alongside Webots (default: enabled; use --no-rviz to disable)")
    parser.add_argument("--close-after-trial", action="store_true",
                        help="Close Webots and RViz automatically after the evaluation")
    parser.add_argument("--preview-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--heading", choices=("toward_goal", "generator"))
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--turn-help", choices=("true", "false", "from-launch"), default="from-launch",
                        help="Override turn_help from the launch file; recorded in the suite sources")
    parser.add_argument("--max-vx", type=float, default=None,
                        help="Override max_vx from the launch file; recorded in the suite sources")
    parser.add_argument("--max-vy", type=float, default=None,
                        help="Override max_vy from the launch file; recorded in the suite sources")
    parser.add_argument("--max-wz", type=float, default=None,
                        help="Override max_wz from the launch file; recorded in the suite sources")
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    out = None
    try:
        timeouts = [args.startup_timeout] + ([args.timeout] if args.timeout is not None else [])
        if args.limit < 1 or any(not math.isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError("Trial limit and timeouts must be finite and positive")
        if not math.isfinite(args.preview_seconds) or args.preview_seconds < 0:
            raise ValueError("Preview duration must be finite and nonnegative")
        if args.suite and args.seed is not None:
            raise ValueError("A reused suite keeps its saved seed; omit --seed")
        if not args.prepare_only and (shutil.which("ros2") is None or shutil.which("webots") is None):
            raise ValueError("ROS 2 and Webots must be available. Source install/setup.bash before evaluating navigation.")
        model = candidate(args.checkpoint, args.history_sha256)
        metadata = json.loads((args.datasets_dir.expanduser() / "datasets.json").read_text())
        if metadata.get("phase") != "datasets" or metadata.get("schema_version") != 1:
            raise ValueError("Unsupported datasets metadata")
        trained = model["training_config"].get("split", {})
        if (not getattr(args, "ignore_dataset_hash", False)
                and trained.get("datasets_sha256")
                and trained["datasets_sha256"] != control.file_hash(args.datasets_dir.expanduser() / "datasets.json")):
            raise ValueError("Checkpoint and prepared datasets do not match")
        training_set = trained.get("environment_set")
        scenarios = metadata["config"]["environment_sets"].get(training_set)
        model.update(environment_set=training_set, scenario_seen_in_training=args.scenario in scenarios if scenarios else None)
        seed = control.generated_seed(args.seed)
        domain = args.domain_id if args.domain_id is not None else random.SystemRandom().randrange(10, 101)
        if not 0 <= domain <= 101:
            raise ValueError("ROS domain ID must be between 0 and 101")
        out = control.new_output(args.output_dir, "evaluation")
        suite = prepare_suite(args, out, seed, metadata, model)
        config, launch = prepare_runtime(out, suite, model, args.limit, domain, args.route_class, args.rviz,
                                         args.preview_seconds, args.close_after_trial)
        print(f"Checkpoint: {model['path']}")
        print(f"Suite: {out / 'suite.json'}")
        print(f"Suite seed: {suite['seed']}")
        print("Input delays: diagnostic only; automatic abort disabled")
        for task in config["suite"]["tasks"][:config["limit"]]:
            print(f"Selected task: {task['id']} | {task['route_class']} | direct_line_safe={task['direct_line_safe']}")
        if config["rviz_config"]:
            print(f"RViz source: {config['rviz_config']['source']}")
        if config["keep_open_after_trial"]:
            print("After the trial: keep Webots and RViz open until Ctrl+C")
        print(f"Output: {out}", flush=True)
        if args.prepare_only:
            return 0
        result = run_session(config, launch, out, args.startup_timeout)
        print(json.dumps({k: result[k] for k in ["status", "completed_trials", "success_rate", "collision_rate", "infrastructure_errors"]}, indent=2))
        return 0 if result["status"] == "completed" else 2
    except (ValueError, KeyError, OSError, ImportError) as exc:
        if out is not None:
            control.write_json(out / "FAILED.json", {"error": str(exc)})
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Teleop sampler with start/goal pair workflow.

Modes:
    use_path_pool=True  (default) -> consume paths from path_pool.json.
        If pool is missing and auto_generate_pool=True, it is generated
        on demand via generate_path_pool.py using num_goals and class_mix.
    use_path_pool=False           -> original random sampling.

Buttons:
    B  -> toggle recording
    Y  -> discard current pair, generate new one
    X  -> emergency teleport (to current path start in pool mode,
          to unexplored room otherwise)
"""

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import rclpy
import rclpy.executors
from rclpy.callback_groups import (ReentrantCallbackGroup,
                                   MutuallyExclusiveCallbackGroup)
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import (PointStamped, Pose2D,
                               PoseWithCovarianceStamped, Twist)
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image

import atexit
import signal

from debugger.utils import DebugClient

try:
    from scipy.ndimage import distance_transform_edt, label
except ImportError as exc:
    raise RuntimeError('scipy is required for room detection') from exc


STATE_IDLE = 0
STATE_WAITING_FOR_START = 1
STATE_RECORDING = 2

CLASS_NAMES = ('straight', 'semi_complex', 'complex', 'v_complex')


def resolve_prohibition_png(map_name):
    maps_root = os.path.join(
        get_package_share_directory('justina_description'),
        'webots_env', 'environments', 'maps')
    if not os.path.isdir(maps_root):
        raise RuntimeError(f'maps root not found: {maps_root}')
    prefix = f'{map_name}_'
    candidates = sorted(
        name for name in os.listdir(maps_root)
        if name.startswith(prefix)
        and os.path.isdir(os.path.join(maps_root, name)))
    if not candidates:
        raise RuntimeError(
            f'no folder starting with {prefix} in {maps_root}')
    folder = candidates[0]
    path = os.path.join(
        maps_root, folder, f'map_{folder}_prohibition.png')
    if not os.path.isfile(path):
        raise RuntimeError(f'prohibition png not found: {path}')
    return path


def resolve_output_dir(output_dir):
    if not output_dir:
        output_dir = os.path.join(
            get_package_share_directory('mapless_dataset'), 'data')
    return os.path.abspath(output_dir)


def count_goals_in_csv(csv_path):
    total = 0
    try:
        with open(csv_path, 'r', newline='') as f:
            sample = f.readline()
            if not sample:
                return 0
            delimiter = '\t' if '\t' in sample else ','
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                return 0
            if 'is_episode_start' not in reader.fieldnames:
                return 0
            for row in reader:
                try:
                    if int(row.get('is_episode_start', '0')) == 1:
                        total += 1
                except (ValueError, TypeError):
                    continue
    except (OSError, UnicodeDecodeError):
        return 0
    return total


def count_goals_in_scenario(scenario_dir):
    if not os.path.isdir(scenario_dir):
        return 0
    total = 0
    try:
        entries = sorted(os.listdir(scenario_dir))
    except OSError:
        return 0
    for name in entries:
        run_dir = os.path.join(scenario_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not name.startswith('run_'):
            continue
        csv_path = os.path.join(run_dir, 'data.csv')
        if not os.path.isfile(csv_path):
            continue
        total += count_goals_in_csv(csv_path)
    return total


def read_world_offsets(png_path):
    yaml_path = os.path.splitext(png_path)[0] + '.yaml'
    if not os.path.isfile(yaml_path):
        return (0.0, 0.0, 0.0)
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return (
            float(data.get('world_offset_x', 0.0)),
            float(data.get('world_offset_y', 0.0)),
            float(data.get('world_offset_yaw', 0.0)),
        )
    except Exception:
        return (0.0, 0.0, 0.0)


ORPHAN_PATTERNS = [
    'webots',
    'justina_controller',
    'dataset_recorder',
    'teleop_dataset_recorder',
    'dataset_obstacles_control.py',
    'rviz2',
    'nav2_amcl',
    'nav2_map_server',
    'nav2_lifecycle_manager',
    'joy_node',
    'robot_teleop.py',
    'local_occ_grid_node',
    'get_goal_point',
    'prohibition_map_publisher',
    'depth_to_pointcloud',
]


def _signal_to_pids(pattern):
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True, text=True, timeout=2.0)
    except Exception:
        return []
    pids = []
    for line in result.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        pids.append(pid)
    return pids


def _kill_pattern(pattern, sig):
    for pid in _signal_to_pids(pattern):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _kill_orphans():
    print('[teleop] cleanup: sending SIGTERM to orphan processes')
    for pat in ORPHAN_PATTERNS:
        _kill_pattern(pat, signal.SIGTERM)
    time.sleep(1.5)
    print('[teleop] cleanup: sending SIGKILL to survivors')
    for pat in ORPHAN_PATTERNS:
        _kill_pattern(pat, signal.SIGKILL)
    leftover = 0
    for pat in ORPHAN_PATTERNS:
        leftover += len(_signal_to_pids(pat))
    if leftover > 0:
        print(f'[teleop] cleanup: warning, {leftover} process(es) alive')
    else:
        print('[teleop] cleanup: done, no orphans left')


def _on_signal(signum, frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(0)


class TeleopDatasetRecorder(Node):

    def __init__(self):
        super().__init__('teleop_dataset_recorder')
        self.declare_node_parameters()
        self.load_node_parameters()

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.dbg = DebugClient(self, verbose=True)
        self.dbg.send('Node initialized and ready.',
                      'NODE_STATUS', 'info', -1.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_goal = self.create_publisher(
            PointStamped, '/mapless_nav/current_goal', 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/mapless_nav/goal_markers', 10)
        self.pub_active = self.create_publisher(
            Bool, '/mapless_nav/goal_active', 10)
        self.pub_episode = self.create_publisher(
            Int32, '/mapless_nav/episode_id', 10)
        self.pub_episode_class = self.create_publisher(
            String, '/mapless_nav/episode_class', 10)
        self.pub_episode_tries = self.create_publisher(
            Int32, '/mapless_nav/episode_tries', 10)
        self.pub_success = self.create_publisher(
            Bool, '/mapless_nav/goal_success', 10)
        self.teleport_pub = self.create_publisher(
            Pose2D, '/webots/teleport', 10)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/hardware/base/cmd_vel', 10)
        self.paused_pub = self.create_publisher(
            Bool, '/teleop/paused', 10)
        self.paused_pub.publish(Bool(data=False))
        self.pub_head_pan = self.create_publisher(
            Float64, '/head/pan_cmd', 10)
        self.pub_head_tilt = self.create_publisher(
            Float64, '/head/tilt_cmd', 10)
        self.pub_head_pose = self.create_publisher(
            Float64MultiArray, '/hardware/head/goal_pose', 10)

        map_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.sub_group = ReentrantCallbackGroup()
        self.heartbeat_group = MutuallyExclusiveCallbackGroup()
        self.collision_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, map_qos,
            callback_group=self.sub_group)
        self.create_subscription(
            Joy, self.joy_topic, self.on_joy, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            Bool, self.collision_topic, self.on_collision, 10,
            callback_group=self.collision_group)
        self.create_subscription(
            Bool, '/webots/teleport_done', self.on_teleport_done, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self.on_amcl_pose, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            Twist, '/hardware/base/cmd_vel', self.on_cmd_vel, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            Float64, '/teleop/head/pan_cmd_raw',
            self.on_head_pan_raw, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            Float64, '/teleop/head/tilt_cmd_raw',
            self.on_head_tilt_raw, 10,
            callback_group=self.sub_group)
        self.create_subscription(
            Float64MultiArray, '/teleop/head/goal_pose_raw',
            self.on_head_pose_raw, 10,
            callback_group=self.sub_group)

        self.map_msg = None
        self.teleport_done = False
        self.last_amcl = None
        self.last_active_time = time.time()
        self.joy_seen = False

        self.room_map = None
        self.room_res = None
        self.room_origin = (0.0, 0.0)
        self.room_cells = {}
        self.valid_rooms = set()
        self.free_cells = []
        self.rooms_ready = False
        self.dist_px = None

        self.recent_goals = deque(maxlen=32)
        self.recent_goal_rooms = deque(maxlen=self.recent_room_window)
        self.recent_zones = deque(maxlen=self.recent_zone_window)

        self.recording = False
        self.state = STATE_IDLE
        self.episode_id = int(time.time()) - 1
        self.current_start = None
        self.current_goal = None
        self.goal_start_time = 0.0

        self.flag_collision = False
        self.flag_button_b = False
        self.flag_button_x = False
        self.flag_button_y = False

        self.collision_now = False
        self._collision_prev = False
        self._collision_since = None
        self.episode_had_collision = False

        self.teleport_in_progress = False
        self._teleport_thread = None

        self._prev_buttons = {}

        self.goals_saved_total = 0
        self.goals_saved_session = 0

        # Path pool state
        self.path_pool_by_class = {name: [] for name in CLASS_NAMES}
        self.class_quotas = {name: 0 for name in CLASS_NAMES}
        self.class_counts = {name: 0 for name in CLASS_NAMES}
        self.current_path = None
        self.current_class = ''
        self.path_attempts = 0
        self.failed_paths = []
        self.path_pool_loaded = False
        self._generating_pool = False
        self._pool_gen_failed = False
        self._pool_gen_error = ''
        self._session_complete = False
        self._pool_deficits = {name: 0 for name in CLASS_NAMES}
        self._class_lock = threading.Lock()

        self.scenario_dir = os.path.join(self.output_dir, self.scenario)
        self.load_goal_counter()
        self.goals_saved_total_at_start = self.goals_saved_total
        self.last_saved_count = self.goals_saved_total

        try:
            png = resolve_prohibition_png(self.map_name)
        except RuntimeError as exc:
            self.dbg.send(f'FATAL: {exc}', 'NODE_STATUS', 'err', -1.0)
            raise

        self.get_logger().info(f'prohibition png: {png}')
        img = Image.open(png).convert('L')
        self.prohib_arr = np.array(img, dtype=np.uint8) > 127

        if self.auto_world_offset:
            ox, oy, oyaw = read_world_offsets(png)
            self.off_x = ox
            self.off_y = oy
            self.off_yaw = oyaw
            self.get_logger().info(
                f'world offset from yaml: '
                f'x={self.off_x:.3f} y={self.off_y:.3f} '
                f'yaw={self.off_yaw:.3f}')
        else:
            self.get_logger().info(
                f'world offset from params: '
                f'x={self.off_x:.3f} y={self.off_y:.3f} '
                f'yaw={self.off_yaw:.3f}')

        self.dbg.send(
            f'total_saved={self.goals_saved_total} current_saved=0',
            'RECORDED', 'info', -1.0)

        # Try to load existing pool.
        if self.use_path_pool:
            pool_path = os.path.join(
                self.output_dir, self.scenario, 'path_pool.json')
            poses_path = os.path.join(
                self.output_dir, self.scenario, 'path_pool_poses.json')
            if os.path.isfile(pool_path) and os.path.isfile(poses_path):
                if self._load_path_pool():
                    if self._pool_ready():
                        self.get_logger().info(
                            'Existing path pool loaded with enough '
                            'fresh paths.')
                    else:
                        self.get_logger().warn(
                            f'Existing pool has deficits: '
                            f'{self._pool_deficits}. Will regenerate '
                            f'on first B press.')
            elif self.auto_generate_pool:
                self.get_logger().warn(
                    f'No pool found at {self.scenario_dir}. '
                    f'It will be generated when recording starts '
                    f'(num_goals={self.num_goals}, '
                    f'class_mix="{self.class_mix_str or "balanced"}").')
            else:
                self.get_logger().error(
                    'use_path_pool=true but no pool and '
                    'auto_generate_pool=false.')

        threading.Thread(
            target=self.build_rooms_worker, daemon=True).start()

        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, self.heartbeat,
                          callback_group=self.heartbeat_group)

        self.get_logger().info(
            f'ready. map={self.map_name} seed={self.seed} '
            f'use_path_pool={self.use_path_pool} '
            f'num_goals={self.num_goals} '
            f'class_mix="{self.class_mix_str}" '
            f'auto_generate_pool={self.auto_generate_pool} '
            f'buttons B={self.record_button_index} '
            f'X={self.teleport_button_index} '
            f'Y={self.regenerate_button_index}')

    #  Parameters

    def declare_node_parameters(self):
        self.declare_parameter('map', 'E01')
        self.declare_parameter('scenario', 'E01')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('seed', 1001)
        self.declare_parameter('count_existing_goals', True)

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('collision_topic', '/navigation/collision')
        self.declare_parameter('joy_topic', '/hardware/joy')
        self.declare_parameter('map_wait_timeout_s', 15.0)

        self.declare_parameter('room_radius_m', 0.45)
        self.declare_parameter('min_room_area_m2', 0.5)
        self.declare_parameter('zone_bin_m', 0.8)
        self.declare_parameter('recent_zone_window', 8)
        self.declare_parameter('recent_room_window', 3)

        self.declare_parameter('min_goal_distance_m', 1.5)
        self.declare_parameter('max_goal_distance_m', 4.0)

        self.declare_parameter('start_at_robot_prob', 0.3)
        self.declare_parameter('start_reached_radius_m', 0.5)
        self.declare_parameter('goal_reached_radius_m', 0.30)
        self.declare_parameter('goal_timeout_s', 180.0)
        self.declare_parameter('goal_pause_s', 0.5)

        self.declare_parameter('idle_timeout_s', 6.0)
        self.declare_parameter('idle_eps_lin', 0.02)
        self.declare_parameter('idle_eps_ang', 0.05)

        self.declare_parameter('record_button_index', 1)
        self.declare_parameter('teleport_button_index', 2)
        self.declare_parameter('regenerate_button_index', 3)
        self.declare_parameter('button_debounce_s', 0.3)

        self.declare_parameter('teleport_at_start', True)
        self.declare_parameter('startup_delay_s', 5.0)
        self.declare_parameter('world_offset_x', 0.0)
        self.declare_parameter('world_offset_y', 0.0)
        self.declare_parameter('world_offset_yaw', 0.0)
        self.declare_parameter('teleport_timeout_s', 4.0)
        self.declare_parameter('post_teleport_settle_s', 1.5)
        self.declare_parameter('post_teleport_amcl_grace_s', 3.0)
        self.declare_parameter('amcl_stable_timeout_s', 40.0)
        self.declare_parameter('amcl_stable_window', 15)
        self.declare_parameter('amcl_stable_tolerance_m', 0.05)

        self.declare_parameter('head_pan_offset', 0.0)
        self.declare_parameter('head_tilt_offset', -0.5)

        self.declare_parameter('collision_stuck_s', 2.0)
        self.declare_parameter('auto_world_offset', True)

        # Path pool mode
        self.declare_parameter('use_path_pool', True)
        self.declare_parameter('num_goals', 100)
        self.declare_parameter('class_mix', '')
        self.declare_parameter('path_max_attempts', 3)

        # Auto-generate pool
        self.declare_parameter('auto_generate_pool', True)
        self.declare_parameter('generator_script', '')
        self.declare_parameter('generator_maps_root', '')
        self.declare_parameter('generator_seed', -1)
        self.declare_parameter('generator_timeout_s', 900.0)

    def load_node_parameters(self):
        self.map_name = self.get_parameter('map').value
        self.scenario = self.get_parameter('scenario').value
        self.output_dir = resolve_output_dir(
            self.get_parameter('output_dir').value)
        self.seed = self.get_parameter('seed').value
        self.count_existing_goals = self.get_parameter(
            'count_existing_goals').value

        self.map_topic = self.get_parameter('map_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.collision_topic = self.get_parameter('collision_topic').value
        self.joy_topic = self.get_parameter('joy_topic').value
        self.map_wait_timeout_s = self.get_parameter(
            'map_wait_timeout_s').value

        self.room_radius_m = self.get_parameter('room_radius_m').value
        self.min_room_area_m2 = self.get_parameter('min_room_area_m2').value
        self.zone_bin_m = self.get_parameter('zone_bin_m').value
        self.recent_zone_window = self.get_parameter(
            'recent_zone_window').value
        self.recent_room_window = self.get_parameter(
            'recent_room_window').value

        self.min_dist = self.get_parameter('min_goal_distance_m').value
        self.max_dist = self.get_parameter('max_goal_distance_m').value

        self.start_at_robot_prob = self.get_parameter(
            'start_at_robot_prob').value
        self.start_reached_radius_m = self.get_parameter(
            'start_reached_radius_m').value
        self.goal_reached_radius_m = self.get_parameter(
            'goal_reached_radius_m').value
        self.goal_timeout_s = self.get_parameter('goal_timeout_s').value
        self.goal_pause_s = self.get_parameter('goal_pause_s').value

        self.idle_timeout_s = self.get_parameter('idle_timeout_s').value
        self.idle_eps_lin = self.get_parameter('idle_eps_lin').value
        self.idle_eps_ang = self.get_parameter('idle_eps_ang').value

        self.record_button_index = self.get_parameter(
            'record_button_index').value
        self.teleport_button_index = self.get_parameter(
            'teleport_button_index').value
        self.regenerate_button_index = self.get_parameter(
            'regenerate_button_index').value
        self.button_debounce_s = self.get_parameter(
            'button_debounce_s').value

        self.teleport_at_start = self.get_parameter(
            'teleport_at_start').value
        self.startup_delay_s = self.get_parameter('startup_delay_s').value
        self.off_x = self.get_parameter('world_offset_x').value
        self.off_y = self.get_parameter('world_offset_y').value
        self.off_yaw = self.get_parameter('world_offset_yaw').value
        self.teleport_timeout_s = self.get_parameter(
            'teleport_timeout_s').value
        self.post_teleport_settle_s = self.get_parameter(
            'post_teleport_settle_s').value
        self.post_teleport_amcl_grace_s = self.get_parameter(
            'post_teleport_amcl_grace_s').value
        self.amcl_stable_timeout_s = self.get_parameter(
            'amcl_stable_timeout_s').value
        self.amcl_stable_window = self.get_parameter(
            'amcl_stable_window').value
        self.amcl_stable_tolerance_m = self.get_parameter(
            'amcl_stable_tolerance_m').value

        self.head_pan_offset = self.get_parameter(
            'head_pan_offset').value
        self.head_tilt_offset = self.get_parameter(
            'head_tilt_offset').value

        self.collision_stuck_s = self.get_parameter(
            'collision_stuck_s').value
        self.auto_world_offset = self.get_parameter(
            'auto_world_offset').value

        self.use_path_pool = bool(self.get_parameter('use_path_pool').value)
        self.num_goals = int(self.get_parameter('num_goals').value)
        self.class_mix_str = str(
            self.get_parameter('class_mix').value).strip()
        self.path_max_attempts = int(
            self.get_parameter('path_max_attempts').value)
        if self.path_max_attempts < 1:
            self.path_max_attempts = 1

        self.auto_generate_pool = bool(
            self.get_parameter('auto_generate_pool').value)

        self.generator_script = str(
            self.get_parameter('generator_script').value).strip()
        if not self.generator_script:
            self.generator_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'generate_path_pool.py')

        self.generator_maps_root = str(
            self.get_parameter('generator_maps_root').value).strip()
        if not self.generator_maps_root:
            self.generator_maps_root = os.path.join(
                get_package_share_directory('justina_description'),
                'webots_env', 'environments', 'maps')

        self.generator_seed = int(self.get_parameter('generator_seed').value)
        self.generator_timeout_s = float(
            self.get_parameter('generator_timeout_s').value)

    #  Pool: parsing and loading

    def _parse_class_mix(self, mix_str):
        """Return dict class_name -> quota, or None if mix_str empty.

        Formats:
          - "10_10_15_20" or "10,10,15,20": integer quotas.
            If sum != num_goals, scaled proportionally.
          - "0.4,0.3,0.2,0.1": proportions.
          - "": returns None.
        """
        s = mix_str.strip()
        if not s:
            return None
        parts = [p.strip() for p in (s.split(',') if ',' in s else s.split('_'))]
        if len(parts) != 4:
            raise ValueError(
                f'class_mix needs 4 values, got {len(parts)}: {mix_str}')
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f'class_mix values must be numeric: {mix_str}')
        if any(v < 0 for v in values):
            raise ValueError('class_mix values must be >= 0')
        total = sum(values)
        if total <= 0:
            raise ValueError('class_mix sum must be > 0')

        all_int = all(abs(v - round(v)) < 1e-9 for v in values)
        if all_int and int(round(total)) == self.num_goals:
            return {k: int(round(v))
                    for k, v in zip(CLASS_NAMES, values)}

        quotas = {}
        allocated = 0
        for i, v in enumerate(values[:-1]):
            q = int(round(self.num_goals * v / total))
            quotas[CLASS_NAMES[i]] = q
            allocated += q
        quotas[CLASS_NAMES[-1]] = max(0, self.num_goals - allocated)
        return quotas

    def _resolve_quotas_for_generation(self):
        """Integer per-class quotas for passing to generate_path_pool.py."""
        parsed = self._parse_class_mix(self.class_mix_str)
        if parsed is not None:
            return parsed
        base, rem = divmod(self.num_goals, len(CLASS_NAMES))
        return {k: base + (1 if i < rem else 0)
                for i, k in enumerate(CLASS_NAMES)}

    def _load_used_endpoints(self):
        """Read episode_metadata.jsonl and failed_paths.json to collect
        start/goal endpoint pairs that were already used in prior sessions.
        Uses endpoint proximity instead of path_id because a regenerated
        pool restarts IDs from 0 with different coordinates."""
        used = []
        meta_path = os.path.join(
            self.output_dir, self.scenario, 'episode_metadata.jsonl')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            s = e.get('start')
                            g = e.get('goal')
                            if (s and g and len(s) >= 2 and len(g) >= 2):
                                used.append((
                                    (float(s[0]), float(s[1])),
                                    (float(g[0]), float(g[1]))))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
            except OSError:
                pass

        failed_path = os.path.join(
            self.output_dir, self.scenario, 'failed_paths.json')
        if os.path.isfile(failed_path):
            try:
                with open(failed_path, 'r') as f:
                    data = json.load(f)
                for e in data.get('entries', []):
                    s = e.get('start')
                    g = e.get('goal')
                    if (s and g and len(s) >= 2 and len(g) >= 2):
                        used.append((
                            (float(s[0]), float(s[1])),
                            (float(g[0]), float(g[1]))))
            except (OSError, json.JSONDecodeError):
                pass

        return used

    def _endpoint_matches(self, path, used, tol=0.30):
        """True if path start/goal matches any used endpoint pair
        within tolerance (meters)."""
        ps = (path['start'][0], path['start'][1])
        pg = (path['goal'][0], path['goal'][1])
        for us, ug in used:
            if (math.hypot(ps[0] - us[0], ps[1] - us[1]) < tol
                    and math.hypot(pg[0] - ug[0], pg[1] - ug[1]) < tol):
                return True
        return False

    def _pool_ready(self):
        """True when pool is loaded and has enough fresh paths for quotas."""
        if not self.path_pool_loaded:
            return False
        return sum(self._pool_deficits.values()) == 0

    def _load_path_pool(self):
        pool_path = os.path.join(
            self.output_dir, self.scenario, 'path_pool.json')
        poses_path = os.path.join(
            self.output_dir, self.scenario, 'path_pool_poses.json')
        if not (os.path.isfile(pool_path) and os.path.isfile(poses_path)):
            self.get_logger().error(
                f'Pool files missing: pool={os.path.isfile(pool_path)} '
                f'poses={os.path.isfile(poses_path)}')
            return False

        try:
            with open(pool_path, 'r') as f:
                pool = json.load(f)
            with open(poses_path, 'r') as f:
                poses_by_id = json.load(f)
        except Exception as exc:
            self.get_logger().error(f'Failed to read pool: {exc}')
            return False

        scenario_in_pool = str(pool.get('scenario', ''))
        if scenario_in_pool and scenario_in_pool != self.scenario:
            self.get_logger().error(
                f'Pool scenario mismatch: expected {self.scenario}, '
                f'got {scenario_in_pool}')
            return False

        try:
            parsed = self._parse_class_mix(self.class_mix_str)
        except ValueError as exc:
            self.get_logger().error(f'Invalid class_mix: {exc}')
            return False

        if parsed is not None:
            quotas = parsed
        else:
            pool_quota = pool.get('quota', {})
            try:
                pq = {k: int(pool_quota.get(k, 0)) for k in CLASS_NAMES}
            except (TypeError, ValueError):
                pq = {}
            if pq and sum(pq.values()) == self.num_goals:
                quotas = pq
            else:
                base, rem = divmod(self.num_goals, len(CLASS_NAMES))
                quotas = {k: base + (1 if i < rem else 0)
                          for i, k in enumerate(CLASS_NAMES)}

        self.path_pool_by_class = {name: [] for name in CLASS_NAMES}
        skipped = 0

        for entry in pool.get('paths', []):
            cls = entry.get('class')
            if cls not in self.path_pool_by_class:
                skipped += 1
                continue
            try:
                pid = int(entry['id'])
                start = [float(entry['start'][0]),
                         float(entry['start'][1])]
                goal = [float(entry['goal'][0]),
                        float(entry['goal'][1])]
                points = poses_by_id.get(str(pid))
                if not points or len(points) < 2:
                    skipped += 1
                    continue
            except (KeyError, ValueError, TypeError, IndexError):
                skipped += 1
                continue
            self.path_pool_by_class[cls].append({
                'id': pid,
                'class': cls,
                'start': start,
                'goal': goal,
            })

        for cls in CLASS_NAMES:
            random.shuffle(self.path_pool_by_class[cls])

        with self._class_lock:
            self.class_quotas = quotas

        loaded = {cls: len(self.path_pool_by_class[cls])
                  for cls in CLASS_NAMES}
        total_loaded = sum(loaded.values())
        if total_loaded == 0:
            self.get_logger().error(
                f'Pool contained zero usable paths (skipped={skipped}).')
            return False

        # Filter out endpoints already recorded in previous sessions.
        used = self._load_used_endpoints()
        filtered_counts = {cls: 0 for cls in CLASS_NAMES}
        if used:
            for cls in CLASS_NAMES:
                before = len(self.path_pool_by_class[cls])
                self.path_pool_by_class[cls] = [
                    p for p in self.path_pool_by_class[cls]
                    if not self._endpoint_matches(p, used)
                ]
                filtered_counts[cls] = before - len(self.path_pool_by_class[cls])
            self.get_logger().info(
                f'Filtered {sum(filtered_counts.values())} paths whose '
                f'endpoints match prior sessions: {filtered_counts} '
                f'(total used endpoints={len(used)})')

        fresh = {cls: len(self.path_pool_by_class[cls])
                 for cls in CLASS_NAMES}
        deficits = {
            cls: max(0, self.class_quotas[cls] - fresh[cls])
            for cls in CLASS_NAMES
        }
        self._pool_deficits = deficits

        self.get_logger().info(
            f'Path pool loaded: scenario={self.scenario} '
            f'pool_version={pool.get("pool_version")} '
            f'usable_paths={total_loaded} skipped={skipped} '
            f'per_class_raw={loaded} per_class_fresh={fresh}')
        self.get_logger().info(
            f'Class quotas: {self.class_quotas} '
            f'(sum={sum(self.class_quotas.values())}, '
            f'num_goals={self.num_goals})')

        if sum(deficits.values()) > 0:
            self.get_logger().warn(
                f'Pool has insufficient fresh paths for quotas: '
                f'{deficits}')

        self.path_pool_loaded = True
        return True

    def _ensure_path_pool(self):
        # Already loaded and has enough fresh paths.
        if self._pool_ready():
            return True

        pool_path = os.path.join(
            self.output_dir, self.scenario, 'path_pool.json')
        poses_path = os.path.join(
            self.output_dir, self.scenario, 'path_pool_poses.json')
        pool_exists = (
            os.path.isfile(pool_path)
            and os.path.isfile(poses_path))

        # Existing pool: try to load.
        if pool_exists and not self.path_pool_loaded:
            if not self._load_path_pool():
                return False

        # If still not ready after loading, we need to regenerate.
        if self.path_pool_loaded and self._pool_ready():
            return True

        if not self.auto_generate_pool:
            with self._class_lock:
                deficits = dict(self._pool_deficits)
            self.get_logger().error(
                f'Pool unavailable or insufficient fresh paths '
                f'(deficits={deficits}) and auto_generate_pool=False. '
                f'Regenerate manually with a larger --st_sem_cmp_vc or '
                f'delete {pool_path}.')
            return False

        # Compute regeneration quotas: base quota + deficit + 1 buffer.
        with self._class_lock:
            base = dict(self.class_quotas)
            deficits = dict(self._pool_deficits)
        if not base or sum(base.values()) == 0:
            base = {cls: 1 for cls in CLASS_NAMES}
        new_quotas = {
            cls: max(1, base.get(cls, 1) + deficits.get(cls, 0) + 1)
            for cls in CLASS_NAMES
        }
        self.get_logger().warn(
            f'Regenerating pool with fresh paths: '
            f'deficits={deficits} -> new_quotas={new_quotas}')

        # Force reload after generation overwrites the flat pool.
        self.path_pool_loaded = False
        return self._generate_pool_subprocess(new_quotas)

    def _generate_pool_subprocess(self, extra_quotas=None):
        gen = self.generator_script
        if not os.path.isfile(gen):
            self.get_logger().error(
                f'generator_script not found: {gen}')
            return False
        maps_root = self.generator_maps_root
        if not os.path.isdir(maps_root):
            self.get_logger().error(
                f'generator_maps_root not found: {maps_root}')
            return False

        try:
            if extra_quotas is not None:
                quotas = {k: int(extra_quotas[k]) for k in CLASS_NAMES}
            else:
                quotas = self._resolve_quotas_for_generation()
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f'quota resolution failed: {exc}')
            return False

        seed = (self.generator_seed if self.generator_seed >= 0
                else random.randint(100000, 999999))
        mix_arg = '_'.join(str(quotas[k]) for k in CLASS_NAMES)
        run_name = f'generation_seed_{seed}'

        cmd = [
            sys.executable, gen,
            '--scenario', self.scenario,
            '--maps-root', maps_root,
            '--output-dir', self.output_dir,
            '--seed', str(seed),
            '--st_sem_cmp_vc', mix_arg,
            '--progress-interval-seconds', '5',
        ]

        self.dbg.send(
            f'generating pool: scenario={self.scenario} seed={seed} '
            f'mix={mix_arg}',
            'STATUS', 'warn', -1.0)
        self.get_logger().warn(
            f'Generating path pool: scenario={self.scenario} '
            f'seed={seed} mix={mix_arg}. This may take several minutes.')

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.generator_timeout_s)
        except subprocess.TimeoutExpired:
            self.get_logger().error(
                f'Pool generation TIMEOUT after '
                f'{self.generator_timeout_s:.0f}s')
            return False
        except Exception as exc:
            self.get_logger().error(
                f'Pool generation failed to launch: {exc}')
            return False

        elapsed = time.time() - t0
        if result.returncode != 0:
            self.get_logger().error(
                f'Pool generation exit={result.returncode} '
                f'elapsed={elapsed:.1f}s')
            tail = '\n'.join(
                (result.stderr or '').splitlines()[-15:])
            if tail:
                self.get_logger().error(
                    f'generator stderr tail:\n{tail}')
            return False

        run_dir = os.path.join(
            self.output_dir, self.scenario, run_name)
        gen_pool = os.path.join(run_dir, 'path_pool.json')
        gen_poses = os.path.join(run_dir, 'path_pool_poses.json')
        if not (os.path.isfile(gen_pool) and os.path.isfile(gen_poses)):
            self.get_logger().error(
                f'Generator returned 0 but outputs missing in {run_dir}')
            return False

        flat_pool = os.path.join(
            self.output_dir, self.scenario, 'path_pool.json')
        flat_poses = os.path.join(
            self.output_dir, self.scenario, 'path_pool_poses.json')
        try:
            os.makedirs(
                os.path.dirname(flat_pool), exist_ok=True)
            shutil.copy2(gen_pool, flat_pool)
            shutil.copy2(gen_poses, flat_poses)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to install pool files: {exc}')
            return False

        self.get_logger().info(
            f'Pool generated and installed after {elapsed:.1f}s '
            f'(run_dir={run_dir})')
        return self._load_path_pool()

    def _next_target_class(self):
        with self._class_lock:
            candidates = []
            for name in CLASS_NAMES:
                if self.class_counts[name] >= self.class_quotas[name]:
                    continue
                if not self.path_pool_by_class.get(name):
                    continue
                candidates.append(name)
            if not candidates:
                return None
            lowest = min(self.class_counts[n] for n in candidates)
            tied = [n for n in candidates
                    if self.class_counts[n] == lowest]
            return random.choice(tied)

    def _select_next_pool_path(self):
        target = self._next_target_class()
        if target is None:
            return None
        with self._class_lock:
            paths = self.path_pool_by_class.get(target, [])
            if not paths:
                return None
            return paths.pop()

    def _should_stop_recording(self):
        """True when all quotas are met OR no class still has both
        remaining quota and remaining pool paths."""
        with self._class_lock:
            for name in CLASS_NAMES:
                if self.class_counts[name] < self.class_quotas[name]:
                    if self.path_pool_by_class.get(name):
                        return False
        return True

    #  Subscriptions

    def on_map(self, msg):
        self.map_msg = msg

    def on_collision(self, msg):
        now = bool(msg.data)
        if now:
            if self._collision_since is None:
                self._collision_since = time.time()
            if not self._collision_prev and not self.teleport_in_progress:
                self.flag_collision = True
            if self.state == STATE_RECORDING:
                self.episode_had_collision = True
        else:
            self._collision_since = None
        self._collision_prev = now
        self.collision_now = now

    def on_teleport_done(self, msg):
        self.teleport_done = bool(msg.data)

    def on_amcl_pose(self, msg):
        self.last_amcl = msg

    def on_cmd_vel(self, msg):
        active = (abs(msg.linear.x) > self.idle_eps_lin
                  or abs(msg.linear.y) > self.idle_eps_lin
                  or abs(msg.angular.z) > self.idle_eps_ang)
        if active:
            self.last_active_time = time.time()

    def on_head_pan_raw(self, msg):
        out = Float64()
        out.data = float(msg.data) + self.head_pan_offset
        self.pub_head_pan.publish(out)

    def on_head_tilt_raw(self, msg):
        out = Float64()
        out.data = float(msg.data) + self.head_tilt_offset
        self.pub_head_tilt.publish(out)

    def on_head_pose_raw(self, msg):
        if len(msg.data) < 2:
            return
        out = Float64MultiArray()
        out.data = [
            float(msg.data[0]) + self.head_pan_offset,
            float(msg.data[1]) + self.head_tilt_offset,
        ]
        self.pub_head_pose.publish(out)

    def on_joy(self, msg):
        self.joy_seen = True

        def edge(idx, key):
            if idx >= len(msg.buttons):
                return False
            cur = msg.buttons[idx] == 1
            prev = self._prev_buttons.get(key, False)
            self._prev_buttons[key] = cur
            return cur and not prev

        if edge(self.record_button_index, 'b'):
            self.flag_button_b = True
        if edge(self.teleport_button_index, 'x'):
            self.flag_button_x = True
        if edge(self.regenerate_button_index, 'y'):
            self.flag_button_y = True

    #  Room building

    def build_rooms_worker(self):
        while self.map_msg is None and rclpy.ok():
            self.dbg.send('Waiting for /map ...',
                          'NODE_STATUS', 'info', -1.0)
            t0 = time.time()
            while (self.map_msg is None
                   and time.time() - t0 < self.map_wait_timeout_s
                   and rclpy.ok()):
                time.sleep(0.1)

        if self.map_msg is None:
            self.dbg.send('FATAL: /map not received',
                          'NODE_STATUS', 'err', -1.0)
            return

        while rclpy.ok():
            try:
                self.build_rooms()
                if self.free_cells:
                    break
                self.get_logger().warn(
                    'no free cells, retrying room build in 3 s')
                time.sleep(3.0)
            except Exception as exc:
                self.get_logger().error(f'room building failed: {exc}')
                self.dbg.send(f'FATAL: room build failed: {exc}',
                              'NODE_STATUS', 'err', -1.0)
                return

        self.get_logger().info(
            f'rooms built, waiting {self.startup_delay_s:.1f}s')
        time.sleep(self.startup_delay_s)

        if self.teleport_at_start:
            spawn = self.sample_spawn_unexplored()
            if spawn is not None:
                ok = self.do_teleport_and_settle(spawn)
                self.get_logger().info(f'initial teleport success={ok}')
            else:
                self.get_logger().warn('no spawn sampled')
        else:
            self.wait_for_amcl_stable()

        self.rooms_ready = True
        self.dbg.send('Node initialized and ready.',
                      'NODE_STATUS', 'info', -1.0)
        self.get_logger().info('rooms ready')

    def build_rooms(self):
        info = self.map_msg.info
        w, h = info.width, info.height
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        data = np.array(self.map_msg.data, dtype=np.int8).reshape(h, w)

        free = (data == 0)

        mask_img = Image.fromarray(
            (self.prohib_arr * 255).astype(np.uint8))
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        free = free & (np.array(mask_img) > 127)

        dist_px = distance_transform_edt(free)
        self.dist_px = dist_px
        min_dist_px = self.room_radius_m / res
        traversable = dist_px >= min_dist_px

        labels, n = label(traversable, structure=np.ones((3, 3)))
        min_cells = max(1, int(self.min_room_area_m2 / (res * res)))
        sizes = np.bincount(labels.ravel())

        valid = set()
        for i in range(1, n + 1):
            if sizes[i] >= min_cells:
                valid.add(i)

        self.room_map = labels
        self.room_res = res
        self.room_origin = (ox, oy)
        self.valid_rooms = valid
        self.room_cells = {}

        ys, xs = np.where(traversable)
        for row, col in zip(ys, xs):
            rid = int(labels[row, col])
            if rid not in valid:
                continue
            mx = ox + (col + 0.5) * res
            my = oy + (h - 1 - row + 0.5) * res
            self.room_cells.setdefault(rid, []).append((mx, my))

        self.free_cells = []
        for rid in valid:
            self.free_cells.extend(self.room_cells[rid])

    #  Counters / debug

    def load_goal_counter(self):
        if not self.count_existing_goals:
            self.goals_saved_total = 0
            return
        try:
            self.goals_saved_total = count_goals_in_scenario(
                self.scenario_dir)
        except Exception as exc:
            self.get_logger().warn(
                f'could not count existing goals: {exc}')
            self.goals_saved_total = 0
        self.get_logger().info(
            f'loaded goal counter: total={self.goals_saved_total} '
            f'from {self.scenario_dir}')

    def update_info_channel(self):
        self.dbg.send(
            f'goals_saved={self.goals_saved_session}',
            'INFO', 'info', -1.0)
        self.dbg.send(
            f'total_saved={self.goals_saved_total} '
            f'current_saved={self.goals_saved_session}',
            'RECORDED', 'info', -1.0)
        if self.use_path_pool:
            with self._class_lock:
                counts = dict(self.class_counts)
                quotas = dict(self.class_quotas)
            self.dbg.send(
                ' '.join(f'{n}={counts[n]}/{quotas[n]}'
                         for n in CLASS_NAMES),
                'CLASSES', 'info', -1.0)

    def heartbeat(self):
        self.dbg.send('Node initialized and ready.',
                      'NODE_STATUS', 'info', -1.0)
        if self._generating_pool:
            self.dbg.send('Generating pool', 'STATUS', 'warn', -1.0)
        elif self.teleport_in_progress:
            self.dbg.send('Teleporting', 'STATUS', 'warn', -1.0)
        elif not self.recording:
            self.dbg.send('Idle', 'STATUS', 'info', -1.0)
        elif self.state == STATE_WAITING_FOR_START:
            self.dbg.send('Waiting for start', 'STATUS', 'warn', -1.0)
        else:
            self.dbg.send('Recording', 'STATUS', 'err', -1.0)
        if not self.joy_seen:
            self.dbg.send('joystick not seen yet',
                          'STATUS', 'warn', -1.0)

    #  Geometry helpers

    def room_of(self, x, y):
        if self.room_map is None:
            return 0
        col = int((x - self.room_origin[0]) / self.room_res)
        row = (self.room_map.shape[0] - 1) - int(
            (y - self.room_origin[1]) / self.room_res)
        h, w = self.room_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return 0
        return int(self.room_map[row, col])

    def zone_key(self, x, y):
        return (int(math.floor(x / self.zone_bin_m)),
                int(math.floor(y / self.zone_bin_m)))

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x = float(tf.transform.translation.x)
            y = float(tf.transform.translation.y)
            return (x, y)
        except Exception:
            return None

    #  Random sampling (non-pool mode)

    def sample_spawn(self):
        if not self.free_cells:
            return None
        x, y = random.choice(self.free_cells)
        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, yaw)

    def sample_spawn_unexplored(self):
        if not self.free_cells:
            return None
        banned = set(self.recent_goal_rooms)
        candidates = [r for r in self.valid_rooms if r not in banned]
        if not candidates:
            candidates = list(self.valid_rooms)
        if not candidates:
            return self.sample_spawn()
        random.shuffle(candidates)
        room = candidates[0]
        cells = self.room_cells.get(room, [])
        if not cells:
            return self.sample_spawn()
        x, y = random.choice(cells)
        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, yaw)

    def sample_goal_in_room(self, room, rx, ry, min_d, max_d,
                            avoid_recent_zones):
        cells = self.room_cells.get(room, [])
        if not cells:
            return None
        recent = set(self.recent_zones) if avoid_recent_zones else set()
        if avoid_recent_zones:
            for _ in range(min(400, len(cells))):
                x, y = random.choice(cells)
                d = math.hypot(x - rx, y - ry)
                if not (min_d <= d <= max_d):
                    continue
                if self.zone_key(x, y) in recent:
                    continue
                return (x, y)
        for _ in range(min(400, len(cells))):
            x, y = random.choice(cells)
            d = math.hypot(x - rx, y - ry)
            if not (min_d <= d <= max_d):
                continue
            return (x, y)
        for _ in range(min(200, len(cells))):
            x, y = random.choice(cells)
            return (x, y)
        return None

    def sample_goal_from(self, ref_x, ref_y, min_d=None, max_d=None):
        if min_d is None:
            min_d = self.min_dist
        if max_d is None:
            max_d = self.max_dist
        banned = set(self.recent_goal_rooms)
        candidates = [r for r in self.valid_rooms if r not in banned]
        random.shuffle(candidates)
        for room in candidates:
            g = self.sample_goal_in_room(
                room, ref_x, ref_y, min_d, max_d, True)
            if g is not None:
                return g
        candidates = list(self.valid_rooms)
        random.shuffle(candidates)
        for room in candidates:
            g = self.sample_goal_in_room(
                room, ref_x, ref_y, min_d, max_d, True)
            if g is not None:
                return g
        for room in candidates:
            g = self.sample_goal_in_room(
                room, ref_x, ref_y, min_d, max_d, False)
            if g is not None:
                return g
        if self.free_cells:
            return random.choice(self.free_cells)
        return None

    def sample_start(self, ref_x=None, ref_y=None):
        pose = self.get_robot_pose()
        rx, ry = (pose if pose else (ref_x, ref_y))
        if rx is not None and random.random() < self.start_at_robot_prob:
            cell = self._closest_free_cell(rx, ry)
            if cell is not None:
                return cell
        spawn = self.sample_spawn_unexplored()
        if spawn is None:
            return None
        return (spawn[0], spawn[1])

    def _closest_free_cell(self, x, y):
        room = self.room_of(x, y)
        cells = self.room_cells.get(room, [])
        if not cells:
            return None
        best = None
        best_d = float('inf')
        for cx, cy in cells:
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best_d = d
                best = (cx, cy)
        return best

    #  Markers / start-goal publishing

    def publish_markers(self):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        if self.current_start is not None:
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'start'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(self.current_start[0])
            m.pose.position.y = float(self.current_start[1])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.9
            markers.markers.append(m)
        if self.current_goal is not None:
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'goal'
            m.id = 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(self.current_goal[0])
            m.pose.position.y = float(self.current_goal[1])
            m.pose.position.z = 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
            markers.markers.append(m)
        self.pub_markers.publish(markers)

    def clear_markers(self):
        markers = MarkerArray()
        for ns, mid in (('start', 0), ('goal', 1)):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            markers.markers.append(m)
        self.pub_markers.publish(markers)

    def publish_start_goal(self):
        if self.current_start is None or self.current_goal is None:
            return
        now = self.get_clock().now().to_msg()
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = now
        msg.point.x = float(self.current_goal[0])
        msg.point.y = float(self.current_goal[1])
        msg.point.z = 0.0
        self.pub_goal.publish(msg)
        self.publish_markers()

    #  Teleport

    def _stop_robot_hard(self, duration_s=0.2):
        zero = Twist()
        t_end = time.time() + duration_s
        while time.time() < t_end and rclpy.ok():
            self.cmd_vel_pub.publish(zero)
            time.sleep(0.05)

    def do_teleport(self, map_x, map_y, map_yaw):
        world_x = map_x + self.off_x
        world_y = map_y + self.off_y
        world_yaw = map_yaw + self.off_yaw

        self.teleport_done = False
        msg = Pose2D()
        msg.x = float(world_x)
        msg.y = float(world_y)
        msg.theta = float(world_yaw)
        self.teleport_pub.publish(msg)

        t0 = time.time()
        while (not self.teleport_done
               and time.time() - t0 < self.teleport_timeout_s
               and rclpy.ok()):
            time.sleep(0.02)

        if not self.teleport_done:
            self.get_logger().warn('teleport confirmation not received')
            return False

        if self.post_teleport_settle_s > 0:
            time.sleep(self.post_teleport_settle_s)
        time.sleep(0.5)
        return True

    def publish_initial_pose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        half_yaw = yaw * 0.5
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)
        cov = [0.0] * 36
        cov[0] = 0.5
        cov[7] = 0.5
        cov[35] = 0.15
        msg.pose.covariance = cov
        self.initialpose_pub.publish(msg)

    def wait_for_amcl_stable(self):
        samples = []
        t0 = time.time()
        while (time.time() - t0 < self.amcl_stable_timeout_s
               and rclpy.ok()):
            if self.last_amcl is not None:
                p = self.last_amcl.pose.pose.position
                samples.append((p.x, p.y))
                if len(samples) > self.amcl_stable_window:
                    samples.pop(0)
                if len(samples) == self.amcl_stable_window:
                    xs = [s[0] for s in samples]
                    ys = [s[1] for s in samples]
                    if (max(xs) - min(xs) < self.amcl_stable_tolerance_m
                            and max(ys) - min(ys)
                            < self.amcl_stable_tolerance_m):
                        return True
            time.sleep(0.1)
        self.get_logger().warn(
            f'AMCL did not stabilize within '
            f'{self.amcl_stable_timeout_s:.1f} s')
        return False

    def do_teleport_and_settle(self, spawn_xyyaw):
        if spawn_xyyaw is None:
            return False
        x, y, yaw = spawn_xyyaw
        self.paused_pub.publish(Bool(data=True))
        time.sleep(0.3)
        try:
            self.get_logger().info(
                f'TT stop -> ({x:+.2f}, {y:+.2f}, {yaw:+.2f})')
            self._stop_robot_hard(duration_s=0.3)
            if not self.do_teleport(x, y, yaw):
                self.get_logger().warn('TT teleport failed')
                return False
            self.last_amcl = None
            self.publish_initial_pose(x, y, yaw)
            if self.post_teleport_amcl_grace_s > 0:
                time.sleep(self.post_teleport_amcl_grace_s)
            self.wait_for_amcl_stable()
            if self.post_teleport_settle_s > 0:
                time.sleep(self.post_teleport_settle_s)
            return True
        finally:
            self._collision_prev = self.collision_now
            if self.collision_now:
                self._collision_since = time.time()
            else:
                self._collision_since = None
            self.paused_pub.publish(Bool(data=False))
            time.sleep(0.1)

    #  Verification & metadata

    def verify_saved(self, episode_id):
        time.sleep(1.5)
        # Launch may be shutting down by now (max_goals reached in C++).
        # rclpy.ok() returns False after rclpy.shutdown(), so skip all
        # rclpy calls (dbg.send, get_logger) to avoid Destroyable errors.
        if not rclpy.ok():
            return
        try:
            count = count_goals_in_scenario(self.scenario_dir)
            if count > self.last_saved_count:
                self.last_saved_count = count
                self.goals_saved_total = count
                self.goals_saved_session = (
                    self.goals_saved_total - self.goals_saved_total_at_start)
                self.update_info_channel()
                self.dbg.send(
                    f'ep {episode_id} saved', 'INFO', 'info', 3.0)
            else:
                self.dbg.send(
                    f'ep {episode_id} not saved by recorder',
                    'NODE_STATUS', 'err', -1.0)
                self.get_logger().warn(
                    f'episode {episode_id} not confirmed')
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warn(f'verify_saved failed: {exc}')

    def check_recorder_alive(self):
        if self.count_publishers('/mapless/local_occ_grid') == 0:
            return False, 'no publisher on /mapless/local_occ_grid'
        if self.count_publishers('/NN_goal') == 0:
            return False, 'no publisher on /NN_goal'
        return True, ''

    def _append_episode_metadata(self, entry):
        path = os.path.join(
            self.output_dir, self.scenario, 'episode_metadata.jsonl')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as exc:
            self.get_logger().warn(
                f'episode_metadata write failed: {exc}')

    def _record_failed_path(self, path_entry, reason, attempts):
        if path_entry is None:
            return
        entry = {
            'path_id': int(path_entry['id']),
            'class': path_entry['class'],
            'attempts': int(attempts),
            'last_status': str(reason),
            'start': list(path_entry['start']),
            'goal': list(path_entry['goal']),
        }
        self.failed_paths.append(entry)
        self.get_logger().warn(
            f'path id={entry["path_id"]} class={entry["class"]} '
            f'discarded after {attempts} attempts ({reason})')
        self._write_failed_paths()

    def _write_failed_paths(self):
        path = os.path.join(
            self.output_dir, self.scenario, 'failed_paths.json')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            by_id = {}
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        prev = json.load(f)
                    for e in prev.get('entries', []):
                        if e.get('path_id') is not None:
                            by_id[e['path_id']] = e
                except Exception:
                    by_id = {}
            for e in self.failed_paths:
                by_id[e['path_id']] = e
            merged = sorted(
                by_id.values(), key=lambda x: x.get('path_id', 0))
            payload = {
                'scenario': self.scenario,
                'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'class_quotas': dict(self.class_quotas),
                'class_counts': dict(self.class_counts),
                'entries': merged,
            }
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception as exc:
            self.get_logger().warn(
                f'failed_paths.json write failed: {exc}')

    #  Recording control

    def toggle_recording(self):
        if not self.rooms_ready:
            self.get_logger().warn('rooms not ready, button ignored')
            return
        if self._generating_pool:
            self.get_logger().info(
                'Pool generation in progress; please wait.')
            return
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self._generating_pool:
            self.get_logger().info(
                'Pool generation already in progress; please wait.')
            return
        if self._session_complete:
            self.get_logger().warn(
                'Session complete: all quotas met (or pools exhausted). '
                'Restart the launch to start a new session.')
            return
        if self._pool_gen_failed:
            self.get_logger().warn(
                f'Previous pool generation failed: '
                f'{self._pool_gen_error}. Restart the launch to retry.')
            return

        if self.use_path_pool and not self._pool_ready():
            if self.auto_generate_pool:
                self._generating_pool = True
                threading.Thread(
                    target=self._pool_gen_and_begin,
                    daemon=True).start()
                return

            with self._class_lock:
                deficits = dict(self._pool_deficits)
            self.dbg.send(
                f'Cannot record: pool not ready (deficits={deficits}) '
                f'and auto_generate_pool=False',
                'NODE_STATUS', 'err', -1.0)
            self.get_logger().error(
                f'Pool not ready (deficits={deficits}) and '
                f'auto_generate_pool=False. Delete the pool or '
                f'increase --st_sem_cmp_vc.')
            return

        self._begin_recording_after_pool()

    def _pool_gen_and_begin(self):
        try:
            ok = self._ensure_path_pool()
            if ok:
                self._pool_gen_failed = False
                self._pool_gen_error = ''
                self._begin_recording_after_pool()
            else:
                self._pool_gen_failed = True
                self._pool_gen_error = (
                    'generate_path_pool.py failed; check logs')
                self.dbg.send(
                    'Pool generation failed; see logs.',
                    'NODE_STATUS', 'err', -1.0)
                self.get_logger().error(
                    'Pool generation failed; recording not started. '
                    'Fix generator and restart the launch.')
        except Exception as exc:
            self._pool_gen_failed = True
            self._pool_gen_error = f'exception: {exc}'
            self.get_logger().error(
                f'_pool_gen_and_begin crashed: {exc}')
        finally:
            self._generating_pool = False

    def _begin_recording_after_pool(self):
        ok, reason = self.check_recorder_alive()
        if not ok:
            self.dbg.send(f'Cannot record: {reason}',
                          'NODE_STATUS', 'err', -1.0)
            self.get_logger().error(f'cannot start recording: {reason}')
            return

        with self._class_lock:
            if self.path_pool_loaded and sum(
                    self.class_quotas.values()) == 0:
                base, rem = divmod(self.num_goals, len(CLASS_NAMES))
                self.class_quotas = {
                    k: base + (1 if i < rem else 0)
                    for i, k in enumerate(CLASS_NAMES)}

        self.goals_saved_session = 0
        self.goals_saved_total_at_start = self.goals_saved_total
        self.recording = True
        self.last_active_time = time.time()
        self.dbg.send('Recording', 'STATUS', 'err', -1.0)
        self.update_info_channel()
        self.get_logger().info('recording started')
        self.generate_new_pair()

    def stop_recording(self):
        # Return current path to pool whenever we stop, regardless of state.
        # B during WAITING_FOR_START used to lose the path because the
        # previous condition only handled STATE_RECORDING.
        if (self.use_path_pool
                and self.current_path is not None
                and self.current_class):
            with self._class_lock:
                self.path_pool_by_class[
                    self.current_class].append(self.current_path)
            self.get_logger().info(
                f'path id={self.current_path["id"]} '
                f'class={self.current_class} returned to pool '
                f'(manual stop via B, state={self.state})')

        if self.state == STATE_RECORDING and self.current_goal is not None:
            self.pub_success.publish(Bool(data=False))
            self.pub_active.publish(Bool(data=False))

        self.recording = False
        self.state = STATE_IDLE
        self.current_start = None
        self.current_goal = None
        self.current_path = None
        self.current_class = ''
        self.path_attempts = 0
        self.clear_markers()
        self._write_failed_paths()
        self.dbg.send('Idle', 'STATUS', 'info', -1.0)

        self.get_logger().info(
            f'recording stopped. session={self.goals_saved_session} '
            f'total={self.goals_saved_total}')
        if self.use_path_pool:
            with self._class_lock:
                counts = dict(self.class_counts)
                quotas = dict(self.class_quotas)
            self.get_logger().info(
                f'class_counts={counts} class_quotas={quotas}')

    #  Pair generation

    def generate_new_pair(self):
        if self.use_path_pool:
            if self._should_stop_recording():
                with self._class_lock:
                    unmet = [(n, self.class_counts[n],
                              self.class_quotas[n])
                             for n in CLASS_NAMES
                             if self.class_counts[n] < self.class_quotas[n]]
                if unmet:
                    self.get_logger().warn(
                        f'Stopping: pool exhausted for unmet quotas '
                        f'{unmet}. Restart the launch with more paths.')
                else:
                    self.get_logger().info(
                        'Stopping: all quotas met.')
                self._session_complete = True
                self.stop_recording()
                return
            path = self._select_next_pool_path()
            if path is None:
                self.get_logger().info(
                    'No more paths available; stopping recording.')
                self.stop_recording()
                return

            self.current_path = path
            self.current_class = path['class']
            self.path_attempts = 0
            self.current_start = (
                float(path['start'][0]), float(path['start'][1]))
            self.current_goal = (
                float(path['goal'][0]), float(path['goal'][1]))
            self.goal_start_time = 0.0

            self.pub_episode_class.publish(
                String(data=self.current_class))
            self.publish_start_goal()
            self.state = STATE_WAITING_FOR_START

            self.get_logger().info(
                f'new pool path id={path["id"]} '
                f'class={self.current_class} '
                f'start=({self.current_start[0]:+.2f},'
                f'{self.current_start[1]:+.2f}) '
                f'-> goal=({self.current_goal[0]:+.2f},'
                f'{self.current_goal[1]:+.2f}) '
                f'episode_id={self.episode_id}')
            self.dbg.send(
                f'pool path id={path["id"]} '
                f'class={self.current_class}',
                'INFO', 'info', 5.0)
            return

        # Non-pool mode (original random sampling).
        pose = self.get_robot_pose()
        rx, ry = pose if pose else (None, None)
        start = self.sample_start(rx, ry)
        if start is None:
            self.get_logger().warn('could not sample start')
            return
        goal = self.sample_goal_from(start[0], start[1])
        if goal is None:
            self.get_logger().warn('could not sample goal')
            return

        self.current_path = None
        self.current_class = ''
        self.path_attempts = 0
        self.current_start = start
        self.current_goal = goal
        self.goal_start_time = 0.0
        self.publish_start_goal()
        self.state = STATE_WAITING_FOR_START

        room_s = self.room_of(start[0], start[1])
        room_g = self.room_of(goal[0], goal[1])
        self.get_logger().info(
            f'new pair start=({start[0]:+.2f},{start[1]:+.2f}) '
            f'room={room_s} -> goal=({goal[0]:+.2f},{goal[1]:+.2f}) '
            f'room={room_g}')
        self.dbg.send(
            f'pair room_s={room_s} room_g={room_g}',
            'INFO', 'info', 5.0)

    def begin_episode_at_start(self):
        # Publish episode_id right before goal_active so the C++ recorder
        # marks is_episode_start=1 on the first sample.
        self.episode_id += 1
        self.pub_episode.publish(Int32(data=self.episode_id))
        # Reset per-episode collision tracker.
        self.episode_had_collision = False

        if self.current_path is None:
            self.path_attempts = 1
        else:
            self.path_attempts += 1

        self.pub_episode_tries.publish(Int32(data=self.path_attempts))
        self.pub_active.publish(Bool(data=True))
        self.goal_start_time = time.time()
        self.last_active_time = time.time()
        self.state = STATE_RECORDING

        self.get_logger().info(
            f'episode {self.episode_id} begin at start, '
            f'goal=({self.current_goal[0]:+.2f},'
            f'{self.current_goal[1]:+.2f}) '
            f'attempt={self.path_attempts}')
        self.dbg.send(
            f'episode {self.episode_id} begin '
            f'attempt={self.path_attempts}',
            'INFO', 'info', 5.0)

    def finish_episode(self, success, reason):
        # Collision during episode cancels the save even if goal reached.
        if success and self.episode_had_collision:
            self.get_logger().warn(
                f'episode {self.episode_id}: collision during episode, '
                f'canceling save')
            success = False
            reason = 'collision_during_episode'

        if success and self.current_goal is not None:
            gx, gy = self.current_goal
            
            self.recent_goals.append((gx, gy))
            self.recent_goal_rooms.append(self.room_of(gx, gy))
            self.recent_zones.append(self.zone_key(gx, gy))

        self.pub_success.publish(Bool(data=success))
        self.pub_active.publish(Bool(data=False))

        self.get_logger().info(
            f'episode {self.episode_id} end | '
            f'success={success} reason={reason} '
            f'class={self.current_class} '
            f'attempt={self.path_attempts}')
        self.dbg.send(
            f'end success={success} reason={reason}',
            'INFO', 'info' if success else 'warn', 5.0)

        if success:
            # Log metadata only when the episode actually committed,
            # so failed attempts do not pollute the used-endpoint filter.
            if (self.use_path_pool
                    and self.current_path is not None
                    and self.current_start is not None
                    and self.current_goal is not None):
                self._append_episode_metadata({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'episode_id': self.episode_id,
                    'path_id': self.current_path['id'],
                    'class': self.current_class,
                    'start': [float(self.current_start[0]),
                              float(self.current_start[1])],
                    'goal': [float(self.current_goal[0]),
                             float(self.current_goal[1])],
                    'attempt': self.path_attempts,
                })
            threading.Thread(
                target=self.verify_saved,
                args=(self.episode_id,),
                daemon=True).start()
            if self.use_path_pool and self.current_class:
                with self._class_lock:
                    self.class_counts[self.current_class] += 1
                self.update_info_channel()
            self.current_path = None

        # Failure with retries remaining -> retry same path.
        if (not success and self.use_path_pool
                and self.current_path is not None):
            if self.path_attempts < self.path_max_attempts:
                self.get_logger().warn(
                    f'path id={self.current_path["id"]} failed '
                    f'(reason={reason}, attempt={self.path_attempts}/'
                    f'{self.path_max_attempts}). '
                    f'Drive to start to retry, or press Y to skip.')
                # Keep current_path, current_start, current_goal.
                # Keep episode_id (retries share the same id).
                self.state = STATE_WAITING_FOR_START
                self.goal_start_time = 0.0
                return
            # Retries exhausted -> mark failed.
            self._record_failed_path(
                self.current_path, reason, self.path_attempts)
            self.current_path = None

        # Success or retries exhausted: move on.
        self.current_start = None
        self.current_goal = None
        self.state = STATE_IDLE

        if self.recording:
            time.sleep(self.goal_pause_s)
            self.generate_new_pair()

    #  Teleport request

    def request_teleport(self, reason):
        if self.teleport_in_progress:
            self.get_logger().warn(
                f'drop teleport ({reason}): already running')
            return
        self.get_logger().warn(f'accept teleport ({reason})')
        self.dbg.send(f'teleport: {reason}',
                      'INFO', 'warn', 5.0)
        self.teleport_in_progress = True
        self._teleport_thread = threading.Thread(
            target=self._teleport_worker, daemon=True)
        self._teleport_thread.start()

    def _teleport_worker(self):
        try:
            if not self.rooms_ready:
                self.get_logger().warn(
                    'teleport ignored, rooms not ready')
                return

            if (self.state == STATE_RECORDING
                    and self.current_goal is not None):
                self.pub_success.publish(Bool(data=False))
                self.pub_active.publish(Bool(data=False))
                self.get_logger().info(
                    f'discarding ep {self.episode_id} for teleport')
                if self.use_path_pool and self.current_path is not None:
                    self.path_attempts += 1
                    if self.path_attempts >= self.path_max_attempts:
                        self._record_failed_path(
                            self.current_path,
                            'teleport_after_collision',
                            self.path_attempts)
                        self.current_path = None

            # Prefer teleporting to current_start if in pool mode.
            if (self.use_path_pool
                    and self.current_path is not None
                    and self.current_start is not None):
                sx, sy = self.current_start
                pose = self.get_robot_pose()
                if pose is not None:
                    yaw = math.atan2(sy - pose[1], sx - pose[0])
                else:
                    yaw = 0.0
                spawn = (sx, sy, yaw)
            else:
                spawn = self.sample_spawn_unexplored()

            if spawn is None:
                self.get_logger().warn('no spawn for teleport')
                return

            ok = self.do_teleport_and_settle(spawn)
            if not ok:
                self.get_logger().warn('teleport failed')
                return

            self.dbg.send('teleport done', 'INFO', 'info', 5.0)

            if self.current_path is not None:
                # We're back on the same path; remain in WAITING_FOR_START.
                self.state = STATE_WAITING_FOR_START
                self.clear_markers()
                self.publish_start_goal()
            elif self.recording:
                self.generate_new_pair()
        except Exception as exc:
            self.get_logger().error(f'teleport worker failed: {exc}')
            self.dbg.send(f'teleport worker failed: {exc}',
                          'NODE_STATUS', 'err', -1.0)
        finally:
            self.teleport_in_progress = False

    #  Tick

    def tick(self):
        try:
            self._tick_impl()
        except Exception as exc:
            self.get_logger().error(f'tick error: {exc}')
            self.dbg.send(f'tick error: {exc}',
                          'NODE_STATUS', 'err', -1.0)

    def _tick_impl(self):
        if self._generating_pool:
            # Ignore button presses while a pool is being generated.
            self.flag_button_b = False
            self.flag_button_x = False
            self.flag_button_y = False
            self.flag_collision = False
            return

        if self.teleport_in_progress:
            self.flag_collision = False
            return

        if self.flag_button_b:
            self.flag_button_b = False
            self.toggle_recording()
            return

        if self.flag_button_x:
            self.flag_button_x = False
            self.request_teleport('button X')
            return

        if self.flag_button_y:
            self.flag_button_y = False
            if self.recording:
                if (self.state == STATE_RECORDING
                        and self.current_goal is not None):
                    self.pub_success.publish(Bool(data=False))
                    self.pub_active.publish(Bool(data=False))
                if self.use_path_pool and self.current_path is not None:
                    self._record_failed_path(
                        self.current_path,
                        'skipped_by_operator',
                        max(1, self.path_attempts))
                self.current_path = None
                self.path_attempts = 0
                self.generate_new_pair()
            return

        if self.flag_collision:
            self.flag_collision = False
            self.get_logger().info('tick: consuming flag_collision')
            self.request_teleport('collision')
            return

        if (self.collision_now
                and self._collision_since is not None
                and time.time() - self._collision_since
                > self.collision_stuck_s):
            self._collision_since = None
            self.request_teleport('stuck collision')
            return

        if not self.recording:
            return

        if self.state == STATE_WAITING_FOR_START:
            if self.current_start is None:
                return
            pose = self.get_robot_pose()
            if pose is None:
                return
            rx, ry = pose
            sx, sy = self.current_start
            if math.hypot(sx - rx, sy - ry) < self.start_reached_radius_m:
                self.begin_episode_at_start()
            return

        if self.state != STATE_RECORDING:
            return

        if self.current_goal is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry = pose
        gx, gy = self.current_goal
        dist = math.hypot(gx - rx, gy - ry)

        if dist < self.goal_reached_radius_m:
            self.finish_episode(True, 'goal_reached')
            return

        now = time.time()
        if (self.goal_timeout_s > 0
                and now - self.goal_start_time > self.goal_timeout_s):
            self.get_logger().warn(
                f'episode {self.episode_id} timeout')
            self.finish_episode(False, 'timeout')
            return

        if (self.idle_timeout_s > 0
                and now - self.last_active_time > self.idle_timeout_s):
            self.get_logger().warn(
                f'episode {self.episode_id} idle '
                f'({now - self.last_active_time:.1f}s)')
            self.finish_episode(False, 'idle')
            return


def main():
    rclpy.init()

    atexit.register(_kill_orphans)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        node = TeleopDatasetRecorder()
    except Exception as exc:
        print(f'failed to start: {exc}')
        if rclpy.ok():
            rclpy.shutdown()
        return

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            if node.recording:
                node.stop_recording()
            node._write_failed_paths()
        except Exception as exc:
            print(f'[teleop] shutdown: failed_paths write: {exc}')
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
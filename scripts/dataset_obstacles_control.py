#!/usr/bin/env python3
"""
Sampler focused on obstacle avoidance and door crossing.

Buttons:
    B  -> toggle recording
    Y  -> discard current pair, generate new one
    X  -> emergency teleport to unexplored room

Any collision triggers an emergency teleport automatically.
"""

import csv
import math
import os
import random
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
from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image

import atexit
import signal
import subprocess

from debugger.utils import DebugClient

try:
    from scipy.ndimage import distance_transform_edt, label
except ImportError as exc:
    raise RuntimeError('scipy is required for room detection') from exc


STATE_IDLE = 0
STATE_WAITING_FOR_START = 1
STATE_RECORDING = 2


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


class DatasetObstaclesControl(Node):

    def __init__(self):
        super().__init__('dataset_obstacles_control')
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

        self.teleport_in_progress = False
        self._teleport_thread = None

        self._prev_buttons = {}

        self.goals_saved_total = 0
        self.goals_saved_session = 0

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

        threading.Thread(
            target=self.build_rooms_worker, daemon=True).start()

        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, self.heartbeat,
                          callback_group=self.heartbeat_group)

        self.get_logger().info(
            f'ready. map={self.map_name} seed={self.seed} '
            f'candidates={self.pair_candidates} '
            f'robot_radius={self.robot_radius_m:.2f} '
            f'buttons B={self.record_button_index} '
            f'X={self.teleport_button_index} '
            f'Y={self.regenerate_button_index}')

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

        self.declare_parameter('robot_radius_m', 0.25)
        self.declare_parameter('tight_margin_m', 0.25)
        self.declare_parameter('line_walk_step_m', 0.15)
        self.declare_parameter('pair_candidates', 30)
        self.declare_parameter('pair_goals_per_start', 2)

        self.declare_parameter('pair_weight_blocked', 3.0)
        self.declare_parameter('pair_weight_tight', 1.5)
        self.declare_parameter('pair_weight_cross_room', 2.0)
        self.declare_parameter('pair_weight_length', 0.5)
        self.declare_parameter('pair_penalty_short', 1.0)

        self.declare_parameter('collision_stuck_s', 2.0)
        self.declare_parameter('auto_world_offset', True)

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

        self.robot_radius_m = self.get_parameter('robot_radius_m').value
        self.tight_margin_m = self.get_parameter('tight_margin_m').value
        self.line_walk_step_m = self.get_parameter(
            'line_walk_step_m').value
        self.pair_candidates = self.get_parameter('pair_candidates').value
        self.pair_goals_per_start = self.get_parameter(
            'pair_goals_per_start').value

        self.pair_weight_blocked = self.get_parameter(
            'pair_weight_blocked').value
        self.pair_weight_tight = self.get_parameter(
            'pair_weight_tight').value
        self.pair_weight_cross_room = self.get_parameter(
            'pair_weight_cross_room').value
        self.pair_weight_length = self.get_parameter(
            'pair_weight_length').value
        self.pair_penalty_short = self.get_parameter(
            'pair_penalty_short').value

        self.collision_stuck_s = self.get_parameter(
            'collision_stuck_s').value
        self.auto_world_offset = self.get_parameter(
            'auto_world_offset').value

    def on_map(self, msg):
        self.map_msg = msg

    def on_collision(self, msg):
        now = bool(msg.data)
        if now != self._collision_prev:
            self.get_logger().info(
                f'collision edge {self._collision_prev} -> {now} '
                f'in_teleport={self.teleport_in_progress}')
        if now:
            if self._collision_since is None:
                self._collision_since = time.time()
            if not self._collision_prev and not self.teleport_in_progress:
                self.flag_collision = True
                self.get_logger().info('flag_collision set')
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

    def heartbeat(self):
        self.dbg.send('Node initialized and ready.',
                      'NODE_STATUS', 'info', -1.0)
        if self.teleport_in_progress:
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

    def world_to_grid(self, x, y):
        if self.dist_px is None:
            return None
        h, w = self.dist_px.shape
        col = int((x - self.room_origin[0]) / self.room_res)
        row = (h - 1) - int((y - self.room_origin[1]) / self.room_res)
        if not (0 <= row < h and 0 <= col < w):
            return None
        return (row, col)

    def obstacle_distance_m(self, x, y):
        if self.dist_px is None:
            return 0.0
        rc = self.world_to_grid(x, y)
        if rc is None:
            return 0.0
        return float(self.dist_px[rc[0], rc[1]]) * self.room_res

    def walk_line(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / self.line_walk_step_m))
        for i in range(n + 1):
            t = i / n
            yield (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    def score_pair(self, start, goal):
        sx, sy = start
        gx, gy = goal
        d = math.hypot(gx - sx, gy - sy)
        if d < self.min_dist or d > self.max_dist:
            return -1.0

        blocked = 0
        tight = 0
        total = 0
        tight_high = self.robot_radius_m + self.tight_margin_m

        for x, y in self.walk_line(sx, sy, gx, gy):
            total += 1
            obs_d = self.obstacle_distance_m(x, y)
            if obs_d < self.robot_radius_m:
                blocked += 1
            elif obs_d < tight_high:
                tight += 1

        blocked_frac = blocked / max(1, total)
        tight_frac = tight / max(1, total)

        room_s = self.room_of(sx, sy)
        room_g = self.room_of(gx, gy)
        crosses = 1.0 if room_s != room_g else 0.0

        d_norm = d / self.max_dist

        score = 0.0
        score += self.pair_weight_blocked * blocked_frac
        score += self.pair_weight_tight * tight_frac
        score += self.pair_weight_cross_room * crosses
        score += self.pair_weight_length * d_norm
        score -= self.pair_penalty_short * (1.0 - d_norm)
        return score

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

        if (rx is not None and random.random() < self.start_at_robot_prob):
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

    def verify_saved(self, episode_id):
        time.sleep(1.5)
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
            self.get_logger().warn(f'verify_saved failed: {exc}')

    def check_recorder_alive(self):
        if self.count_publishers('/mapless/local_occ_grid') == 0:
            return False, 'no publisher on /mapless/local_occ_grid'
        if self.count_publishers('/NN_goal') == 0:
            return False, 'no publisher on /NN_goal'
        return True, ''

    def toggle_recording(self):
        if not self.rooms_ready:
            self.get_logger().warn('rooms not ready, button ignored')
            return
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        ok, reason = self.check_recorder_alive()
        if not ok:
            self.dbg.send(f'Cannot record: {reason}',
                          'NODE_STATUS', 'err', -1.0)
            self.get_logger().error(f'cannot start recording: {reason}')
            return

        self.goals_saved_session = 0
        self.goals_saved_total_at_start = self.goals_saved_total
        self.recording = True
        self.last_active_time = time.time()

        self.dbg.send('Recording', 'STATUS', 'err', -1.0)
        self.update_info_channel()
        self.get_logger().info('recording started')

        self.generate_new_pair()

    def stop_recording(self):
        if self.state == STATE_RECORDING and self.current_goal is not None:
            self.pub_success.publish(Bool(data=False))
            self.pub_active.publish(Bool(data=False))
        self.recording = False
        self.state = STATE_IDLE
        self.current_start = None
        self.current_goal = None

        self.clear_markers()

        self.dbg.send('Idle', 'STATUS', 'info', -1.0)
        self.get_logger().info(
            f'recording stopped. session={self.goals_saved_session} '
            f'total={self.goals_saved_total}')

    def generate_new_pair(self):
        pose = self.get_robot_pose()
        rx, ry = pose if pose else (None, None)

        best_start = None
        best_goal = None
        best_score = -1.0
        total_tries = 0

        for _ in range(self.pair_candidates):
            start = self.sample_start(rx, ry)
            if start is None:
                continue
            for _ in range(self.pair_goals_per_start):
                goal = self.sample_goal_from(start[0], start[1])
                if goal is None:
                    continue
                total_tries += 1
                s = self.score_pair(start, goal)
                if s > best_score:
                    best_score = s
                    best_start = start
                    best_goal = goal

        if best_start is None or best_goal is None:
            self.get_logger().warn('could not sample pair')
            return

        self.current_start = best_start
        self.current_goal = best_goal
        self.goal_start_time = 0.0

        self.publish_start_goal()
        self.state = STATE_WAITING_FOR_START

        room_s = self.room_of(best_start[0], best_start[1])
        room_g = self.room_of(best_goal[0], best_goal[1])
        self.get_logger().info(
            f'new pair start=({best_start[0]:+.2f},{best_start[1]:+.2f}) '
            f'room={room_s} -> goal=({best_goal[0]:+.2f},{best_goal[1]:+.2f}) '
            f'room={room_g} score={best_score:.2f} tried={total_tries}')
        self.dbg.send(
            f'pair room_s={room_s} room_g={room_g} '
            f'score={best_score:.2f}',
            'INFO', 'info', 5.0)

    def begin_episode_at_start(self):
        self.episode_id += 1
        self.pub_episode.publish(Int32(data=self.episode_id))
        self.pub_active.publish(Bool(data=True))
        self.goal_start_time = time.time()
        self.last_active_time = time.time()
        self.state = STATE_RECORDING
        self.get_logger().info(
            f'episode {self.episode_id} begin at start, goal='
            f'({self.current_goal[0]:+.2f},{self.current_goal[1]:+.2f})')
        self.dbg.send(
            f'episode {self.episode_id} begin',
            'INFO', 'info', 5.0)

    def finish_episode(self, success, reason):
        if success and self.current_goal is not None:
            gx, gy = self.current_goal
            self.recent_goals.append((gx, gy))
            self.recent_goal_rooms.append(self.room_of(gx, gy))
            self.recent_zones.append(self.zone_key(gx, gy))

        self.pub_success.publish(Bool(data=success))
        self.pub_active.publish(Bool(data=False))

        self.get_logger().info(
            f'episode {self.episode_id} end | '
            f'success={success} reason={reason}')
        self.dbg.send(
            f'end success={success} reason={reason}',
            'INFO', 'info' if success else 'warn', 5.0)

        if success:
            threading.Thread(
                target=self.verify_saved,
                args=(self.episode_id,),
                daemon=True).start()

        self.current_start = None
        self.current_goal = None
        self.state = STATE_IDLE

        if self.recording:
            time.sleep(self.goal_pause_s)
            self.generate_new_pair()

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

            if self.state == STATE_RECORDING and self.current_goal is not None:
                self.pub_success.publish(Bool(data=False))
                self.pub_active.publish(Bool(data=False))
                self.get_logger().info(
                    f'discarding ep {self.episode_id} for teleport')
            self.current_start = None
            self.current_goal = None
            self.state = STATE_IDLE
            self.clear_markers()

            spawn = self.sample_spawn_unexplored()
            if spawn is None:
                self.get_logger().warn('no spawn for teleport')
                return

            ok = self.do_teleport_and_settle(spawn)
            if not ok:
                self.get_logger().warn('teleport failed')
                return

            self.dbg.send('teleport done', 'INFO', 'info', 5.0)

            self.get_logger().info('teleport worker body done')

            if self.recording:
                self.generate_new_pair()
        except Exception as exc:
            self.get_logger().error(f'teleport worker failed: {exc}')
            self.dbg.send(f'teleport worker failed: {exc}',
                          'NODE_STATUS', 'err', -1.0)
        finally:
            self.teleport_in_progress = False
            self.get_logger().info('teleport_in_progress cleared')

    def tick(self):
        try:
            self._tick_impl()
        except Exception as exc:
            self.get_logger().error(f'tick error: {exc}')
            self.dbg.send(f'tick error: {exc}',
                          'NODE_STATUS', 'err', -1.0)

    def _tick_impl(self):
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
        node = DatasetObstaclesControl()
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
        if node.recording:
            node.stop_recording()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
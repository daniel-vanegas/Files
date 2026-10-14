#!/usr/bin/env python3
import json
import math
import os
import random
import threading
import time
from collections import deque

import numpy as np
import rclpy
import rclpy.executors
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (
    PointStamped,
    Pose2D,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
)
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, Empty, Int32
from tf2_ros import Buffer, TransformListener

try:
    from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    _HAVE_NAV2 = True
except ImportError:
    _HAVE_NAV2 = False

from debugger.utils import DebugClient
from PIL import Image

try:
    from nav2_msgs.srv import ClearEntireCostmap
    _HAVE_CLEAR_SVC = True
except ImportError:
    _HAVE_CLEAR_SVC = False


class RandomGoalSampler(Node):

    def __init__(self):
        super().__init__('random_goal_sampler')

        self.declare_parameter('prohibition_png', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_wait_timeout_s', 20.0)
        self.declare_parameter('scenario', 'E01')
        self.declare_parameter('output_dir', 'src/Navigation/mapless_navigation/dataset/data')
        self.declare_parameter('use_path_pool', True)
        self.declare_parameter('pool_min_clearance_m', 0.15)
        self.declare_parameter('pool_max_retries', 3)

        self.declare_parameter('num_goals', 100)
        self.declare_parameter('seed', -1)

        self.declare_parameter('room_radius_m', 0.78)
        self.declare_parameter('min_room_area_m2', 2.0)

        self.declare_parameter('zone_bin_m', 0.8)
        self.declare_parameter('recent_zone_window', 8)
        self.declare_parameter('max_goals_per_room', 3)
        self.declare_parameter('recent_room_window', 5)

        self.declare_parameter('veto_room_duration_s', 90.0)
        self.declare_parameter('veto_room_min_fails', 3)

        self.declare_parameter('min_goal_distance_m', 2.0)
        self.declare_parameter('max_goal_distance_m', 3.0)
        self.declare_parameter('min_goal_distance_after_success_m', 1.0)
        self.declare_parameter('max_goal_distance_after_success_m', 3.0)
        self.declare_parameter('max_euclidean_goal_distance_m', 3.0)
        self.declare_parameter('max_path_length_m', 3.0)
        self.declare_parameter('goal_position_tolerance_m', 0.25)
        self.declare_parameter('plan_service_name', '/path_planner/plan_path_with_augmented')
        self.declare_parameter('candidate_attempts_per_class', 120)

        self.declare_parameter('goal_timeout_s', 60.0)
        self.declare_parameter('goal_pause_s', 0.5)
        self.declare_parameter('min_attempt_s', 3.0)
        self.declare_parameter('min_goal_interval_s', 1.0)
        self.declare_parameter('max_consecutive_fast_fails', 10)
        self.declare_parameter('instant_fail_threshold_s', 0.5)
        self.declare_parameter('max_instant_fails', 3)

        self.declare_parameter('teleport_at_start', True)
        self.declare_parameter('min_spawn_goal_distance_m', 1.5)
        self.declare_parameter('teleport_timeout_s', 4.0)
        self.declare_parameter('initial_pose_wait_s', 0.5)
        self.declare_parameter('post_teleport_settle_s', 1.5)

        self.declare_parameter('post_teleport_clear_costmap', True)
        self.declare_parameter('post_teleport_costmap_clear_wait_s', 0.5)
        self.declare_parameter('post_teleport_nav_settle_s', 1.5)
        self.declare_parameter('costmap_clear_service_timeout_s', 2.0)

        self.declare_parameter('max_attempts_per_goal', 2)

        self.declare_parameter('amcl_stable_timeout_s', 20.0)
        self.declare_parameter('amcl_stable_window', 15)
        self.declare_parameter('amcl_stable_tolerance_m', 0.05)
        self.declare_parameter('post_teleport_amcl_grace_s', 2.0)
        self.declare_parameter('navigation_backend', 'nav2')

        self.declare_parameter('world_offset_x', 0.0)
        self.declare_parameter('world_offset_y', 0.0)
        self.declare_parameter('world_offset_yaw', 0.0)

        self.declare_parameter('enable_collision_abort', True)
        self.declare_parameter('collision_topic', '/navigation/collision')
        self.declare_parameter('collision_stop_duration_s', 0.5)
        self.declare_parameter('cmd_vel_topic', '/hardware/base/cmd_vel')

        self.prohibition = self.get_parameter('prohibition_png').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_wait_timeout_s = self.get_parameter('map_wait_timeout_s').value
        self.scenario = str(self.get_parameter('scenario').value)
        self.output_dir = os.path.abspath(str(self.get_parameter('output_dir').value))
        self.use_path_pool = bool(self.get_parameter('use_path_pool').value)
        self.pool_min_clearance_m = float(self.get_parameter('pool_min_clearance_m').value)
        self.pool_max_retries = int(self.get_parameter('pool_max_retries').value)

        self.num_goals = self.get_parameter('num_goals').value
        seed = self.get_parameter('seed').value

        self.room_radius_m = self.get_parameter('room_radius_m').value
        self.min_room_area_m2 = self.get_parameter('min_room_area_m2').value

        self.zone_bin_m = self.get_parameter('zone_bin_m').value
        self.recent_zone_window = self.get_parameter('recent_zone_window').value
        self.max_goals_per_room = self.get_parameter('max_goals_per_room').value
        self.recent_room_window = self.get_parameter('recent_room_window').value
        self.veto_room_duration_s = self.get_parameter('veto_room_duration_s').value
        self.veto_room_min_fails = self.get_parameter('veto_room_min_fails').value

        self.min_dist = self.get_parameter('min_goal_distance_m').value
        self.max_dist = self.get_parameter('max_goal_distance_m').value
        self.min_dist_after = self.get_parameter('min_goal_distance_after_success_m').value
        self.max_dist_after = self.get_parameter('max_goal_distance_after_success_m').value
        self.max_euclidean_goal_distance = float(self.get_parameter('max_euclidean_goal_distance_m').value)
        self.max_path_length = float(self.get_parameter('max_path_length_m').value)
        self.goal_position_tolerance = float(
            self.get_parameter('goal_position_tolerance_m').value)
        self.plan_service_name = self.get_parameter('plan_service_name').value
        self.candidate_attempts_per_class = int(self.get_parameter('candidate_attempts_per_class').value)

        self.goal_timeout_s = self.get_parameter('goal_timeout_s').value
        self.goal_pause_s = self.get_parameter('goal_pause_s').value
        self.min_attempt_s = self.get_parameter('min_attempt_s').value
        self.min_goal_interval_s = self.get_parameter('min_goal_interval_s').value
        self.max_consecutive_fast_fails = self.get_parameter('max_consecutive_fast_fails').value
        self.instant_fail_threshold_s = self.get_parameter('instant_fail_threshold_s').value
        self.max_instant_fails = self.get_parameter('max_instant_fails').value

        self.teleport_at_start = self.get_parameter('teleport_at_start').value
        self.min_spawn_dist = self.get_parameter('min_spawn_goal_distance_m').value
        self.teleport_timeout_s = self.get_parameter('teleport_timeout_s').value
        self.initial_pose_wait_s = self.get_parameter('initial_pose_wait_s').value
        self.post_teleport_settle_s = self.get_parameter('post_teleport_settle_s').value

        self.post_teleport_clear_costmap = self.get_parameter('post_teleport_clear_costmap').value
        self.post_teleport_costmap_clear_wait_s = self.get_parameter('post_teleport_costmap_clear_wait_s').value
        self.post_teleport_nav_settle_s = self.get_parameter('post_teleport_nav_settle_s').value
        self.costmap_clear_service_timeout_s = self.get_parameter('costmap_clear_service_timeout_s').value

        self.max_attempts_per_goal = self.get_parameter('max_attempts_per_goal').value

        self.amcl_stable_timeout_s = self.get_parameter('amcl_stable_timeout_s').value
        self.amcl_stable_window = self.get_parameter('amcl_stable_window').value
        self.amcl_stable_tolerance_m = self.get_parameter('amcl_stable_tolerance_m').value
        self.post_teleport_amcl_grace_s = self.get_parameter('post_teleport_amcl_grace_s').value
        self.navigation_backend = self.get_parameter('navigation_backend').value

        self.off_x = self.get_parameter('world_offset_x').value
        self.off_y = self.get_parameter('world_offset_y').value
        self.off_yaw = self.get_parameter('world_offset_yaw').value

        self.enable_collision_abort = self.get_parameter('enable_collision_abort').value
        self.collision_topic = self.get_parameter('collision_topic').value
        self.collision_stop_duration_s = self.get_parameter('collision_stop_duration_s').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        if seed is None or int(seed) < 0:
            seed = int.from_bytes(os.urandom(8), byteorder='big')
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            seed_info = f'random:{seed}'
        else:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            seed_info = str(seed)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_goal = self.create_publisher(PointStamped, '/mapless_nav/current_goal', 10)
        self.pub_active = self.create_publisher(Bool, '/mapless_nav/goal_active', 10)
        self.pub_episode = self.create_publisher(Int32, '/mapless_nav/episode_id', 10)
        self.pub_goal_success = self.create_publisher(Bool, '/mapless_nav/goal_success', 10)
        self.teleport_pub = self.create_publisher(Pose2D, '/webots/teleport', 10)
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.simple_move_stop_pub = self.create_publisher(Empty, '/mapless_nav/stop', 10)

        self.teleport_done = False
        self.create_subscription(Bool, '/webots/teleport_done', self._on_teleport_done, 10)

        self.last_amcl = None
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl_pose, 10)

        self.nav_status = None
        self.create_subscription(GoalStatus, '/mapless_nav/nav_status', self._on_nav_status, 10)

        self.collision_event = threading.Event()
        self.create_subscription(Bool, self.collision_topic, self._on_collision, 10)

        self.map_msg = None
        self._map_received_event = threading.Event()
        self.clearance_m = None
        map_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        self.prohib_arr = None
        if self.prohibition and os.path.isfile(self.prohibition):
            img = Image.open(self.prohibition).convert('L')
            self.prohib_arr = np.array(img, dtype=np.uint8) > 127

        self.plan_client = self.create_client(GetPlan, self.plan_service_name)

        self.class_names = ('straight', 'semi_complex', 'complex', 'v_complex')
        self.class_counts = {name: 0 for name in self.class_names}
        self.class_quota = int(math.ceil(float(self.num_goals) / 4.0))
        self._class_lock = threading.Lock()
        self.path_pool_by_class = {name: [] for name in self.class_names}
        self.path_pool_loaded = False

        self._costmap_clients = []
        if self.post_teleport_clear_costmap and _HAVE_CLEAR_SVC:
            for svc_name in (
                    '/global_costmap/clear_entirely_global_costmap',
                    '/local_costmap/clear_entirely_local_costmap'):
                cli = self.create_client(ClearEntireCostmap, svc_name)
                self._costmap_clients.append((svc_name, cli))

        self.room_map = None
        self.room_res = None
        self.room_origin = (0.0, 0.0)
        self.room_cells = {}
        self.valid_rooms = set()
        self.free_cells = []

        self.recent_goals = deque(maxlen=32)
        self.recent_goal_rooms = deque(maxlen=self.recent_room_window)
        self.recent_zones = deque(maxlen=self.recent_zone_window)
        self.vetoed_rooms = {}
        self.fail_count_per_room = {}

        self.get_logger().info(
            f'Sampler ready. seed={seed_info} num_goals={self.num_goals} '
            f'room_radius_m={self.room_radius_m:.2f} '
            f'zone_bin_m={self.zone_bin_m:.2f} '
            f'max_goals_per_room={self.max_goals_per_room} '
            f'recent_room_window={self.recent_room_window} '
            f'veto_room_duration_s={self.veto_room_duration_s:.0f} '
            f'use_path_pool={self.use_path_pool} scenario={self.scenario}')

        if self.navigation_backend == 'nav2':
            if not _HAVE_NAV2:
                raise RuntimeError('navigation_backend=nav2 but nav2_simple_commander not available')
            self.navigator = BasicNavigator()
            self.navigator.waitUntilNav2Active()
            self.get_logger().info('Nav2 active.')
        else:
            self.navigator = None
            self.get_logger().info('Using classic navigation backend (navigation_planner).')

        self.dbg = DebugClient(self, verbose=True)
        self.dbg.send('Node initialized and ready.', 'NODE_STATUS', 'info', -1.0)
        self.dbg.send(f'completed=0/{self.num_goals}', 'TOTALS', 'info', -1.0)
        self._publish_class_counts()

        self.episode_id = 0
        self._completed_goals = 0
        self.pub_episode.publish(Int32(data=self.episode_id))

        self.finished = False
        self.shutdown_flag = False
        self._worker_launched = False

        self.create_timer(3.0, self._launch_worker_once)
        self.heartbeat_timer = self.create_timer(1.0, self._heartbeat)

    def _heartbeat(self):
        if not hasattr(self, 'dbg'):
            return
        self.dbg.send('Node initialized and ready.', 'NODE_STATUS', 'info', -1.0)

    def _on_map(self, msg):
        first_map = self.map_msg is None
        self.map_msg = msg
        self._map_received_event.set()

        if first_map:
            status = (
                f'Map received: {msg.info.width}x{msg.info.height}, '
                f'resolution={msg.info.resolution:.3f}, '
                f'frame={msg.header.frame_id}'
            )
            self.get_logger().info(status)
            if hasattr(self, 'dbg'):
                self.dbg.send(status, 'SAMPLER_STATUS', 'info', -1.0)

    def _on_teleport_done(self, msg):
        self.teleport_done = bool(msg.data)

    def _on_amcl_pose(self, msg):
        self.last_amcl = msg

    def _on_collision(self, msg):
        if not self.enable_collision_abort:
            return
        if msg.data:
            self.collision_event.set()

    def _on_nav_status(self, msg):
        self.nav_status = msg.status

    def _launch_worker_once(self):
        if self._worker_launched:
            return
        self._worker_launched = True
        threading.Thread(target=self._sampling_worker, daemon=True).start()

    def _stop_robot_hard(self, duration_s=None):
        if duration_s is None:
            duration_s = self.collision_stop_duration_s

        if self.navigation_backend == 'classic':
            self.simple_move_stop_pub.publish(Empty())
            time.sleep(0.05)
            return

        zero = Twist()
        t_end = time.time() + duration_s
        while time.time() < t_end and not self.shutdown_flag:
            self.cmd_vel_pub.publish(zero)
            time.sleep(0.05)

    def _clear_costmaps(self):
        if not self._costmap_clients:
            return False
        for svc_name, cli in self._costmap_clients:
            try:
                if not cli.service_is_ready():
                    if not cli.wait_for_service(timeout_sec=0.5):
                        self.get_logger().warn(f'Costmap clear {svc_name} not ready, skipping.')
                        continue
                req = ClearEntireCostmap.Request()
                cli.call_async(req)
            except Exception as e:
                self.get_logger().warn(f'Costmap clear {svc_name} failed: {e}')
        return True

    def _post_teleport_settle(self):
        if self.post_teleport_clear_costmap:
            self._clear_costmaps()
            if self.post_teleport_costmap_clear_wait_s > 0:
                time.sleep(self.post_teleport_costmap_clear_wait_s)
        if self.post_teleport_settle_s > 0:
            time.sleep(self.post_teleport_settle_s)
        if self.post_teleport_nav_settle_s > 0:
            time.sleep(self.post_teleport_nav_settle_s)

    def _build_rooms(self):
        info = self.map_msg.info
        w, h = info.width, info.height
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        data = np.array(self.map_msg.data, dtype=np.int8).reshape(h, w)

        free = (data == 0)

        if self.prohib_arr is not None:
            mask_img = Image.fromarray((self.prohib_arr * 255).astype(np.uint8))
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            free = free & (np.array(mask_img) > 127)

        try:
            from scipy.ndimage import distance_transform_edt, label
        except ImportError:
            self.get_logger().error('scipy is required for room detection.')
            raise

        self.get_logger().info(
            f'Computing distance transform on {w}x{h} map '
            f'(room_radius_m={self.room_radius_m:.2f}, res={res:.3f})...')

        dist_px = distance_transform_edt(free)
        self.clearance_m = dist_px.astype(np.float32) * float(res)
        min_dist_px = self.room_radius_m / res
        traversable = dist_px >= min_dist_px

        n_free = int(np.count_nonzero(traversable))
        self.get_logger().info(f'Distance transform done. {n_free} traversable cells. Labeling rooms...')

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

        self.get_logger().info(f'Rooms: {len(valid)} valid of {n}, traversable cells: {len(self.free_cells)}')

    def _room_of(self, x, y):
        if self.room_map is None:
            return 0
        col = int((x - self.room_origin[0]) / self.room_res)
        row = (self.room_map.shape[0] - 1) - int((y - self.room_origin[1]) / self.room_res)
        h, w = self.room_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return 0
        return int(self.room_map[row, col])

    def _zone_key(self, x, y):
        return (int(math.floor(x / self.zone_bin_m)), int(math.floor(y / self.zone_bin_m)))

    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def _sampling_debug_state(self):
        now = time.time()
        self.vetoed_rooms = {r: t for r, t in self.vetoed_rooms.items() if t > now}

    def _banned_rooms(self, robot_room):
        banned = set()
        banned.add(robot_room)
        for r in self.recent_goal_rooms:
            banned.add(r)
        for r in self.vetoed_rooms:
            banned.add(r)
        return banned

    def _sample_goal_in_room(self, room, rx, ry, min_d, max_d, avoid_recent_zones):
        cells = self.room_cells.get(room, [])
        if not cells:
            return None

        recent_zone_set = set(self.recent_zones) if avoid_recent_zones else set()

        if avoid_recent_zones:
            for _ in range(min(400, len(cells))):
                x, y = random.choice(cells)
                d = math.hypot(x - rx, y - ry)
                if not (min_d <= d <= max_d):
                    continue
                if self._zone_key(x, y) in recent_zone_set:
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

    def _sample_goal(self, min_d=None, max_d=None):
        if min_d is None:
            min_d = self.min_dist
        if max_d is None:
            max_d = self.max_dist

        pose = self._robot_pose()
        if pose is None:
            self.get_logger().warn('_sample_goal: no robot pose')
            return None
        rx, ry = pose
        robot_room = self._room_of(rx, ry)

        if robot_room in self.room_cells:
            g = self._sample_goal_in_room(robot_room, rx, ry, min_d, max_d, avoid_recent_zones=True)
            if g is not None:
                return g
            self.get_logger().warn(
                f'_sample_goal: room {robot_room} ({len(self.room_cells[robot_room])} cells) '
                f'failed for d in [{min_d:.2f}, {max_d:.2f}]')
        else:
            self.get_logger().warn(
                f'_sample_goal: robot at ({rx:.2f},{ry:.2f}) in room {robot_room} (not a valid room)')

        candidates = list(self.valid_rooms)
        random.shuffle(candidates)
        for room in candidates:
            g = self._sample_goal_in_room(room, rx, ry, min_d, max_d, avoid_recent_zones=False)
            if g is not None:
                self.get_logger().info(f'_sample_goal: fallback to room {room}')
                return g

        self.get_logger().warn('_sample_goal: no candidate in any room')
        return None

    def _sample_goal_balanced(self):
        pose = self._robot_pose()
        if pose is None:
            return self._sample_goal()

        rx, ry = pose
        current_room = self._room_of(rx, ry)

        available_rooms = [r for r in self.valid_rooms if r != current_room and r not in self.vetoed_rooms]
        if not available_rooms:
            available_rooms = list(self.valid_rooms)

        target_room = random.choice(available_rooms)

        return self._sample_goal_in_room(
            target_room, rx, ry,
            min_d=self.min_dist,
            max_d=self.max_dist,
            avoid_recent_zones=True
        )

    def _sample_short_goal(self):
        return self._sample_goal(min_d=self.min_dist_after, max_d=self.max_dist_after)

    def _sample_spawn_away_from(self, goal_xy):
        pose = self._robot_pose()
        rx, ry = pose if pose else (0.0, 0.0)
        current_room = self._room_of(rx, ry)

        banned = self._banned_rooms(current_room)
        if goal_xy is not None:
            gx, gy = goal_xy
            banned.add(self._room_of(gx, gy))
        else:
            gx, gy = None, None

        candidates = [r for r in self.valid_rooms if r not in banned]
        random.shuffle(candidates)
        for room in candidates:
            cells = self.room_cells.get(room, [])
            if not cells:
                continue
            for _ in range(min(200, len(cells))):
                x, y = random.choice(cells)
                if gx is not None and math.hypot(x - gx, y - gy) < self.min_spawn_dist:
                    continue
                yaw = random.uniform(-math.pi, math.pi)
                return (x, y, yaw)

        candidates = [r for r in self.valid_rooms if r != current_room]
        random.shuffle(candidates)
        for room in candidates:
            cells = self.room_cells.get(room, [])
            if not cells:
                continue
            x, y = random.choice(cells)
            yaw = random.uniform(-math.pi, math.pi)
            return (x, y, yaw)

        if self.free_cells:
            x, y = random.choice(self.free_cells)
            yaw = random.uniform(-math.pi, math.pi)
            return (x, y, yaw)
        return None

    def _goals_same_room(self):
        if len(self.recent_goal_rooms) < self.max_goals_per_room:
            return False
        recent = list(self.recent_goal_rooms)[-self.max_goals_per_room:]
        return len(set(recent)) == 1

    def _teleport_and_settle(self, spawn_xyyaw):
        if spawn_xyyaw is None:
            status = 'stage=initial_spawn_failed reason=no_valid_spawn'
            self.get_logger().warn(status)
            self.dbg.send(status, 'WORKER_STAGE', 'error', -1.0)
            return False

        x, y, yaw = spawn_xyyaw
        self.dbg.send(
            f'stage=teleport_started x={x:.2f} y={y:.2f} yaw={yaw:.2f}',
            'WORKER_STAGE', 'info', -1.0)
        self.get_logger().info(f'Teleporting to ({x:.2f}, {y:.2f}, {yaw:.2f})')
        self._stop_robot_hard(duration_s=0.2)

        self.dbg.send('stage=waiting_for_teleport_confirmation', 'WORKER_STAGE', 'info', -1.0)
        if not self._do_teleport(x, y, yaw):
            status = 'stage=teleport_failed reason=no_teleport_done_confirmation'
            self.get_logger().warn(status)
            self.dbg.send(status, 'WORKER_STAGE', 'error', -1.0)
            return False

        self.dbg.send('stage=teleport_confirmed waiting_for_amcl', 'WORKER_STAGE', 'info', -1.0)
        self.last_amcl = None
        self._publish_initial_pose(x, y, yaw)
        if self.post_teleport_amcl_grace_s > 0:
            time.sleep(self.post_teleport_amcl_grace_s)

        self.dbg.send('stage=waiting_for_amcl_stability', 'WORKER_STAGE', 'info', -1.0)
        amcl_stable = self._wait_for_amcl_stable()
        self.dbg.send(
            f'stage=amcl_wait_finished stable={str(amcl_stable).lower()} '
            f'amcl_message_received={str(self.last_amcl is not None).lower()}',
            'WORKER_STAGE', 'info' if amcl_stable else 'warn', -1.0)

        self.dbg.send('stage=post_teleport_settle', 'WORKER_STAGE', 'info', -1.0)
        self._post_teleport_settle()
        self.recent_goals.clear()
        self.recent_goal_rooms.clear()
        self.recent_zones.clear()

        self.collision_event.clear()
        time.sleep(0.3)
        if self.collision_event.is_set():
            status = 'stage=teleport_rejected reason=collision_at_spawn'
            self.get_logger().warn(status)
            self.dbg.send(status, 'WORKER_STAGE', 'warn', -1.0)
            return False

        self.dbg.send('stage=teleport_and_settle_finished', 'WORKER_STAGE', 'info', -1.0)
        return True

    def _do_teleport(self, map_x, map_y, map_yaw):
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
        while (not self.teleport_done and
               time.time() - t0 < self.teleport_timeout_s and
               not self.shutdown_flag):
            time.sleep(0.02)

        if not self.teleport_done:
            self.get_logger().warn('Teleport confirmation not received.')
            return False

        if self.post_teleport_settle_s > 0:
            time.sleep(self.post_teleport_settle_s)
        time.sleep(self.initial_pose_wait_s)
        return True

    def _publish_initial_pose(self, x, y, yaw):
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
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.07
        msg.pose.covariance = cov

        self.initialpose_pub.publish(msg)

    def _wait_for_amcl_stable(self, timeout_s=None, window=None, tol=None):
        if timeout_s is None:
            timeout_s = self.amcl_stable_timeout_s
        if window is None:
            window = self.amcl_stable_window
        if tol is None:
            tol = self.amcl_stable_tolerance_m

        samples = []
        t0 = time.time()
        while time.time() - t0 < timeout_s and not self.shutdown_flag:
            if self.last_amcl is not None:
                p = self.last_amcl.pose.pose.position
                samples.append((p.x, p.y))
                if len(samples) > window:
                    samples.pop(0)
                if len(samples) == window:
                    xs = [s[0] for s in samples]
                    ys = [s[1] for s in samples]
                    if (max(xs) - min(xs) < tol and max(ys) - min(ys) < tol):
                        return True
            time.sleep(0.1)

        if not self.shutdown_flag:
            self.get_logger().warn(f'AMCL did not stabilize within {timeout_s:.1f} s.')
        return False

    def _publish_class_counts(self):
        with self._class_lock:
            counts = dict(self.class_counts)
        self.dbg.send(
            f'straight={counts["straight"]} '
            f'semi={counts["semi_complex"]} '
            f'complex={counts["complex"]} '
            f'v_complex={counts["v_complex"]}',
            'CLASSES', 'info', -1.0)

    def _next_target_class(self):
        with self._class_lock:
            remaining = [name for name in self.class_names
                         if self.class_counts[name] < self.class_quota]
            if not remaining:
                return None
            lowest = min(self.class_counts[name] for name in remaining)
            tied = [name for name in remaining if self.class_counts[name] == lowest]
            return random.choice(tied)

    def _load_path_pool(self):
        pool_path = os.path.join(self.output_dir, self.scenario, 'path_pool.json')
        poses_path = os.path.join(self.output_dir, self.scenario, 'path_pool_poses.json')
        self.dbg.send(
            f'stage=pool_load_started scenario={self.scenario} pool={pool_path}',
            'WORKER_STAGE', 'info', -1.0)

        if not os.path.isfile(pool_path) or not os.path.isfile(poses_path):
            status = (
                f'PATH_POOL_ERROR missing_file pool_exists={os.path.isfile(pool_path)} '
                f'poses_exists={os.path.isfile(poses_path)} pool={pool_path}')
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=pool_load_failed reason=missing_file', 'WORKER_STAGE', 'error', -1.0)
            return False

        try:
            with open(pool_path, 'r') as f:
                pool = json.load(f)
            with open(poses_path, 'r') as f:
                poses_by_id = json.load(f)
        except Exception as exc:
            status = f'PATH_POOL_ERROR failed_to_read error={exc}'
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=pool_load_failed reason=invalid_json', 'WORKER_STAGE', 'error', -1.0)
            return False

        if (int(pool.get('pool_version', 0)) < 4 or
                pool.get('classifier') != 'geometry_morphology_v2'):
            status = (
                f'PATH_POOL_ERROR incompatible_pool_version={pool.get("pool_version", 0)} '
                f'classifier={pool.get("classifier", "missing")} '
                'regenerate_pool_with_geometry_morphology_v2=true')
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=pool_load_failed reason=old_version', 'WORKER_STAGE', 'error', -1.0)
            return False

        pool_scenario = str(pool.get('scenario', ''))
        if pool_scenario and pool_scenario != self.scenario:
            status = f'PATH_POOL_ERROR scenario_mismatch expected={self.scenario} actual={pool_scenario}'
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=pool_load_failed reason=scenario_mismatch', 'WORKER_STAGE', 'error', -1.0)
            return False

        candidate_counts = {name: 0 for name in self.class_names}
        rejected = {'missing_poses': 0, 'bad_record': 0, 'distance': 0,
                    'path_length': 0, 'clearance': 0, 'endpoint_mismatch': 0}

        for entry in pool.get('paths', []):
            cls = entry.get('class')
            if cls not in self.path_pool_by_class:
                rejected['bad_record'] += 1
                continue
            try:
                path_id = str(entry['id'])
                start = [float(entry['start'][0]), float(entry['start'][1])]
                goal = [float(entry['goal'][0]), float(entry['goal'][1])]
                metrics = entry['metrics']
                euclidean = math.hypot(goal[0] - start[0], goal[1] - start[1])
                path_length = float(metrics['path_length_m'])
                clearance = float(metrics['min_clearance_m'])
                points = poses_by_id.get(path_id)
                if not points or len(points) < 2:
                    rejected['missing_poses'] += 1
                    continue
                points = [[float(point[0]), float(point[1])] for point in points]
            except (KeyError, TypeError, ValueError, IndexError):
                rejected['bad_record'] += 1
                continue

            if euclidean < float(self.min_dist) - 1e-6 or euclidean > self.max_euclidean_goal_distance + 1e-6:
                rejected['distance'] += 1
                continue
            if path_length > self.max_path_length + 1e-6:
                rejected['path_length'] += 1
                continue
            if clearance + 1e-6 < self.pool_min_clearance_m:
                rejected['clearance'] += 1
                continue
            if (math.hypot(points[0][0] - start[0], points[0][1] - start[1]) > 0.05 or
                    math.hypot(points[-1][0] - goal[0], points[-1][1] - goal[1]) > 0.05):
                rejected['endpoint_mismatch'] += 1
                continue

            self.path_pool_by_class[cls].append({
                'id': int(entry['id']),
                'class': cls,
                'start': start,
                'goal': goal,
                'poses': points,
                'metrics': metrics,
                'score': float(entry.get('score', 0.0)),
            })
            candidate_counts[cls] += 1

        for cls in self.class_names:
            random.shuffle(self.path_pool_by_class[cls])

        loaded_total = sum(candidate_counts.values())
        self.get_logger().info(
            f'Path pool loaded: scenario={self.scenario} '
            f'version={pool.get("pool_version")} '
            f'usable={loaded_total}/{len(pool.get("paths", []))} '
            f'class_counts={candidate_counts} rejected={rejected}')

        insufficient = [cls for cls in self.class_names
                        if candidate_counts[cls] < self.class_quota]
        if insufficient:
            status = (
                f'PATH_POOL_ERROR insufficient_class_paths quota_per_class={self.class_quota} '
                f'insufficient={insufficient} available={candidate_counts} rejected={rejected}')
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=pool_load_failed reason=insufficient_class_paths', 'WORKER_STAGE', 'error', -1.0)
            return False

        self.path_pool_loaded = True
        self.dbg.send(
            f'stage=pool_load_finished scenario={self.scenario} total={loaded_total} '
            f'quota_per_class={self.class_quota}',
            'WORKER_STAGE', 'info', -1.0)
        return True

    def _take_pool_path(self, target_class):
        paths = self.path_pool_by_class.get(target_class, [])
        if not paths:
            return None
        return paths.pop()

    def _plan_path(self, goal_xy, timeout_s=2.0):
        pose = self._robot_pose()
        if pose is None:
            return None
        if not self.plan_client.service_is_ready():
            if not self.plan_client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(f'Plan service not available: {self.plan_service_name}')
                return None

        start = PoseStamped()
        start.header.frame_id = self.map_frame
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose.position.x = float(pose[0])
        start.pose.position.y = float(pose[1])
        start.pose.orientation.w = 1.0

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(goal_xy[0])
        goal.pose.position.y = float(goal_xy[1])
        goal.pose.orientation.w = 1.0

        request = GetPlan.Request()
        request.start = start
        request.goal = goal
        request.tolerance = 0.05

        done = threading.Event()
        result_holder = {'result': None, 'error': None}
        future = self.plan_client.call_async(request)

        def _finish(fut):
            try:
                result_holder['result'] = fut.result()
            except Exception as exc:
                result_holder['error'] = exc
            finally:
                done.set()

        future.add_done_callback(_finish)
        if not done.wait(timeout_s):
            self.get_logger().warn('Path planning request timed out.')
            return None
        if result_holder['error'] is not None:
            self.get_logger().warn(f'Path planning failed: {result_holder["error"]}')
            return None
        response = result_holder['result']
        if response is None or not response.plan.poses:
            return None
        return response.plan.poses

    def _path_metrics(self, poses):
        # Classification uses only the x and y coordinates of the smoothed path.
        points = [(float(p.pose.position.x), float(p.pose.position.y)) for p in poses]
        if len(points) < 2:
            return None

        lengths = []
        headings = []
        for a, b in zip(points[:-1], points[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue
            lengths.append(length)
            headings.append(math.atan2(dy, dx))

        if not lengths:
            return None

        path_length = float(sum(lengths))
        euclidean = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        turns = []
        for a, b in zip(headings[:-1], headings[1:]):
            delta = abs(math.atan2(math.sin(b - a), math.cos(b - a)))
            turns.append(math.degrees(delta))
        max_turn = max(turns, default=0.0)
        total_turn = sum(turns)
        sinuosity = path_length / max(euclidean, 1e-6)

        clearances = []
        if self.clearance_m is not None and self.map_msg is not None:
            info = self.map_msg.info
            height, width = self.clearance_m.shape
            for x, y in points:
                col = int((x - info.origin.position.x) / info.resolution)
                row = (height - 1) - int((y - info.origin.position.y) / info.resolution)
                if 0 <= row < height and 0 <= col < width:
                    clearances.append(float(self.clearance_m[row, col]))
                else:
                    clearances.append(0.0)
        min_clearance = min(clearances) if clearances else 0.0

        return {
            'points': points,
            'path_length': path_length,
            'euclidean': euclidean,
            'max_turn_deg': max_turn,
            'total_turn_deg': total_turn,
            'sinuosity': sinuosity,
            'min_clearance_m': min_clearance,
        }

    def _classify_path(self, metrics):
        if metrics is None:
            return None
        max_turn = metrics['max_turn_deg']
        clearance = metrics['min_clearance_m']
        sinuosity = metrics['sinuosity']
        total_turn = metrics['total_turn_deg']

        if max_turn >= 75.0 or clearance < 0.35 or sinuosity >= 1.45 or (total_turn >= 120.0 and max_turn >= 40.0):
            return 'v_complex'
        if max_turn >= 40.0 or clearance < 0.45 or sinuosity >= 1.20:
            return 'complex'
        if max_turn >= 15.0 or clearance < 0.60 or sinuosity >= 1.08 or total_turn >= 25.0:
            return 'semi_complex'
        return 'straight'

    def _sample_complex_goal(self, target_class):
        pose = self._robot_pose()
        if pose is None or not self.free_cells:
            return None
        rx, ry = pose
        candidates = list(self.free_cells)
        random.shuffle(candidates)
        for x, y in candidates[:min(500, len(candidates))]:
            distance = math.hypot(x - rx, y - ry)
            if distance < 0.5 or distance > self.max_euclidean_goal_distance:
                continue
            room = self._room_of(x, y)
            if room == self._room_of(rx, ry) and target_class in ('complex', 'v_complex'):
                continue
            return (x, y)
        return None

    def _find_goal_for_class(self, target_class):
        self.dbg.send(
            f'stage=candidate_search_started target={target_class} '
            f'attempts={self.candidate_attempts_per_class}',
            'WORKER_STAGE', 'info', -1.0)

        if not self.plan_client.wait_for_service(timeout_sec=2.0):
            status = f'stage=plan_service_unavailable service={self.plan_service_name}'
            self.get_logger().error(status)
            self.dbg.send(status, 'WORKER_STAGE', 'error', -1.0)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            return None, None

        pose = self._robot_pose()
        if pose is None:
            status = (
                f'stage=robot_pose_unavailable transform='
                f'{self.map_frame}->{self.base_frame}')
            self.get_logger().error(status)
            self.dbg.send(status, 'WORKER_STAGE', 'error', -1.0)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            return None, None

        rejected = {
            'no_candidate': 0,
            'no_pose': 0,
            'distance': 0,
            'no_path': 0,
            'bad_metrics': 0,
            'path_too_long': 0,
            'wrong_class': 0,
        }

        for candidate_index in range(self.candidate_attempts_per_class):
            if self.shutdown_flag:
                return None, None

            if candidate_index % 10 == 0:
                self.dbg.send(
                    f'stage=searching_candidates target={target_class} '
                    f'attempt={candidate_index + 1}/{self.candidate_attempts_per_class}',
                    'WORKER_STAGE', 'info', -1.0)

            if target_class in ('complex', 'v_complex') and random.random() < 0.7:
                candidate = self._sample_complex_goal(target_class)
            else:
                candidate = self._sample_goal_balanced()
            if candidate is None:
                rejected['no_candidate'] += 1
                continue

            pose = self._robot_pose()
            if pose is None:
                rejected['no_pose'] += 1
                continue

            euclidean = math.hypot(candidate[0] - pose[0], candidate[1] - pose[1])
            if euclidean > self.max_euclidean_goal_distance or euclidean <= 0.05:
                rejected['distance'] += 1
                continue

            poses = self._plan_path(candidate)
            if not poses:
                rejected['no_path'] += 1
                continue

            metrics = self._path_metrics(poses)
            if metrics is None:
                rejected['bad_metrics'] += 1
                continue

            if (metrics['path_length'] > self.max_path_length or
                    metrics['euclidean'] > self.max_euclidean_goal_distance):
                rejected['path_too_long'] += 1
                continue

            predicted = self._classify_path(metrics)
            if predicted != target_class:
                rejected['wrong_class'] += 1
                continue

            candidate_status = (
                f'target_class={target_class} candidate=accepted '
                f'euclidean={metrics["euclidean"]:.2f}m '
                f'path={metrics["path_length"]:.2f}m '
                f'max_turn={metrics["max_turn_deg"]:.1f}deg '
                f'total_turn={metrics["total_turn_deg"]:.1f}deg '
                f'clearance={metrics["min_clearance_m"]:.2f}m '
                f'sinuosity={metrics["sinuosity"]:.2f}'
            )
            self.get_logger().info(candidate_status)
            self.dbg.send(candidate_status, 'CANDIDATE', 'info', -1.0)
            self.dbg.send(
                f'stage=candidate_accepted target={target_class} '
                f'attempt={candidate_index + 1}',
                'WORKER_STAGE', 'info', -1.0)
            return candidate, metrics

        status = (
            f'stage=candidate_search_failed target={target_class} '
            f'attempts={self.candidate_attempts_per_class} '
            f'no_candidate={rejected["no_candidate"]} '
            f'no_pose={rejected["no_pose"]} '
            f'distance={rejected["distance"]} '
            f'no_path={rejected["no_path"]} '
            f'bad_metrics={rejected["bad_metrics"]} '
            f'path_too_long={rejected["path_too_long"]} '
            f'wrong_class={rejected["wrong_class"]}'
        )
        self.get_logger().warn(status)
        self.dbg.send(status, 'WORKER_STAGE', 'error', -1.0)
        self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
        return None, None

    def _sampling_worker(self):
        self.get_logger().info('Waiting for /map...')
        self.dbg.send('Waiting for /map...', 'SAMPLER_STATUS', 'warn', -1.0)
        self.dbg.send('stage=waiting_for_map', 'WORKER_STAGE', 'warn', -1.0)

        t0 = time.time()
        last_wait_report = -3.0
        while (
            self.map_msg is None
            and time.time() - t0 < self.map_wait_timeout_s
            and not self.shutdown_flag
        ):
            self._map_received_event.wait(timeout=0.5)
            elapsed = time.time() - t0
            if elapsed - last_wait_report >= 3.0:
                self.dbg.send(
                    f'stage=waiting_for_map elapsed={elapsed:.1f}s',
                    'WORKER_STAGE', 'warn', -1.0)
                last_wait_report = elapsed

        if self.map_msg is None:
            status = f'ERROR: No /map received within {self.map_wait_timeout_s:.1f}s'
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=map_wait_failed', 'WORKER_STAGE', 'error', -1.0)
            self.finished = True
            return

        map_info = self.map_msg.info
        self.dbg.send(
            f'stage=map_received size={map_info.width}x{map_info.height} '
            f'resolution={map_info.resolution:.3f}',
            'WORKER_STAGE', 'info', -1.0)

        try:
            self.dbg.send('stage=build_rooms_started', 'WORKER_STAGE', 'info', -1.0)
            self._build_rooms()
            self.dbg.send(
                f'stage=build_rooms_finished rooms={len(self.valid_rooms)} '
                f'free_cells={len(self.free_cells)}',
                'WORKER_STAGE', 'info', -1.0)
        except Exception as exc:
            status = f'ERROR: Room build failed: {exc}'
            self.get_logger().error(status)
            self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=build_rooms_failed', 'WORKER_STAGE', 'error', -1.0)
            self.finished = True
            return

        if not self.free_cells:
            self.get_logger().error('No traversable cells found in room map. Aborting.')
            self.dbg.send('ERROR: No free cells in map', 'SAMPLER_STATUS', 'error', -1.0)
            self.dbg.send('stage=no_free_cells', 'WORKER_STAGE', 'error', -1.0)
            self.finished = True
            return

        if self.use_path_pool and not self._load_path_pool():
            self.finished = True
            return

        self.dbg.send('stage=ready', 'WORKER_STAGE', 'info', -1.0)
        self.dbg.send(
            f'Ready to sample goals. rooms={len(self.valid_rooms)} '
            f'free_cells={len(self.free_cells)} pool_loaded={self.path_pool_loaded}',
            'SAMPLER_STATUS', 'info', -1.0)

        if self.teleport_at_start and not self.use_path_pool:
            self.dbg.send('stage=initial_spawn_selection', 'WORKER_STAGE', 'info', -1.0)
            spawn = self._sample_spawn_away_from(None)
            if spawn is None:
                self.dbg.send(
                    'stage=initial_spawn_unavailable', 'WORKER_STAGE', 'error', -1.0)
            teleport_ok = self._teleport_and_settle(spawn)
            self.dbg.send(
                f'stage=initial_teleport_finished success={str(teleport_ok).lower()}',
                'WORKER_STAGE', 'info' if teleport_ok else 'warn', -1.0)
            if not teleport_ok:
                self.get_logger().warn('Initial teleport failed; waiting for AMCL stabilization.')
                self.dbg.send('stage=fallback_wait_for_amcl', 'WORKER_STAGE', 'warn', -1.0)
                amcl_stable = self._wait_for_amcl_stable()
                self.dbg.send(
                    f'stage=fallback_amcl_wait_finished stable={str(amcl_stable).lower()}',
                    'WORKER_STAGE', 'info' if amcl_stable else 'warn', -1.0)
        elif not self.use_path_pool:
            self.dbg.send('stage=waiting_for_amcl_stability', 'WORKER_STAGE', 'info', -1.0)
            amcl_stable = self._wait_for_amcl_stable()
            self.dbg.send(
                f'stage=amcl_wait_finished stable={str(amcl_stable).lower()}',
                'WORKER_STAGE', 'info' if amcl_stable else 'warn', -1.0)
        else:
            self.dbg.send('stage=pool_mode_initial_teleport_deferred', 'WORKER_STAGE', 'info', -1.0)

        pose = self._robot_pose()
        if pose is None and not self.use_path_pool:
            self.dbg.send(
                f'stage=robot_pose_unavailable_after_start transform={self.map_frame}->{self.base_frame}',
                'WORKER_STAGE', 'error', -1.0)
            self.dbg.send(
                'Sampler cannot start goal search because TF map->base_link is unavailable.',
                'SAMPLER_STATUS', 'error', -1.0)
            self.finished = True
            return

        if pose is not None:
            self.dbg.send(
                f'stage=robot_pose_ready x={pose[0]:.2f} y={pose[1]:.2f}',
                'WORKER_STAGE', 'info', -1.0)
        elif self.use_path_pool:
            self.dbg.send(
                'stage=robot_pose_deferred_until_pool_path_teleport',
                'WORKER_STAGE', 'info', -1.0)
        self._publish_class_counts()
        completed = 0
        attempts_without_candidate = 0

        while rclpy.ok() and not self.shutdown_flag and completed < self.num_goals:
            self._sampling_debug_state()
            target_class = self._next_target_class()
            if target_class is None:
                self.dbg.send('stage=class_quotas_complete', 'WORKER_STAGE', 'info', -1.0)
                break

            self.dbg.send(
                f'stage=selecting_target_class target={target_class} completed={completed}/{self.num_goals}',
                'WORKER_STAGE', 'info', -1.0)

            pool_path = None
            pool_spawn = None
            if self.use_path_pool:
                pool_path = self._take_pool_path(target_class)
                if pool_path is None:
                    status = (
                        f'PATH_POOL_ERROR exhausted target={target_class} '
                        f'completed={completed}/{self.num_goals} '
                        f'counts={self.class_counts}')
                    self.get_logger().error(status)
                    self.dbg.send(status, 'SAMPLER_STATUS', 'error', -1.0)
                    self.dbg.send(
                        f'stage=pool_exhausted target={target_class}',
                        'WORKER_STAGE', 'error', -1.0)
                    break

                display_class = 'semi' if target_class == 'semi_complex' else target_class
                self.dbg.send(
                    f'current path is: {display_class}', 'CLASS', 'info', -1.0)
                path_points = pool_path['poses']
                dx = path_points[1][0] - path_points[0][0]
                dy = path_points[1][1] - path_points[0][1]
                yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else 0.0
                pool_spawn = (pool_path['start'][0], pool_path['start'][1], yaw)
                self.dbg.send(
                    f'stage=pool_path_selected id={pool_path["id"]} target={target_class} '
                    f'remaining_in_class={len(self.path_pool_by_class[target_class])} '
                    f'start=({pool_path["start"][0]:.2f},{pool_path["start"][1]:.2f}) '
                    f'goal=({pool_path["goal"][0]:.2f},{pool_path["goal"][1]:.2f}) '
                    f'length={float(pool_path["metrics"]["path_length_m"]):.2f}m '
                    f'clearance={float(pool_path["metrics"]["min_clearance_m"]):.2f}m',
                    'WORKER_STAGE', 'info', -1.0)
                if not self._teleport_and_settle(pool_spawn):
                    attempts_without_candidate += 1
                    status = (
                        f'POOL_PATH_REJECTED id={pool_path["id"]} '
                        f'reason=teleport_failed retries={attempts_without_candidate}')
                    self.get_logger().warn(status)
                    self.dbg.send(status, 'SAMPLER_STATUS', 'warn', -1.0)
                    if attempts_without_candidate >= self.pool_max_retries:
                        self.dbg.send(
                            'stage=pool_teleport_retries_exhausted',
                            'WORKER_STAGE', 'error', -1.0)
                        break
                    continue
                goal = tuple(pool_path['goal'])
                metrics = pool_path['metrics']
                attempts_without_candidate = 0
            else:
                goal, metrics = self._find_goal_for_class(target_class)
                if goal is None:
                    attempts_without_candidate += 1
                    self.dbg.send(
                        f'target={target_class} no_valid_candidate retries={attempts_without_candidate}',
                        'SAMPLER_STATUS', 'warn', 5.0)
                    if attempts_without_candidate >= 3:
                        self.get_logger().error(
                            f'Unable to fill class {target_class}. Current counts: {self.class_counts}')
                        break
                    spawn = self._sample_spawn_away_from(None)
                    if spawn is not None:
                        self._teleport_and_settle(spawn)
                    continue
                attempts_without_candidate = 0
                display_class = 'semi' if target_class == 'semi_complex' else target_class
                self.dbg.send(
                    f'current path is: {display_class}', 'CLASS', 'info', -1.0)

            goal_room = self._room_of(goal[0], goal[1])
            self.dbg.send(
                f'stage=navigating_to_goal target={target_class} '
                f'x={goal[0]:.2f} y={goal[1]:.2f}',
                'WORKER_STAGE', 'info', -1.0)
            status = 'failed'
            elapsed = 0.0
            for attempt in range(1, int(self.max_attempts_per_goal) + 1):
                if self.use_path_pool and attempt > 1 and pool_spawn is not None:
                    if not self._teleport_and_settle(pool_spawn):
                        status = 'failed'
                        elapsed = 0.0
                        continue
                status, elapsed = self._run_goal(goal, completed, attempt)
                if status == 'success':
                    break
                if status == 'collision':
                    self._stop_robot_hard()
                    spawn = self._sample_spawn_away_from(goal)
                    if spawn is not None:
                        self._teleport_and_settle(spawn)
                    break
                if status in ('timeout', 'failed', 'fast_fail'):
                    time.sleep(self.min_goal_interval_s)

            if status == 'success':
                with self._class_lock:
                    self.class_counts[target_class] += 1
                    counts = dict(self.class_counts)
                completed += 1
                self._completed_goals = completed
                self.recent_goals.append(goal)
                self.recent_goal_rooms.append(goal_room)
                self.recent_zones.append(self._zone_key(goal[0], goal[1]))
                self.dbg.send(f'completed={completed}/{self.num_goals}', 'TOTALS', 'info', -1.0)
                self._publish_class_counts()
                self.dbg.send(
                    f'accepted={target_class} completed={completed}/{self.num_goals} '
                    f'straight={counts["straight"]} semi={counts["semi_complex"]} '
                    f'complex={counts["complex"]} v_complex={counts["v_complex"]}',
                    'SAMPLER_STATUS', 'info', 5.0)
                if not self.use_path_pool and self._goals_same_room():
                    spawn = self._sample_spawn_away_from(None)
                    if spawn is not None:
                        self._teleport_and_settle(spawn)
                time.sleep(self.min_goal_interval_s)
            else:
                self.pub_goal_success.publish(Bool(data=False))
                self.pub_active.publish(Bool(data=False))
                self.get_logger().warn(
                    f'Goal rejected for target class {target_class}: status={status}. Counts unchanged.')
                time.sleep(self.goal_pause_s)

        self._publish_class_counts()
        with self._class_lock:
            final_counts = dict(self.class_counts)
        if completed >= self.num_goals:
            self.get_logger().info('Sampler finished. All requested goals collected.')
            self.dbg.send('All requested goals collected.', 'SAMPLER_STATUS', 'info', -1.0)
        else:
            self.get_logger().error(
                f'Sampler stopped before target: completed={completed}/{self.num_goals}, classes={final_counts}')
            self.dbg.send(
                f'Incomplete dataset completed={completed}/{self.num_goals} '
                f'straight={final_counts["straight"]} semi={final_counts["semi_complex"]} '
                f'complex={final_counts["complex"]} v_complex={final_counts["v_complex"]}',
                'SAMPLER_STATUS', 'error', -1.0)
            self.get_logger().error('Sampler terminated incomplete; see Incomplete dataset status above.')
        self.finished = True

    def _distance_to_goal(self, goal_xy):
        pose = self._robot_pose()
        if pose is None:
            return None
        return math.hypot(float(goal_xy[0]) - pose[0], float(goal_xy[1]) - pose[1])

    def _run_goal(self, goal_xy, slot_idx, attempt):
        self.collision_event.clear()

        xg, yg = goal_xy

        goal_msg = PointStamped()
        goal_msg.header.frame_id = self.map_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.point.x = xg
        goal_msg.point.y = yg
        goal_msg.point.z = 0.0
        self.pub_goal.publish(goal_msg)

        self.episode_id += 1
        self.pub_episode.publish(Int32(data=self.episode_id))
        self.pub_active.publish(Bool(data=True))

        t0 = time.time()

        if self.navigation_backend == 'classic':
            self.nav_status = None
            status = None
            while not self.shutdown_flag:
                if self.collision_event.is_set():
                    self._stop_robot_hard()
                    status = 'collision'
                    break
                distance_to_goal = self._distance_to_goal(goal_xy)
                if (distance_to_goal is not None and
                        distance_to_goal < self.goal_position_tolerance):
                    self._stop_robot_hard()
                    status = 'success'
                    break
                if self.nav_status == GoalStatus.STATUS_SUCCEEDED:
                    final_distance = self._distance_to_goal(goal_xy)
                    if (final_distance is not None and
                            final_distance < self.goal_position_tolerance):
                        self._stop_robot_hard()
                        status = 'success'
                    else:
                        self.get_logger().warn(
                            'Classic planner finished outside the goal position tolerance. ' +
                            'The goal will not be counted as successful.')
                        status = 'failed'
                    break
                if self.nav_status == GoalStatus.STATUS_ABORTED:
                    status = 'failed'
                    break
                if time.time() - t0 > self.goal_timeout_s:
                    self.get_logger().warn(f'SAMPLE {slot_idx + 1:02d} timeout (classic).')
                    status = 'timeout'
                    break
                time.sleep(0.05)

            elapsed = time.time() - t0
            if status is None:
                status = 'failed'

            self.get_logger().info(
                f'SAMPLE {slot_idx + 1:02d} attempt {attempt} -> {status} ({elapsed:.2f} s)')

            if status == 'success':
                self.pub_goal_success.publish(Bool(data=True))
            else:
                self.pub_goal_success.publish(Bool(data=False))
            self.pub_active.publish(Bool(data=False))
            return (status, elapsed)

        nav_goal = PoseStamped()
        nav_goal.header.frame_id = self.map_frame
        nav_goal.header.stamp = self.get_clock().now().to_msg()
        nav_goal.pose.position.x = xg
        nav_goal.pose.position.y = yg
        nav_goal.pose.orientation.w = 1.0
        self.navigator.goToPose(nav_goal)

        status = None
        while not self.navigator.isTaskComplete():
            if self.shutdown_flag:
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass
                return ('failed', time.time() - t0)
            if self.collision_event.is_set():
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass
                self._stop_robot_hard()
                status = 'collision'
                break
            distance_to_goal = self._distance_to_goal(goal_xy)
            if (distance_to_goal is not None and
                    distance_to_goal < self.goal_position_tolerance):
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass
                self._stop_robot_hard()
                status = 'success'
                break
            if time.time() - t0 > self.goal_timeout_s:
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass
                self.get_logger().warn(f'SAMPLE {slot_idx + 1:02d} timeout.')
                status = 'timeout'
                break
            time.sleep(0.05)

        elapsed = time.time() - t0

        if status is None:
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                final_distance = self._distance_to_goal(goal_xy)
                if (final_distance is not None and
                        final_distance < self.goal_position_tolerance):
                    status = 'success'
                else:
                    self.get_logger().warn(
                        'Nav2 finished outside the goal position tolerance. ' +
                        'The goal will not be counted as successful.')
                    status = 'failed'
            elif elapsed < self.min_attempt_s:
                status = 'fast_fail'
            else:
                status = 'failed'

        self.get_logger().info(
            f'SAMPLE {slot_idx + 1:02d} attempt {attempt} -> {status} ({elapsed:.2f} s)')

        try:
            if status == 'success':
                self.pub_goal_success.publish(Bool(data=True))
            else:
                self.pub_goal_success.publish(Bool(data=False))
            self.pub_active.publish(Bool(data=False))
        except Exception:
            pass

        return (status, elapsed)


def main():
    rclpy.init()
    node = RandomGoalSampler()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_flag = True
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
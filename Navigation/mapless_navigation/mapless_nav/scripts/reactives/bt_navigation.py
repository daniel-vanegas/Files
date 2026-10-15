#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int8
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
import copy
import time
import py_trees as pt
import py_trees_ros as ptr

from debugger import DebugClient

# ----------------------------------------------------------------------
# Helper functions (exactly as in the original SMACH)
# ----------------------------------------------------------------------
def zero_runs(a):
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    return np.where(absdiff == 1)[0].reshape(-1, 2)

def free_side(laser_l, laser_r, obst_l, obst_r, node):
    fl = len(laser_l) * 11
    fr = len(laser_r) * 11
    l_sum = laser_l.sum()
    r_sum = laser_r.sum()
    free_l = (l_sum == fl) or np.all(laser_l > 1.0)
    free_r = (r_sum == fr) or np.all(laser_r > 1.0)
    dense_l = obst_l.sum()
    dense_r = obst_r.sum()

    node.obst_detected[0] = (not free_l) or (dense_l >= len(laser_l))
    node.obst_detected[1] = (not free_r) or (dense_r >= len(laser_r))

    if free_l and free_r:
        best = 'R' if (l_sum + dense_l) >= (r_sum + dense_r) else 'L'
    elif free_l and dense_l < len(laser_r):
        best = 'L'
    elif free_r and dense_r < len(laser_r):
        best = 'R'
    else:
        best = 'B'
    return best

def laser_side(node):
    laser_l, laser_r, laser_c = node.feat.get_laser()
    free_l = zero_runs(np.where(laser_l > 1, 0, 1))
    max_l_range = [0, 0]
    for rng in free_l:
        n = rng[1] - rng[0]
        if n > len(laser_c):
            if n > (max_l_range[1] - max_l_range[0]):
                max_l_range = rng
    l_range = max_l_range[1] - max_l_range[0]

    free_r = zero_runs(np.where(laser_r > 1, 0, 1))
    max_r_range = [0, 0]
    for rng in free_r:
        n = rng[1] - rng[0]
        if n > len(laser_c):
            if n > (max_r_range[1] - max_r_range[0]):
                max_r_range = rng
    r_range = max_r_range[1] - max_r_range[0]

    if l_range == 0 and r_range == 0:
        return 'B'
    return 'L' if l_range >= r_range else 'R'

def turn_direction(node):
    if node.last_goal[1] > 0.2:
        return 'L'
    if node.last_goal[1] < -0.2:
        return 'R'
    return 'F'

# ----------------------------------------------------------------------
# Features class (same as original)
# ----------------------------------------------------------------------
class Features:
    def __init__(self):
        self.left_obst = []
        self.right_obst = []
        self.cent_obst = []
        self.laser_obst_left = []
        self.laser_obst_right = []
        self.laser_obst_cent = []

    def set_features(self, left, right, center):
        self.left_obst = left
        self.right_obst = right
        self.cent_obst = center

    def get_features(self):
        return self.left_obst, self.right_obst, self.cent_obst

    def set_laser(self, left, right, center):
        self.laser_obst_left = left
        self.laser_obst_right = right
        self.laser_obst_cent = center

    def get_laser(self):
        return self.laser_obst_left, self.laser_obst_right, self.laser_obst_cent

# ----------------------------------------------------------------------
# ROS2 Node that holds the behavior tree
# ----------------------------------------------------------------------
class BTNavigationNode(Node):
    def __init__(self):
        super().__init__('bt_navigation')
        self.dbg = DebugClient(self, verbose=True)
        self.dbg.send("BT Navigation node started.", msg_id="NODE_STATUS")

        # ---- State variables (same as original globals) ----
        self.last_goal = [0.0, 0.0]          # distance, angle
        self.target_reached = True
        self.obst_detected = [False, False, False]  # L, R, C
        self.obst_memory = [False, False, False]
        self.interrupt = False
        self.speed_factor = 0.8

        # ---- Publishers ----
        # Use correct topic (from topic list: /hardware/base/cmd_vel)
        self.cmd_vel_pub = self.create_publisher(Twist, '/hardware/base/cmd_vel', 10)
        self.head_pub = self.create_publisher(Float64MultiArray, '/hardware/head/goal_pose', 1)
        self.goal_stat_pub = self.create_publisher(Int8, '/maples_nav/goal_reached', 1)

        # ---- Subscribers ----
        self.create_subscription(OccupancyGrid, '/local_occ_grid', self.occ_grid_callback, 10)
        self.create_subscription(LaserScan, '/hardware/base/scan', self.laser_callback, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.clicked_point_callback, 10)
        self.create_subscription(Float32MultiArray, '/NN_goal', self.nn_goal_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.move_goal_callback, 10)

        # ---- Head pose ----
        head_cmd = Float64MultiArray()
        head_cmd.data = [0.0, -0.4]
        self.head_pub.publish(head_cmd)

        # ---- Features object ----
        self.feat = Features()

        # ---- Simple base movement helper ----
        class SimpleBase:
            def __init__(self, node):
                self.node = node
            def move_vel(self, lx, ly, az):
                cmd = Twist()
                cmd.linear.x = lx
                cmd.linear.y = ly
                cmd.angular.z = az
                self.node.cmd_vel_pub.publish(cmd)
        self.base = SimpleBase(self)

        # ---- Build the behavior tree ----
        self.root = self.create_tree()
        self.tree = ptr.trees.BehaviourTree(self.root)

        # Tree tick rate (10 Hz)
        self.timer = self.create_timer(0.1, self.tick_tree)

        # Heartbeat (5 sec)
        self.create_timer(5.0, self.heartbeat_callback)

        self.get_logger().info("Behavior tree navigation ready")

    def heartbeat_callback(self):
        self.dbg.send("Node alive (heartbeat)", msg_id="NODE_STATUS")

    def tick_tree(self):
        self.tree.tick()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def occ_grid_callback(self, msg):
        # Feature extraction (as in original)
        data = np.asarray(msg.data, dtype=np.int8)
        res = round(msg.info.resolution, 2)
        rows = msg.info.height
        data = data.reshape((rows, rows))
        data = np.rot90(np.flip(data, axis=0))
        c = rows // 2
        near_obst_dist = int(1 / res)

        data_L = data[c:, :c-6]
        data_L2 = data_L.sum(axis=0) // 100 if data_L.size else np.array([])
        data_L2 = data_L2[len(data_L2)//2:] if len(data_L2) > 0 else np.array([])

        data_R = data[c:, c+6:]
        data_R2 = data_R.sum(axis=0) // 100 if data_R.size else np.array([])
        data_R2 = data_R2[:len(data_R2)//2] if len(data_R2) > 0 else np.array([])

        data_C = data[-near_obst_dist:, c-6:c+6]
        data_C2 = data_C.sum(axis=0) // 100 if data_C.size else np.array([])

        self.feat.set_features([data_L, data_L2], [data_R, data_R2], [data_C, data_C2])

    def laser_callback(self, msg):
        n = len(msg.ranges)
        c = n // 2 + 1
        left = np.clip(msg.ranges[c+25:], 0, 11)
        right = np.clip(msg.ranges[:c-25], 0, 11)
        center = np.clip(msg.ranges[c-17:c+17], 0, 11)
        self.feat.set_laser(left, right, center)

    def clicked_point_callback(self, msg):
        self.get_logger().info(f"Clicked point: ({msg.point.x:.2f}, {msg.point.y:.2f})")

    def nn_goal_callback(self, msg):
        if len(msg.data) < 2:
            return
        distance = msg.data[0]
        angle = msg.data[1]
        if distance < 0.01:
            return
        self.last_goal = [distance, angle]
        self.target_reached = False
        self.interrupt = False
        self.dbg.send(f"Goal: {distance:.2f}m, {angle:.2f}rad", msg_id="NAV_GOAL")

    def move_goal_callback(self, msg):
        point = PointStamped()
        point.header = msg.header
        point.point = msg.pose.position
        self.clicked_point_callback(point)

    # ------------------------------------------------------------------
    # Behavior tree creation
    # ------------------------------------------------------------------
    def create_tree(self):
        root = pt.composites.Sequence("Root", memory=False)
        root.add_children([
            WaitForGoal(self),
            pt.decorators.Repeat(
                "NavigateLoop",
                child=self.navigate_until_goal(),
                num_success=1   # stop after one successful completion (goal reached)
            )
        ])
        return root

    def navigate_until_goal(self):
        """Sequence that runs navigation steps and returns SUCCESS only when goal reached."""
        return pt.composites.Sequence("NavigateUntilGoal", memory=False, children=[
            self.check_goal_reached(),   # if reached -> SUCCESS, else RUNNING
            Evaluate(self),
            pt.composites.Selector("ObstacleOrFree", memory=False, children=[
                EvadeObstacle(self),
                self.turn_and_move()
            ])
        ])

    def check_goal_reached(self):
        class CheckGoalReached(pt.behaviour.Behaviour):
            def __init__(self, node, tol=0.5):
                super().__init__("CheckGoalReached")
                self.node = node
                self.tol = tol
            def update(self):
                if self.node.last_goal[0] < self.tol:
                    self.node.get_logger().info("Goal reached!")
                    self.node.base.move_vel(0.0, 0.0, 0.0)
                    status = Int8()
                    status.data = 3
                    self.node.goal_stat_pub.publish(status)
                    self.node.target_reached = True
                    return pt.common.Status.SUCCESS
                return pt.common.Status.RUNNING
        return CheckGoalReached(self)

    def turn_and_move(self):
        seq = pt.composites.Sequence("TurnThenMove", memory=True)
        turn = TurnObjective(self)
        move_selector = pt.composites.Selector("MoveSelector", memory=False)
        move_selector.add_children([
            MoveLeft(self),
            MoveRight(self),
            MoveForward(self)
        ])
        seq.add_children([turn, move_selector])
        return seq

# ----------------------------------------------------------------------
# Behaviour classes (each corresponds to one state in the original SMACH)
# ----------------------------------------------------------------------
class WaitForGoal(pt.behaviour.Behaviour):
    """Wait until a new goal is published (target_reached becomes False)."""
    def __init__(self, node):
        super().__init__("WaitForGoal")
        self.node = node

    def update(self):
        if not self.node.target_reached:
            return pt.common.Status.SUCCESS
        return pt.common.Status.RUNNING

class Evaluate(pt.behaviour.Behaviour):
    """Determines if obstacle is in front, and sets 'evasion' direction."""
    def __init__(self, node):
        super().__init__("Evaluate")
        self.node = node
        self.tries = 0

    def update(self):
        if self.tries == 0:
            self.node.get_logger().info("--> BEHAVIOUR: Evaluate")

        if self.node.last_goal[0] < 0.5 or self.node.interrupt:
            self.node.get_logger().warn(">>>>> STOPPED <<<<<")
            self.node.base.move_vel(0.0, 0.0, 0.0)
            self.tries = 0
            self.node.target_reached = True
            self.node.goal_stat_pub.publish(Int8(data=3))
            return pt.common.Status.FAILURE

        self.tries += 1
        try:
            obst_l, obst_r, obst_c = self.node.feat.get_features()
            # Safety: if no occupancy grid data, treat as free path
            if len(obst_c) < 2 or obst_c[1] is None or obst_c[1].size == 0:
                self.tries = 0
                bb = pt.blackboard.Blackboard()
                bb.evasion = '-'
                self.node.get_logger().info("Evaluate: no grid data -> free path")
                return pt.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"Evaluate error: {e}")
            self.tries = 0
            bb = pt.blackboard.Blackboard()
            bb.evasion = '-'
            return pt.common.Status.SUCCESS

        laser_l, laser_r, _ = self.node.feat.get_laser()
        bb = pt.blackboard.Blackboard()
        bb.command_time = 300

        if obst_c[1].sum() == 0 and self.node.last_goal[0] > 0.5:
            self.tries = 0
            bb.evasion = '-'
            self.node.get_logger().info("Evaluate: free path")
            return pt.common.Status.SUCCESS
        elif obst_c[1].sum() > 0:
            decision = free_side(laser_l, laser_r, obst_l[1], obst_r[1], self.node)
            bb.evasion = decision
            self.node.obst_detected[2] = True
            self.node.obst_memory = copy.deepcopy(self.node.obst_detected)
            self.tries = 0
            self.node.get_logger().info(f"Evaluate: obstacle detected, evasion={decision}")
            return pt.common.Status.SUCCESS
        else:
            self.node.base.move_vel(0.0, 0.0, 0.0)
            return pt.common.Status.RUNNING

class EvadeObstacle(pt.behaviour.Behaviour):
    def __init__(self, node):
        super().__init__("EvadeObstacle")
        self.node = node
        self.timer = None
        self.phase = 0

    def update(self):
        if self.timer is None:
            self.node.get_logger().info("--> BEHAVIOUR: EvadeObstacle")
            bb = pt.blackboard.Blackboard()
            bb.command_time = 180
            obst_l, obst_r, obst_c = self.node.feat.get_features()
            laser_l, laser_r, _ = self.node.feat.get_laser()
            self.node.obst_detected[2] = (len(obst_c) > 1 and obst_c[1].sum() != 0)
            free_side(laser_l, laser_r, obst_l[1], obst_r[1], self.node)
            lid_side = laser_side(self.node)
            evasion = bb.evasion

            # If no obstacle, return FAILURE so selector falls back to turn_and_move
            if evasion == '-':
                self.node.get_logger().info("EvadeObstacle: free path, skipping")
                return pt.common.Status.FAILURE

            self.node.get_logger().info(f"EVASION: {evasion}, lidar_side: {lid_side}")

            if evasion == 'L':
                speed = 0.4
                index = 0
                if lid_side == 'L':
                    speed = 0.6
                elif lid_side == 'R':
                    speed = -0.5
            else:  # evasion == 'R'
                speed = -0.4
                index = 1
                if lid_side == 'L':
                    speed = 0.5
                elif lid_side == 'R':
                    speed = -0.6

            self.evasion_data = {'speed': speed, 'index': index}
            self.phase = 1
            self.timer = time.time()
            return pt.common.Status.RUNNING

        now = time.time()
        data = self.evasion_data
        obst_l, obst_r, obst_c = self.node.feat.get_features()
        # Abort if data missing
        if len(obst_c) < 2:
            self.cleanup()
            return pt.common.Status.FAILURE

        if self.phase == 1:
            if obst_c[1].sum() > 0:
                self.node.base.move_vel(0.0, 0.0, data['speed'] * self.node.speed_factor)
                return pt.common.Status.RUNNING
            else:
                self.phase = 2
                self.timer = now
                self.node.get_logger().info("Evade: front free, moving forward")
                self.node.base.move_vel(0.5 * self.node.speed_factor, 0.0, 0.0)
                return pt.common.Status.RUNNING

        elif self.phase == 2:
            if now - self.timer < 1.0:
                self.node.base.move_vel(0.5 * self.node.speed_factor, 0.0, 0.0)
                return pt.common.Status.RUNNING
            else:
                # Update features after moving
                obst_l, obst_r, obst_c = self.node.feat.get_features()
                laser_l, laser_r, _ = self.node.feat.get_laser()
                self.node.obst_detected[2] = (len(obst_c) > 1 and obst_c[1].sum() != 0)
                free_side(laser_l, laser_r, obst_l[1], obst_r[1], self.node)

                if self.node.last_goal[0] < 1.2:
                    bb = pt.blackboard.Blackboard()
                    bb.evasion = '-'
                    self.node.get_logger().info("Evade: goal close, switching to free")
                    self.cleanup()
                    return pt.common.Status.SUCCESS

                if not self.node.obst_detected[data['index']]:
                    self.node.get_logger().info("Evade: side clear, success")
                    self.cleanup()
                    return pt.common.Status.SUCCESS

                # Otherwise backup and retry
                self.phase = 3
                self.timer = now
                self.node.base.move_vel(-0.5 * self.node.speed_factor, 0.0, 0.0)
                self.node.get_logger().info("Evade: backing up, retry")
                return pt.common.Status.RUNNING

        elif self.phase == 3:
            if now - self.timer < 0.5:
                self.node.base.move_vel(-0.5 * self.node.speed_factor, 0.0, 0.0)
                return pt.common.Status.RUNNING
            else:
                self.cleanup()
                return pt.common.Status.RUNNING   # try again

        return pt.common.Status.SUCCESS

    def cleanup(self):
        self.node.base.move_vel(0.0, 0.0, 0.0)
        self.timer = None
        self.phase = 0
        self.evasion_data = None

class TurnObjective(pt.behaviour.Behaviour):
    def __init__(self, node):
        super().__init__("TurnObjective")
        self.node = node

    def update(self):
        self.node.get_logger().info("--> BEHAVIOUR: TurnObjective")
        bb = pt.blackboard.Blackboard()
        bb.command_time = 400
        dir = turn_direction(self.node)
        bb.turn_direction = dir
        return pt.common.Status.SUCCESS

class MoveLeft(pt.behaviour.Behaviour):
    def __init__(self, node):
        super().__init__("MoveLeft")
        self.node = node
        self.timer = None

    def update(self):
        if self.timer is None:
            self.node.get_logger().info("--> BEHAVIOUR: MoveLeft")
            bb = pt.blackboard.Blackboard()
            if bb.turn_direction != 'L':
                return pt.common.Status.FAILURE
            self.timer = time.time()
            self.node.base.move_vel(0.0, 0.0, 0.5 * self.node.speed_factor)
            return pt.common.Status.RUNNING

        bb = pt.blackboard.Blackboard()
        elapsed = (time.time() - self.timer) / 0.1
        if elapsed < bb.command_time:
            self.node.base.move_vel(0.0, 0.0, 0.5 * self.node.speed_factor)
            return pt.common.Status.RUNNING
        else:
            self.node.base.move_vel(0.0, 0.0, 0.0)
            self.timer = None
            if bb.evasion != '-':
                bb.command_time = 400
                return pt.common.Status.FAILURE   # 'F' transition
            return pt.common.Status.SUCCESS

class MoveRight(pt.behaviour.Behaviour):
    def __init__(self, node):
        super().__init__("MoveRight")
        self.node = node
        self.timer = None

    def update(self):
        if self.timer is None:
            self.node.get_logger().info("--> BEHAVIOUR: MoveRight")
            bb = pt.blackboard.Blackboard()
            if bb.turn_direction != 'R':
                return pt.common.Status.FAILURE
            self.timer = time.time()
            self.node.base.move_vel(0.0, 0.0, -0.5 * self.node.speed_factor)
            return pt.common.Status.RUNNING

        bb = pt.blackboard.Blackboard()
        elapsed = (time.time() - self.timer) / 0.1
        if elapsed < bb.command_time:
            self.node.base.move_vel(0.0, 0.0, -0.5 * self.node.speed_factor)
            return pt.common.Status.RUNNING
        else:
            self.node.base.move_vel(0.0, 0.0, 0.0)
            self.timer = None
            if bb.evasion != '-':
                bb.command_time = 400
                return pt.common.Status.FAILURE
            return pt.common.Status.SUCCESS

class MoveForward(pt.behaviour.Behaviour):
    def __init__(self, node):
        super().__init__("MoveForward")
        self.node = node
        self.timer = None

    def update(self):
        if self.timer is None:
            self.node.get_logger().info("--> BEHAVIOUR: MoveForward")
            bb = pt.blackboard.Blackboard()
            self.timer = time.time()
            self.node.base.move_vel(0.8 * self.node.speed_factor, 0.0, 0.0)
            return pt.common.Status.RUNNING

        bb = pt.blackboard.Blackboard()
        elapsed = (time.time() - self.timer) / 0.1
        if elapsed < bb.command_time:
            self.node.base.move_vel(0.8 * self.node.speed_factor, 0.0, 0.0)
            return pt.common.Status.RUNNING
        else:
            self.node.base.move_vel(0.0, 0.0, 0.0)
            self.timer = None
            if bb.evasion != '-':
                bb.command_time = 300
                return pt.common.Status.FAILURE   # 'obstacle' transition
            return pt.common.Status.SUCCESS

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = BTNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.base.move_vel(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
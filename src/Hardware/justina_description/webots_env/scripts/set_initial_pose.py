#!/usr/bin/env python3
import rclpy
import argparse
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class InitialPosePublisher(Node):
    def __init__(self, x, y, yaw):
        super().__init__('set_initial_pose')
        self.x = x
        self.y = y
        self.yaw = yaw
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/navigation/initialpose', 10)
        self.timer = self.create_timer(1.0, self.publish_initial_pose)

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        # Convert yaw (radians) to quaternion
        half_yaw = self.yaw / 2.0
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)
        # Simple covariance (diagonal 0.25)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.25
        self.pub.publish(msg)
        self.get_logger().info(f'Initial pose published (x={self.x}, y={self.y}, yaw={self.yaw:.2f} rad)')
        self.timer.cancel()
        self.destroy_node()
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Publish initial pose for AMCL')
    parser.add_argument('--x', type=float, default=7.0, help='X position in map frame')
    parser.add_argument('--y', type=float, default=3.0, help='Y position in map frame')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw angle in radians')
    # Parse only known arguments; ignore extra ROS arguments
    args, unknown = parser.parse_known_args()

    rclpy.init()  # rclpy will take care of ROS args from sys.argv
    node = InitialPosePublisher(args.x, args.y, args.yaw)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
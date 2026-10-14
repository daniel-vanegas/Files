#!/usr/bin/env python3
"""
Relay /map -> /navigation/augmented_map preserving TRANSIENT_LOCAL
durability so late subscribers (like RViz) still get the map.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile,
                       QoSDurabilityPolicy,
                       QoSReliabilityPolicy,
                       QoSHistoryPolicy)
from nav_msgs.msg import OccupancyGrid


class MapRelay(Node):

    def __init__(self):
        super().__init__('map_relay')

        self.declare_parameter('in_topic', '/map')
        self.declare_parameter('out_topic', '/navigation/augmented_map')

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value

        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos)
        self.sub = self.create_subscription(
            OccupancyGrid, in_topic, self._on_map, qos)

        self.get_logger().info(f'Relay {in_topic} -> {out_topic}')

    def _on_map(self, msg):
        self.pub.publish(msg)


def main():
    rclpy.init()
    try:
        node = MapRelay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

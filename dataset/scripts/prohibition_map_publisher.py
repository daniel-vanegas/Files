#!/usr/bin/env python3
import os
import yaml
import numpy as np

from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile,
                       QoSDurabilityPolicy,
                       QoSReliabilityPolicy,
                       QoSHistoryPolicy)
from nav_msgs.msg import OccupancyGrid


class ProhibitionMapPublisher(Node):

    def __init__(self):
        super().__init__('prohibition_map_publisher')

        self.declare_parameter('map_yaml', '')
        self.declare_parameter('topic_name', '/navigation/prohibition_map')
        self.declare_parameter('frame_id', 'map')

        map_yaml = self.get_parameter('map_yaml').value
        topic_name = self.get_parameter('topic_name').value
        self.frame_id = self.get_parameter('frame_id').value

        if not map_yaml or not os.path.isfile(map_yaml):
            self.get_logger().error(
                f'Map yaml not found: "{map_yaml}"')
            return

        with open(map_yaml) as f:
            meta = yaml.safe_load(f)

        res = float(meta['resolution'])
        origin = meta['origin']
        img_name = meta['image']
        negate = int(meta.get('negate', 0))
        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.196))

        map_dir = os.path.dirname(map_yaml)
        img_path = (img_name if os.path.isabs(img_name)
                    else os.path.join(map_dir, img_name))

        if not os.path.isfile(img_path):
            self.get_logger().error(f'Map image not found: "{img_path}"')
            return

        img = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
        h, w = img.shape

        if negate:
            occ_prob = img.astype(np.float32) / 255.0
        else:
            occ_prob = (255.0 - img.astype(np.float32)) / 255.0

        data = np.full((h, w), -1, dtype=np.int8)
        data[occ_prob > occupied_thresh] = 100
        data[occ_prob < free_thresh] = 0

        data = np.flipud(data)

        self.grid = OccupancyGrid()
        self.grid.header.frame_id = self.frame_id
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.grid.info.resolution = res
        self.grid.info.width = w
        self.grid.info.height = h
        self.grid.info.origin.position.x = float(origin[0])
        self.grid.info.origin.position.y = float(origin[1])
        self.grid.info.origin.position.z = 0.0
        # self.grid.info.origin.orientation.w = 1.0
        self.grid.data = data.flatten().tolist()

        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(OccupancyGrid, topic_name, qos)

        # Republish periodically so late subscribers receive the map.
        self.create_timer(3.0, self._republish)

        self.get_logger().info(
            f'Publishing prohibition map on {topic_name} '
            f'from {img_path} ({w}x{h} @ {res} m/cell)')

    def _republish(self):
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.grid)


def main():
    rclpy.init()
    node = ProhibitionMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

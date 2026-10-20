#!/usr/bin/env python3
from std_msgs.msg import String


class DebugClient:
    def __init__(self, node, verbose=True):
        self.node = node
        self.verbose = verbose
        self.pub = node.create_publisher(String, '/debugger/debug_msgs', 10)
        self.node_name = node.get_name()

    def send(self, message, msg_id="General", level="info", ttl=-1):
        """
        level: 'info' (default), 'warn' (yellow), 'err' (red)
        """
        if msg_id == "NODE_STATUS" or self.verbose:
            msg = String()
            # Clean string spaces to prevent table formatting breaks
            clean_id = str(msg_id).strip()
            msg.data = f"{self.node_name}|{clean_id}|{level}|{ttl}|{message}"
            self.pub.publish(msg)
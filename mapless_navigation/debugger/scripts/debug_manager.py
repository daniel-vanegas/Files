#!/usr/bin/env python3
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DebugManager(Node):
    def __init__(self):
        super().__init__('debug_manager')
        self.node_states = {}               # (node, msg_id) -> dict
        self.node_first_time = {}           # node -> timestamp of first message
        self.node_msgs = {}                 # node -> list of msg_ids in arrival order
        # A node is alive if NODE_STATUS arrived within this window.
        # NODE_STATUS is published at 1 Hz by convention.
        self.heartbeat_timeout = 3.0

        self.sub = self.create_subscription(
            String, '/debugger/debug_msgs', self.callback, 10)
        self.timer = self.create_timer(0.1, self.render)
        sys.stdout.write("\033[2J\033[H")

    def callback(self, msg):
        try:
            parts = msg.data.split('|')
            if len(parts) == 5:
                node, msg_id, level, ttl, content = [p.strip() for p in parts]
                key = (node, msg_id)

                if node not in self.node_first_time:
                    self.node_first_time[node] = time.time()
                    self.node_msgs[node] = []

                if msg_id not in self.node_msgs[node]:
                    self.node_msgs[node].append(msg_id)

                self.node_states[key] = {
                    "content": content,
                    "last_seen": time.time(),
                    "level": level.lower(),
                    "ttl": float(ttl)
                }
        except Exception:
            pass

    def _is_node_alive(self, node, now):
        status_data = self.node_states.get((node, "NODE_STATUS"))
        return (status_data is not None and
                (now - status_data["last_seen"]) < self.heartbeat_timeout)

    def _is_msg_visible(self, node, msg_id, now, node_alive):
        data = self.node_states.get((node, msg_id))
        if not data:
            return False
        
        # Hide transient messages expired by TTL
        if node_alive and data["ttl"] > 0 and (now - data["last_seen"]) > data["ttl"]:
            return False

        # Hide persistent messages from dead nodes after grace period (7 seconds)
        age = now - data["last_seen"]
        if not node_alive and msg_id != "NODE_STATUS" and age > 7.0:
            return False

        return True

    def render(self):
        now = time.time()
        sys.stdout.write("\033[H\033[J")

        node_order = sorted(self.node_first_time.keys())
        
        # Identify active nodes and active IDs for dynamic resizing
        active_nodes = set()
        active_ids = []

        for node in node_order:
            node_alive = self._is_node_alive(node, now)
            for msg_id in self.node_msgs[node]:
                if self._is_msg_visible(node, msg_id, now, node_alive):
                    active_nodes.add(node)
                    active_ids.append(msg_id)

        # Dynamic widths recalculate down when elements expire
        max_node_len = max((len(node) for node in active_nodes), default=0)
        node_col_width = max(max_node_len + 2, len("NODE"))

        max_id_len = max((len(msg_id) for msg_id in active_ids), default=0)
        id_col_width = max(max_id_len, len("ID"))

        out = "=" * (node_col_width + 1 + id_col_width + 1 + 40) + "\n"
        out += f"{'NODE':<{node_col_width}} | {'ID':<{id_col_width}} | MESSAGE\n"
        out += "-" * (node_col_width + 1 + id_col_width + 1 + 40) + "\n"

        for node in node_order:
            all_ids = self.node_msgs[node]
            node_alive = self._is_node_alive(node, now)
            status_data = self.node_states.get((node, "NODE_STATUS"))

            # Render NODE_STATUS first if active
            if "NODE_STATUS" in all_ids and status_data:
                if self._is_msg_visible(node, "NODE_STATUS", now, node_alive):
                    out += self._format_line(
                        node, "NODE_STATUS", status_data, now,
                        node_col_width, id_col_width, node_alive)

            for msg_id in all_ids:
                if msg_id == "NODE_STATUS":
                    continue

                data = self.node_states.get((node, msg_id))
                if not data:
                    continue

                # Skip rendering if message is not active
                if not self._is_msg_visible(node, msg_id, now, node_alive):
                    continue

                out += self._format_line(
                    node, msg_id, data, now,
                    node_col_width, id_col_width, node_alive)

        out += "\n(Ctrl+C to stop monitor)"
        sys.stdout.write(out)
        sys.stdout.flush()

    def _format_line(self, node, msg_id, data, now, col_width, id_col_width, node_alive):
        age = now - data["last_seen"]
        content = data["content"]
        level = data["level"]
        ttl = data["ttl"]

        # Persistent message on a live node: content shown as-is.
        if ttl == -1 and node_alive:
            if level == "err":
                status_str = f"\033[91m{content}\033[0m"
            elif level == "warn":
                status_str = f"\033[93m{content}\033[0m"
            else:
                status_str = content
        else:
            # Age-based logic. Also applies to ttl == -1
            # messages when the node is not alive.
            if age > 7.0:
                display = ("Node deactivated or lost"
                           if msg_id == "NODE_STATUS" else "---")
                status_str = f"\033[91m{display}\033[0m"
            elif age > 5.0:
                tag = "\033[91m\033[7m\033[1m [ LOST ] \033[0m"
                display = ("Node deactivated or lost"
                           if msg_id == "NODE_STATUS" else content)
                status_str = f"{tag} {display}"
            elif age > 1.5:
                tag = "\033[93m\033[7m\033[1m [Inactive] \033[0m"
                display = ("Node is not responding"
                           if msg_id == "NODE_STATUS" else content)
                status_str = f"{tag} {display}"
            else:
                if level == "err":
                    status_str = f"\033[91m{content}\033[0m"
                elif level == "warn":
                    status_str = f"\033[93m{content}\033[0m"
                else:
                    status_str = content

        bracketed = f"[{node}]"
        return f"{bracketed:<{col_width}} | {msg_id:<{id_col_width}} | {status_str}\n"


def main(args=None):
    rclpy.init(args=args)
    node = DebugManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
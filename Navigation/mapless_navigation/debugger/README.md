# Hybrid Debugger for ROS2

A cross-language debug tool that displays live, color-coded status messages from **Python** and **C++** nodes in a single terminal table.

## Features

- Unified table for all nodes via `/debugger/debug_msgs`
- Enable debugger with verbose output
  - c++: 
    - debugger::DebugClient dbg_(this, true);
  - python:
    - self.dbg = DebugClient(self, verbose=True)
- Color coding:
  - `err` → red
  - `warn` → yellow
  - `info` → default
- `NODE_STATUS` messages pinned to the top
- Identical `DebugClient` API for **Python** and **C++**
- Smart timeout visualization.
  - **1.5 s** → `[Inactive]`
  - **5.0 s** → `[LOST]`
  - **> 7.0 s** → `---` (red)

- Configurable auto erase messages from the table:
  - **ttl** → Message is erased from the table automatically after n seconds of inactivity. Default value: -1 for no timeout.

---
## Message Fields

- **msg_id**: Identifier (e.g., `"NODE_STATUS"`, `"SCAN"`, `"PLANNER"`)
- **level**:
  - `"info"` (default)
  - `"warn"`
  - `"err"`
- **ttl**: Time-to-live in seconds

---

## Installation

From your ROS2 workspace root:

```bash
colcon build --packages-select debugger --symlink-install
source install/setup.bash
```

---

## Running the Debug Manager

Run this in a dedicated terminal:

```bash
ros2 run debugger debug_manager
```

---

## Usage

#### CMakeLists.txt

```cmake
find_package(debugger REQUIRED)
target_include_directories(my_node PRIVATE ${debugger_INCLUDE_DIRS})
```

### Python

```python
from debugger import DebugClient

self.dbg = DebugClient(self, verbose=True)

self.dbg.send("Node ready", "NODE_STATUS", "info")
self.dbg.send("Low battery", "BATTERY", "warn")
# This message will disappear after 5 seconds of inactivity
self.dbg.send("Sensor failed", "ERROR", "err", ttl=5.0))

```

---

### C++


#### Node Code

```cpp
#include "debugger/debug_client.hpp"

debugger::DebugClient dbg_(this, true);

dbg_.send("Node ready", "NODE_STATUS", "info");
dbg_.send("Low battery", "BATTERY", "warn");
dbg_.send("Sensor failed", "ERROR", "err");

-------------------
Example
-------------------
#include "debugger/debug_client.hpp"

class GetGoalPoint : public rclcpp::Node
{
public:
    GetGoalPoint()
    : Node("get_goal_point"),
      dbg_(this, true) // <---- Add this
    {
        // ================= DEBUGGER =================
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");
        
        // Erasable message (vanishes after 7 seconds)
        dbg_.send("Processing frame...", "DEBUG", "info", 7.0);
    }

private:
    void update()
    {
        dbg_.send("Node initialized and ready.", "NODE_STATUS", "info");
        // Functional code ...
    }

    // 
    debugger::DebugClient dbg_;
};
```


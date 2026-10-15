cmake_minimum_required(VERSION 3.8)
project(mapless_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(debugger REQUIRED) 

# ===== Executable =====
add_executable(local_occ_grid_node src/local_occ_grid_node.cpp)
ament_target_dependencies(local_occ_grid_node
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(local_occ_grid_node PRIVATE ${debugger_INCLUDE_DIRS})

add_executable(get_goal_point src/get_goal_point.cpp)
ament_target_dependencies(get_goal_point
  rclcpp
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(get_goal_point PRIVATE ${debugger_INCLUDE_DIRS})

# add_executable(tf_goal_point src/tf_goal_point.cpp)

# ament_target_dependencies(tf_goal_point
#   rclcpp
#   std_msgs
#   tf2
#   tf2_ros
#   tf2_geometry_msgs
# )

#target_include_directories(tf_goal_point PRIVATE ${debugger_INCLUDE_DIRS})

# ===== Install =====
install(TARGETS local_occ_grid_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS get_goal_point
  DESTINATION lib/${PROJECT_NAME}
)

# install(TARGETS tf_goal_point
#   DESTINATION lib/${PROJECT_NAME}
# )

# ===== Install Python scripts =====
install(
  PROGRAMS
    scripts/reactives/pot_fields.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/reactives/bt_navigation.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Install RNN script =====
install(
  PROGRAMS
    scripts/RNN_mapless.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Install TorchModels Python module =====
install(
  DIRECTORY
    scripts/TorchModels
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

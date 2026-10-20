cmake_minimum_required(VERSION 3.8)
project(navigation_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(debugger REQUIRED)

add_executable(path_planner_node
  src/path_planner_node.cpp
  src/path_planner.cpp
)
ament_target_dependencies(path_planner_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  std_srvs
  action_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(path_planner_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${debugger_INCLUDE_DIRS}
)

install(TARGETS path_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
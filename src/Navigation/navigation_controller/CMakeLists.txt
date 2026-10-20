cmake_minimum_required(VERSION 3.8)
project(navigation_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(debugger REQUIRED)

add_executable(simple_move_node src/simple_move_node.cpp)
ament_target_dependencies(simple_move_node
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  action_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_include_directories(simple_move_node PRIVATE ${debugger_INCLUDE_DIRS})

add_executable(obs_detect_node src/obs_detect_node.cpp)
ament_target_dependencies(obs_detect_node
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  Eigen3
)
target_include_directories(obs_detect_node PRIVATE ${debugger_INCLUDE_DIRS})

install(TARGETS simple_move_node obs_detect_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
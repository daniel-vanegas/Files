cmake_minimum_required(VERSION 3.16)
project(mapless_dataset)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(debugger REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Recorder (C++)
add_executable(dataset_recorder src/dataset_recorder.cpp)
ament_target_dependencies(dataset_recorder
  rclcpp std_msgs nav_msgs geometry_msgs sensor_msgs
  tf2 tf2_ros tf2_geometry_msgs
  debugger
  ament_index_cpp
)
target_include_directories(dataset_recorder PRIVATE 
  ${OpenCV_INCLUDE_DIRS}
  ${debugger_INCLUDE_DIRS}
)
target_link_libraries(dataset_recorder ${OpenCV_LIBS})

install(TARGETS dataset_recorder DESTINATION lib/${PROJECT_NAME})

# Python scripts
install(PROGRAMS
  scripts/random_goal_sampler.py
  scripts/generate_npz_data.py
  scripts/Nav2_map_relay.py
  scripts/prohibition_map_publisher.py
  scripts/teleop_dataset_recorder.py
  scripts/dataset_obstacles_control.py
  scripts/make_dataset_split.py
  scripts/path_generator.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
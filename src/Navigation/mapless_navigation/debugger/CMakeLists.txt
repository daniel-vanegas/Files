cmake_minimum_required(VERSION 3.8)
project(debugger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(debugger)

# Install the debug manager script
install(
  PROGRAMS scripts/debug_manager.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME debug_manager
)

# Install the C++ header
install(
  FILES include/debugger/debug_client.hpp
  DESTINATION include/${PROJECT_NAME}
)

# Export the include directory
ament_export_include_directories(include)

ament_package()

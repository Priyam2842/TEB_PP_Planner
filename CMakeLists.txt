cmake_minimum_required(VERSION 3.8)
project(teb_pure_pursuit_planner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install executable Python nodes
install(
  PROGRAMS
    scripts/TEB_Pure_Pursuit_Planner.py
    scripts/Pure_pursuit_Gazebo.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config directories
install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

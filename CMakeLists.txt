cmake_minimum_required(VERSION 3.0.2)
project(f1tenth_racing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  ackermann_msgs
  geometry_msgs
  visualization_msgs
  sensor_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs nav_msgs ackermann_msgs geometry_msgs visualization_msgs sensor_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# installing Python scripts
catkin_install_python(PROGRAMS
  src/pure_pursuit_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# installing launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# installing config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# installing CSV files
install(DIRECTORY csv/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/csv
)
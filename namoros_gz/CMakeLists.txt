cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(namoros_gz)

# ROS 2 build system
find_package(ament_cmake REQUIRED)

# Ignition dependencies
find_package(ignition-cmake2 REQUIRED)
ign_find_package(ignition-plugin1 REQUIRED COMPONENTS register)
ign_find_package(ignition-gazebo6 REQUIRED)

set(IGN_PLUGIN_VER ${ignition-plugin1_VERSION_MAJOR})
set(IGN_GAZEBO_VER ${ignition-gazebo6_VERSION_MAJOR})

# Include directories
include_directories(
  include
  ${IGNITION-GAZEBO${IGN_GAZEBO_VER}_INCLUDE_DIRS}
  ${IGNITION-PLUGIN${IGN_PLUGIN_VER}_INCLUDE_DIRS}
)

# Build the plugin
add_library(${PROJECT_NAME} SHARED
  src/Namo.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME}
  ignition-plugin${IGN_PLUGIN_VER}::ignition-plugin${IGN_PLUGIN_VER}
  ignition-gazebo${IGN_GAZEBO_VER}::ignition-gazebo${IGN_GAZEBO_VER}
)

# Install the plugin .so
install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

ament_package()

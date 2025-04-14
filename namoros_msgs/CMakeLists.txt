cmake_minimum_required(VERSION 3.8)
project(namoros_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NamoPath.msg"
  "msg/NamoPlan.msg"
  "msg/NamoPose.msg"
  "msg/NamoAction.msg"
  "msg/ComputePlan.srv"
  "msg/SimulatePath.srv"
  "msg/GetEntityPolygon.srv"
  "msg/AddOrUpdateMovableObstacle.srv"
  "msg/SynchronizeState.srv"
  "msg/NamoEntity.msg"
  "msg/NamoConflict.msg"
  "msg/DetectConflicts.srv"
  "msg/ComputeEvasionPlan.srv"
  "msg/UpdatePlan.srv"
  "msg/EndPostpone.srv"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()

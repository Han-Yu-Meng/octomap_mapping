project(octomap_mapping)

find_package(PCL       REQUIRED)
find_package(octomap   REQUIRED)

fins_add_node(${PROJECT_NAME}
  node.cpp
)

fins_link_ros_dependencies(${PROJECT_NAME}
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
)

target_include_directories(${PROJECT_NAME} PRIVATE 
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${OCTOMAP_LIBRARIES}
  ${PCL_LIBRARIES}
)
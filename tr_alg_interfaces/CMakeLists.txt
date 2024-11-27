cmake_minimum_required(VERSION 3.8)
project(tr_alg_interfaces)

if(USE_BM)
  set(tr_interfaces_DIR /data/thoreyes/ros/thoreyes_base/install/share/tr_interfaces/cmake/)

elseif(USE_IX)
  set(tr_interfaces_DIR /data/thoreyes_ix/ros/thoreyes_base/install/share/tr_interfaces/cmake/)

elseif(USE_HW)
  set(tr_interfaces_DIR /data/thoreyes_hw/ros/thoreyes_base/install/share/tr_interfaces/cmake/)

else()
  set(tr_interfaces_DIR /data/thoreyes_nv/ros/thoreyes_base/install/share/tr_interfaces/cmake/)
endif()



if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wdeprecated-declarations -Wsign-compare)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic )
endif()

SET(CMAKE_BUILD_TYPE "RELEASE")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_compile_options(-std=c++17)


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tr_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

file(GLOB_RECURSE MSG_FILE ./msg/*.msg)
file(GLOB_RECURSE SRV_FILE ./msg/*.srv)


rosidl_generate_interfaces(${PROJECT_NAME}
  "./msg/DeviceUsage.msg"  
  "./msg/ChannelList.msg"
  "./msg/AlgHeader.msg"
  "./msg/ExtObjItem.msg"
  "./msg/ExtResultItem.msg"
  "./msg/ResultItem.msg"
  "./msg/Results.msg"
  "./srv/GetAvaliableDevice.srv"
  "./srv/GetModuleEnable.srv"
  "./srv/GetChannelStatus.srv"
  "./srv/SetChannelConfig.srv"
  "./srv/SetModuleEnable.srv"
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)
 
ament_package()


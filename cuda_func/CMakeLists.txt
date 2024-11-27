cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(help_func_lib LANGUAGES CXX)
SET(CMAKE_INSTALL_PREFIX /data_temp/thoreyes_ix_cvcuda/ros/help_func_lib)

if(NOT "${TARGET_DIR}" STREQUAL "")
  message(STATUS "Use TARGET_DIR=${TARGET_DIR} from command argument")
elseif(DEFINED ENV{TARGET_DIR})
  set(TARGET_DIR "$ENV{TARGET_DIR}")
  message(STATUS "Use TARGET_DIR=${TARGET_DIR} from environment")
else()
  set(TARGET_DIR "${CMAKE_SOURCE_DIR}/build" CACHE STRING "target directory")
  message(STATUS "target dir: ${TARGET_DIR}")
endif()

set(CMAKE_BUILD_TYPE Release)

#
# CUDA_PATH can be specified through below means shown in priority order
#   1. cmake command line argument, -DCUDA_PATH=/path/to/cuda
#   2. bash environment variable, export CUDA_PATH=/path/to/cuda
#
if(NOT "${CUDA_PATH}" STREQUAL "")
  message(STATUS "Use CUDA_PATH=${CUDA_PATH} from command argument")
elseif(DEFINED ENV{CUDA_PATH})
  set(CUDA_PATH "$ENV{CUDA_PATH}")
  message(STATUS "Use CUDA_PATH=${CUDA_PATH} from enviroment")
else()
  set(CUDA_PATH "/usr/local/corex" CACHE PATH "cuda installation root path")
  message(STATUS "Use CUDA_PATH=${CUDA_PATH} ")
endif()

#
# Compiler can be specified through below means shown priority order
#   1. relative to CUDA_PATH/bin
#   2. cmake command line argument, -DCOMPILER_PATH=/path/to/compiler/dir
#
if(NOT "${COMPILER_PATH}" STREQUAL "")
  set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++)
  set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang)
else()
  set(CMAKE_CXX_COMPILER ${CUDA_PATH}/bin/clang++)
  set(CMAKE_C_COMPILER ${CUDA_PATH}/bin/clang)
endif()
message(STATUS "Use ${CMAKE_CXX_COMPILER} and ${CMAKE_C_COMPILER} as C++ and C compiler")

#
# GPU arch
#
if("${CUDA_ARCH}" STREQUAL "")
  set(CUDA_ARCH "ivcore11" CACHE STRING "GPU architecture tag, ivcore11")
endif()

###################################################################################################
# Auxiliary functions
###################################################################################################
macro(cuda_add_library)
  foreach(File ${ARGN})
    if(${File} MATCHES ".*\.cu$")
      set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
  add_library(${ARGN})
endmacro()

macro(cuda_add_executable)
  foreach(File ${ARGN})
  if(${File} MATCHES ".*\.cu$")
    set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
  endif()
  endforeach()
  add_executable(${ARGN})
endmacro()

###################################################################################################
# Main part
###################################################################################################

string(APPEND CMAKE_CXX_FLAGS " --std=c++17")
#string(APPEND CMAKE_CXX_FLAGS " -fvisibility=hidden")
string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_PATH}")
string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=${CUDA_ARCH}")
string(APPEND CMAKE_CXX_FLAGS " -Wl,--disable-new-dtags")
string(APPEND CMAKE_CXX_FLAGS " -mllvm -pragma-unroll-threshold=100000")
string(APPEND CMAKE_CXX_FLAGS " -mllvm -unroll-threshold=5000")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-command-line-argument")
string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")

#
# Homemade find cudart
#
find_library(
  CUDART_LIBRARY cudart
  PATHS
  ${CUDA_PATH}
  PATH_SUFFIXES
  lib/x64
  lib64
  lib
  NO_DEFAULT_PATH
)
if(NOT CUDART_LIBRARY)
  message(FATAL "Cannot find cudart")
endif()

#
# Homemade find cuda driver
#
find_library(
  CUDA_DRIVER_LIBRARY cuda
  PATHS
  ${CUDA_PATH}
  PATH_SUFFIXES
  lib/x64
  lib64
  lib
  NO_DEFAULT_PATH
)
if(NOT CUDA_DRIVER_LIBRARY)
  message(FATAL "Cannot find cuda driver")
endif()

#
# Common include
#
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_PATH}/include
)

###################################################################################################
# add executable
###################################################################################################



file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)


# 添加 CUDA 动态库
cuda_add_library(
  help_func_lib
  SHARED
  ${SRCS}
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_link_libraries(
  help_func_lib
  ${CUDART_LIBRARY}
  ${CUDA_DRIVER_LIBRARY}
)



# 设置安装规则
install(TARGETS help_func_lib
    DESTINATION lib
    COMPONENT lib
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include
    COMPONENT headers
)
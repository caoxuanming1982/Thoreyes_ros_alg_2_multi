cmake_minimum_required(VERSION 3.8)
project(alg_module_base)


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

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_compile_options(-std=c++17)



#find_package(tinyxlm2 REQUIRED)
#include_directories(${TINYXML2_INCLUDE_DIRS})
#link_directories(${TINYXML2_LIB_DIRS})

# define architecture
MESSAGE("System Arch: " ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
        set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(AARCH64 1)
endif()

# sophon arch define


if(USE_BM)
  SET(CMAKE_INSTALL_PREFIX /data/thoreyes/ros/alg_module_base)

# add libsophon
  find_package(libsophon REQUIRED)
  include_directories(${LIBSOPHON_INCLUDE_DIRS})
  link_directories(${LIBSOPHON_LIB_DIRS})

  set(SOPHON_LIBS bmion bmjpuapi bmjpulite bmlib bmrt bmvideo bmvppapi bmvpulite vpp cpuop usercpu vpp yuv video_bm bmcv)

# add sophon-ffmpeg
  set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
  find_package(FFMPEG REQUIRED NO_DEFAULT_PATH)
  include_directories(${FFMPEG_INCLUDE_DIRS})
  link_directories(${FFMPEG_LIB_DIRS})


# add sophon-opencv
  set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
  find_package(OpenCV REQUIRED NO_DEFAULT_PATH)
  include_directories(${OpenCV_INCLUDE_DIRS})

# sophon sdk
  include_directories(
    /opt/sophon/sophon-sail/include
  )

  link_directories(
    /opt/sophon/sophon-sail/lib
  )
  ADD_DEFINITIONS(-DUSE_BM)
  ADD_DEFINITIONS(-DUSE_BMCV -DUSE_FFMPEG -DUSE_OPENCV -D__DISPLAY__)
  if(X86_64)
          ADD_DEFINITIONS(-DUSE_PCIE -DBM_PCIE_MODE)
  endif()
  
elseif(USE_IX)
  SET(CMAKE_INSTALL_PREFIX /data/thoreyes_ix/ros/alg_module_base)
  ADD_DEFINITIONS(-DUSE_IX)
  set(Torch_DIR  /opt/libtorch/share/cmake/Torch)
  set(ATen_DIR  /opt/libtorch/share/cmake/ATen)
  set(PYTORCH_DISABLE_TORCH_FUNCTION_DEPRECATION_WARNINGS 1)
  find_package(Torch REQUIRED)
  if(TORCH_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})
    message(STATUS "torch include ${TORCH_INCLUDE_DIRS}")
#    include_directories(${TORCH_INCLUDE_DIRS}/torch/csrc/api/include)
    link_directories(/opt/libtorch/lib)
    message(STATUS "torch libarys ${TORCH_LIBRARIES}")
    add_compile_options(${TORCH_CXX_FLAGS} -Wno-unused-parameter -Wsign-compare -Wvariadic-macros -Wunknown-pragmas -Wsign-compare)
  else()
    message(ERROR "torch not found")

  endif()
  if(USE_CVCUDA)
  set(cvcuda_DIR /opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/cmake/cvcuda/)

  find_package(cvcuda REQUIRED)
  endif()
#  set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIRS})
  if(OpenCV_FOUND)
  message(STATUS "OpenCV library status:")
  message(STATUS "    config: ${OpenCV_DIR}")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(ERROR "opencv not found")
  endif()

  find_package(CUDA REQUIRED)

#  set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "cuda include ${CUDA_INCLUDE_DIRS}")
  ADD_DEFINITIONS(-DNDEBUG -DSTRIP_ERROR_MESSAGES)
  if(HAVE_OPENCV_CUDA)
    message(STATUS "opencv with cuda")
  endif()

  ADD_DEFINITIONS(-DUSE_FP16)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    include_directories(${OpenMP_INCLUDE_DIRS})
    link_directories(${OpenMP_LIB_DIRS})
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
    message(STATUS "openmp found")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}")
    
    message(STATUS "CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
    message(STATUS "CMAKE_C_FLAGS ${OpenMP_C_FLAGS}")

  else()
    set(OpenMP_LIBRARIES "")
  endif()

else()
  SET(CMAKE_INSTALL_PREFIX /data/thoreyes_nv/ros/alg_module_base)
#  SET(CMAKE_INSTALL_PREFIX /data/thoreyes/ros/alg_module_base_nv)
  set(Torch_DIR  /opt/libtorch/share/cmake/Torch)
  set(ATen_DIR  /opt/libtorch/share/cmake/ATen)
  set(PYTORCH_DISABLE_TORCH_FUNCTION_DEPRECATION_WARNINGS 1)
  find_package(Torch REQUIRED)
  if(TORCH_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})
    message(STATUS "torch include ${TORCH_INCLUDE_DIRS}")
#    include_directories(${TORCH_INCLUDE_DIRS}/torch/csrc/api/include)
    link_directories(/opt/libtorch/lib)
    message(STATUS "torch libarys ${TORCH_LIBRARIES}")
    add_compile_options(${TORCH_CXX_FLAGS} -Wno-unused-parameter -Wsign-compare -Wvariadic-macros -Wunknown-pragmas -Wsign-compare)
  else()
    message(ERROR "torch not found")

  endif()

#  set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIRS})
  if(OpenCV_FOUND)
  message(STATUS "OpenCV library status:")
  message(STATUS "    config: ${OpenCV_DIR}")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(ERROR "opencv not found")
  endif()

  find_package(CUDA REQUIRED)

#  set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "cuda include ${CUDA_INCLUDE_DIRS}")
  ADD_DEFINITIONS(-DNDEBUG -DSTRIP_ERROR_MESSAGES)
  if(HAVE_OPENCV_CUDA)
    message(STATUS "opencv with cuda")
  endif()

  ADD_DEFINITIONS(-DUSE_FP16)
  ADD_DEFINITIONS(-DUSE_CUDA)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    include_directories(${OpenMP_INCLUDE_DIRS})
    link_directories(${OpenMP_LIB_DIRS})
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
    message(STATUS "openmp found")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}")
    
    message(STATUS "CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
    message(STATUS "CMAKE_C_FLAGS ${OpenMP_C_FLAGS}")

  else()
    set(OpenMP_LIBRARIES "")
  endif()

endif()

find_package(tinyxml2 REQUIRED)
include_directories(${TINYXML2_INCLUDE_DIRS})
link_directories(${TINYXML2_LIB_DIRS})

set(tinyxml2_LIBRARIES tinyxml2)

include_directories(./include)
include_directories(./include_private)


file(GLOB SRC_FILES src/*.cpp)
file(GLOB SRC_FILES_CV_LIB src/cv_lib/*.cpp)
file(GLOB SRC_FILES_NETWORK src/network_engine/*.cpp)

file(GLOB SRC_FILES_CV_LIB_BM src/cv_lib/bm/*.cpp)
file(GLOB SRC_FILES_CV_LIB_CUDA src/cv_lib/cuda/*.cpp)
file(GLOB SRC_FILES_CV_LIB_CV src/cv_lib/cv/*.cpp)
file(GLOB SRC_FILES_CV_LIB_CVCUDA src/cv_lib/cvcuda/*.cpp)
file(GLOB SRC_FILES_CV_LIB_HW src/cv_lib/hw/*.cpp)

file(GLOB SRC_FILES_NETWORK_BM src/network_engine/bm/*.cpp)
file(GLOB SRC_FILES_NETWORK_TORCH src/network_engine/torch/*.cpp)
file(GLOB SRC_FILES_NETWORK_HW src/network_engine/hw/*.cpp)
file(GLOB SRC_FILES_NETWORK_TENSORRT src/network_engine/tensorrt/*.cpp)


set(SRC_FILES ${SRC_FILES} ${SRC_FILES_CV_LIB} ${SRC_FILES_CV_LIB})

if(USE_BM)
  set(SRC_FILES ${SRC_FILES} ${SRC_FILES_CV_LIB_BM} ${SRC_FILES_NETWORK_BM})

elseif(USE_IX)
  set(SRC_FILES ${SRC_FILES} ${SRC_FILES_CV_LIB_CV} ${SRC_FILES_NETWORK_TORCH})

elseif(USE_HW)
  set(SRC_FILES ${SRC_FILES} ${SRC_FILES_CV_LIB_HW} ${SRC_FILES_NETWORK_HW})

else()
  set(SRC_FILES ${SRC_FILES} ${SRC_FILES_CV_LIB_CUDA} ${SRC_FILES_NETWORK_TORCH})

endif()


install(DIRECTORY ./include/ DESTINATION include/${PROJECT_NAME}/)
install(DIRECTORY ./doc/ DESTINATION doc/)

add_library(${PROJECT_NAME}_share SHARED ${SRC_FILES})
add_executable(${PROJECT_NAME}_test_cv_lib_main  ./test/test_cv_lib.cpp)

if(USE_BM)

target_link_libraries(${PROJECT_NAME}_share
  ${FFMPEG_LIBS} ${OpenCV_LIBS} ${SOPHON_LIBS} ${tinyxml2_LIBRARIES} #${tr_interfaces_LIBRARIES} ${tr_alg_interfaces_LIBRARIES}
  sail
)
target_link_libraries(${PROJECT_NAME}_test_cv_lib_main
  ${FFMPEG_LIBS} ${OpenCV_LIBS} ${SOPHON_LIBS} ${tinyxml2_LIBRARIES} 
  ${PROJECT_NAME}_share
)

elseif(USE_IX)
target_link_libraries(${PROJECT_NAME}_share
  ${FFMPEG_LIBS} ${OpenCV_LIBS} ${tinyxml2_LIBRARIES} ${TORCH_LIBRARIES} ${OpenMP_LIBRARIES}#${tr_interfaces_LIBRARIES} ${tr_alg_interfaces_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_test_cv_lib_main
${FFMPEG_LIBS} ${OpenCV_LIBS} ${tinyxml2_LIBRARIES} ${TORCH_LIBRARIES} ${OpenMP_LIBRARIES}#${tr_interfaces_LIBRARIES} ${tr_alg_interfaces_LIBRARIES}
${PROJECT_NAME}_share
)

elseif(USE_HW)
  message(ERROR "not support hw device")

else()

target_link_libraries(${PROJECT_NAME}_share
  ${FFMPEG_LIBS} ${OpenCV_LIBS} ${tinyxml2_LIBRARIES} ${TORCH_LIBRARIES} ${OpenMP_LIBRARIES}#${tr_interfaces_LIBRARIES} ${tr_alg_interfaces_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_test_cv_lib_main
${FFMPEG_LIBS} ${OpenCV_LIBS} ${tinyxml2_LIBRARIES} ${TORCH_LIBRARIES} ${OpenMP_LIBRARIES}#${tr_interfaces_LIBRARIES} ${tr_alg_interfaces_LIBRARIES}
${PROJECT_NAME}_share
)
endif()

install(
  TARGETS ${PROJECT_NAME}_share
  EXPORT ${PROJECT_NAME}_share
  DESTINATION lib)



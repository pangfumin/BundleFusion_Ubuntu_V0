cmake_minimum_required(VERSION 3.5)
project(BundleFusion)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

find_package(GLEW REQUIRED)
message(STATUS "Found GLEW in ${GLEW_INCLUDE_DIR}")
if (GLEW_FOUND)
  include_directories(${GLEW_INCLUDE_DIRS})
  link_directories(${GLEW_LIBRARIES})
endif()

find_package(OpenGL REQUIRED)
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(CUDA)
find_package(X11 REQUIRED)  
set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_60;-O3;-G;-g;-std=c++11)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenGL_INCLUDE_DIRS})

include_directories(
    ${PROJECT_SOURCE_DIR}/external/mLib/include
    ${PROJECT_SOURCE_DIR}/mLibExternal/include
    ${PROJECT_SOURCE_DIR}/FriedLiver/Include/cutil/inc
    ${PROJECT_SOURCE_DIR}/FriedLiver/Include
    ${PROJECT_SOURCE_DIR}/FriedLiver/Include/Uplink
    ${PROJECT_SOURCE_DIR}/FriedLiver/Source
    ${PROJECT_SOURCE_DIR}/FriedLiver/Source/SiftGPU
    ${PROJECT_SOURCE_DIR}/FriedLiver/Source/Solver
)
add_library(LIBEXT SHARED
    external/mLib/src/core-base/common.cpp
    external/mLib/src/core-util/timer.cpp
    external/mLib/src/core-util/utility.cpp
    external/mLib/include/mLibDepthCamera.cpp
    #external/mLib/src/ext-depthcamera/image.cpp
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

file(GLOB CURRENT_HEADERS  FriedLiver/Source/*.h )
file(GLOB CURRENT_SOURCES  FriedLiver/Source/*.cpp)


CUDA_ADD_EXECUTABLE(FriedLiver 
    FriedLiver/Source/FriedLiver.cpp FriedLiver/Source/ConditionManager.cpp
    FriedLiver/Source/RGBDSensor.cpp
    # FriedLiver/Source/PrimeSenseSensor.cpp
    #FriedLiver/Source/BinaryDumpReader.cpp
    FriedLiver/Source/SensorDataReader.cpp
    FriedLiver/Source/CUDAImageManager.cpp
    FriedLiver/Source/CUDAImageUtil.cu
    FriedLiver/Source/TimingLog.cpp
    FriedLiver/Source/Bundler.cpp
    FriedLiver/Source/OnlineBundler.cu
    FriedLiver/Source/OnlineBundler.cpp
    FriedLiver/Source/SBA.cu
    FriedLiver/Source/SBA.cpp
    FriedLiver/Source/CUDACache.cu
    FriedLiver/Source/CUDACache.cpp
    FriedLiver/Source/GlobalAppState.cpp
    FriedLiver/Source/TrajectoryManager.cpp

    # sift
    FriedLiver/Source/SiftGPU/SIFTImageManager.cu
    FriedLiver/Source/SiftGPU/SIFTImageManager.cpp
    FriedLiver/Source/SiftGPU/SiftGPU.cpp
    FriedLiver/Source/SiftGPU/ProgramCU.cu
    FriedLiver/Source/SiftGPU/CUDASiftConstant.cu
    #FriedLiver/Source/stdafx.cpp
    FriedLiver/Source/SiftGPU/SiftMatch.cpp
    FriedLiver/Source/SiftGPU/SiftPyramid.cpp
    FriedLiver/Source/SiftGPU/GlobalUtil.cpp
    FriedLiver/Source/SiftGPU/CuTexImage.cpp

    FriedLiver/Source/Solver/CUDASolverBundling.cpp
    FriedLiver/Source/Solver/SolverBundling.cu
    FriedLiver/Source/DepthSensing/DepthSensing.cpp
    FriedLiver/Source/DepthSensing/CUDAMarchingCubesSDF.cu
    FriedLiver/Source/DepthSensing/CUDAMarchingCubesHashSDF.cpp
    FriedLiver/Source/DepthSensing/CUDASceneRepChunkGrid.cu
    FriedLiver/Source/DepthSensing/CUDASceneRepChunkGrid.cpp
    FriedLiver/Source/DepthSensing/CUDAConstant.cu
    FriedLiver/Source/DepthSensing/CUDASceneRepHashSDF.cu
    FriedLiver/Source/DepthSensing/CUDASceneRepHashSDF.cpp
    FriedLiver/Source/DepthSensing/TimingLogDepthSensing.cpp
    FriedLiver/Source/DepthSensing/ScanCS.cu
    FriedLiver/Source/DepthSensing/CUDAScan.cpp
    FriedLiver/Source/DepthSensing/CUDARayCastSDF.cu
    FriedLiver/Source/DepthSensing/CUDARayCastSDF.cpp
    FriedLiver/Source/DepthSensing/CameraUtil.cu

#FriedLiver/Source/DepthSensing/StdOutputLogger.cpp
)

target_link_libraries(FriedLiver
        LIBEXT
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARY}
    ${GLFW3_LIBRARY}
    GLEW
    X11 Xrandr Xinerama Xi Xxf86vm Xcursor pthread dl
    GLU
#    OpenNI2
    ${OpenCV_LIBS}
)

# test
CUDA_ADD_EXECUTABLE( test_SiftGPU
    test/test_SiftGPU.cpp
    # sift
    FriedLiver/Source/SiftGPU/SIFTImageManager.cu
    FriedLiver/Source/SiftGPU/SIFTImageManager.cpp
    FriedLiver/Source/SiftGPU/SiftGPU.cpp
    FriedLiver/Source/SiftGPU/ProgramCU.cu
    FriedLiver/Source/SiftGPU/CUDASiftConstant.cu
    #FriedLiver/Source/stdafx.cpp
    FriedLiver/Source/SiftGPU/SiftMatch.cpp
    FriedLiver/Source/SiftGPU/SiftPyramid.cpp
    FriedLiver/Source/SiftGPU/GlobalUtil.cpp
    FriedLiver/Source/SiftGPU/CuTexImage.cpp)
target_link_libraries(test_SiftGPU
    LIBEXT
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARY}
    ${GLFW3_LIBRARY}
    GLEW
    X11 Xrandr Xinerama Xi Xxf86vm Xcursor pthread dl
    GLU
#        OpenNI2
    ${OpenCV_LIBS}
)


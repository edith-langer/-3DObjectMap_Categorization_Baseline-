cmake_minimum_required(VERSION 3.10)
project(3d_obj_map)

set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

find_package(OpenCV 3 REQUIRED)

SET(CMAKE_PREFIX_PATH "/home/edith/Software/pcl-1.10.0/build/install")
SET(PCL_DIR "/home/edith/Software/pcl-1.10.0/build")
SET(PCL_INCLUDE_DIRS "/home/edith/Software/pcl-1.10.0/build/install/include/pcl-1.10")
list(APPEND CMAKE_MODULE_PATH "/home/edith/Software/pcl-1.10.0/cmake/Modules")
find_package(PCL REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(3d_obj_map src/ObjectMap.cpp src/Transform.cpp)
TARGET_LINK_LIBRARIES(3d_obj_map ${PCL_LIBRARIES} ${OpenCV_LIBS} Boost::program_options Boost::filesystem Eigen3::Eigen)

add_executable(scene_comparison src/SceneComparisons.cpp)
TARGET_LINK_LIBRARIES(scene_comparison ${OpenCV_LIBS} Boost::program_options Boost::filesystem Eigen3::Eigen)

add_executable(evaluation src/BaselineEvaluation.cpp)
TARGET_LINK_LIBRARIES(evaluation ${PCL_LIBRARIES} ${OpenCV_LIBS} Boost::program_options Boost::filesystem Eigen3::Eigen)



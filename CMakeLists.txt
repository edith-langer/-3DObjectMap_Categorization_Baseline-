cmake_minimum_required(VERSION 3.10)
project(3d_obj_map)

set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

find_package(OpenCV 3 REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(3d_obj_map src/ObjectMap.cpp src/Transform.cpp)
TARGET_LINK_LIBRARIES(3d_obj_map ${OpenCV_LIBS} Boost::program_options Boost::filesystem Eigen3::Eigen)

add_executable(scene_comparison src/SceneComparisons.cpp)
TARGET_LINK_LIBRARIES(scene_comparison ${OpenCV_LIBS} Boost::program_options Boost::filesystem Eigen3::Eigen)



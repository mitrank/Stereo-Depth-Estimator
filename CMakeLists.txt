cmake_minimum_required(VERSION 3.20.0.rc1)

project(stereo_depth_estimator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(Python COMPONENTS Development)
find_package( OpenCV REQUIRED )

#include_directories(External/armadillo-10.2.1)
include_directories(External/matplotlib-cpp ${OpenCV_INCLUDE_DIRS})

add_executable(stereo_depth_estimator src/main.cpp)

target_include_directories(stereo_depth_estimator PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(stereo_depth_estimator armadillo ${Python_LIBRARIES} ${OpenCV_LIBS})

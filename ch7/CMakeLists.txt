cmake_minimum_required( VERSION 3.0 )
set(CMAKE_CXX_FLAGS "-std=c++14 -mfma")
project(ch7)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(orbcv ./src/orb_cv.cpp)
# opencv的路径OpenCV_LIBS
target_link_libraries(orbcv ${OpenCV_LIBS})

add_executable(orbself ./src/orb_self.cpp)
target_link_libraries(orbself ${OpenCV_LIBS})

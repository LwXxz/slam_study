cmake_minimum_required(VERSION 3.0 )

#链接c++11
set(CMAKE_CXX_STANDARD 11)
project(rgbd)

add_executable(joinMap ./src/joinMap.cpp)

#链接OpenCV库 添加头文件
find_package(OpenCV REQUIRED) 
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

# Eigen
include_directories("/usr/local/include/eigen3")
target_link_libraries(joinMap ${OpenCV_LIBS}  ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} fmt)



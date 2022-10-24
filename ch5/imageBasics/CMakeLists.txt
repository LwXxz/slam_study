cmake_minimum_required( VERSION 3.0 )

project( imageBasics )

find_package (OpenCV 4.5.4 REQUIRED)
include_directories ("/usr/local/include/opencv4")
include_directories ("/usr/local/include/eigen3")

add_executable(imageBasics src/imageBasics.cpp)
# 链接opencv
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage src/undistortImage.cpp)
# 连接opencv
target_link_libraries(undistortImage ${OpenCV_LIBS})
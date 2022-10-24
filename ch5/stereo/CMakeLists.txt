cmake_minimum_required( VERSION 3.0 )
set(CMAKE_CXX_STANDARD 17)
project( stereo )

add_executable( stereo ./src/stereo.cpp )

find_package(OpenCV 4.5.4 REQUIRED) 
include_directories ("/usr/local/include/opencv4")
find_package(Pangolin REQUIRED)

target_link_libraries(stereo ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
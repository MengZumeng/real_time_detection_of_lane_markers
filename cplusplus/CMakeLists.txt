cmake_minimum_required(VERSION 3.5.1)
project(untitled1)
add_definitions("-Wall -g")
add_definitions(-std=c++14)
find_package(OpenCV 2.4 REQUIRED)
set(CMAKE_CXX_STANDARD 14)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(untitled1 mcvGetIPM.cpp)

target_link_libraries(untitled1 ${OpenCV_LIBS})
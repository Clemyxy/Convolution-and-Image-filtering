cmake_minimum_required(VERSION 3.20)
project(TP1)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

set(SRCS src/ATIVTP.cpp src/Convolution.cpp src/Filter.cpp)

add_executable(TP1 ${SRCS})

target_link_libraries(TP1 ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.0)

project(deeplab_v3_Test)

SET(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -Wall -std=c++11)

set(INC_DIR /usr/local/include include)
set(LINK_DIR /usr/local/lib)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

set(Lib_DIR so)
link_directories(${Lib_DIR})


add_library(TFlite_Lib tensorflowlite)
SET_TARGET_PROPERTIES(TFlite_Lib PROPERTIES OUTPUT_NAME "hello")

find_package(OpenCV 3 REQUIRED)

include_directories(include)

add_executable(deeplab_v3_Test
        include/
        ImageSegmentation.hpp
        ImageSegmentation.cpp
        tflite_segment.h
        tflite_segment.cpp)
        
target_link_libraries(deeplab_v3_Test ${OpenCV_LIBS})
target_link_libraries(deeplab_v3_Test ${TFlite_Lib})

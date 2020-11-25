set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER "/opt/fullhan/toolchain/gcc-arm-fullhan-eabi-6-2017-q2-update/bin/arm-fullhan-eabi-gcc")
set(CMAKE_CXX_COMPILER "/opt/fullhan/toolchain/gcc-arm-fullhan-eabi-6-2017-q2-update/bin/arm-fullhan-eabi-g++")
#--specs=nosys.specs
#set(CMAKE_C_COMPILER "arm-fullhan-linux-uclibcgnueabi-gcc")
#set(CMAKE_CXX_COMPILER "arm-fullhan-linux-uclibcgnueabi-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=armv6 -mfloat-abi=softfp -mfpu=neon --specs=nosys.specs")
set(CMAKE_CXX_FLAGS "-march=armv6 -mfloat-abi=softfp -mfpu=neon --specs=nosys.specs")
#set(OpenMP_CXX_FLAGS "")
#set(OpenMP_C_FLAGS "")


#LINK_DIRECTORIES(../libs)
#set(CMAKE_LIBS liblinux-atmoic.a)

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

add_definitions(-D__ARM_NEON)

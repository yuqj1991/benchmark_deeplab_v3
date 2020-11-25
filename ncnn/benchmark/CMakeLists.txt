add_library(linux-atmoic STATIC IMPORTED)
set_property(TARGET linux-atmoic PROPERTY IMPORTED_LOCATION /home/yuqianjin/workspace/ncnn/libs/liblinux-atmoic.a)

add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")

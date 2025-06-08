cmake_minimum_required(VERSION 3.10)
project(Fenrir VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)

add_executable(fenrir
    src/fenrir.cpp
)

target_include_directories(fenrir PUBLIC
    inc
)

target_link_libraries(fenrir PUBLIC caer)
cmake_minimum_required(VERSION 3.10)
project(Fenrir VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Add the subdirectories that contain source files and build logic
add_subdirectory(src)

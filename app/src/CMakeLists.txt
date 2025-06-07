add_library(Fenrir STATIC
    fenrir.cpp
    ../inc/fenrir.hpp
)

target_include_directories(Fenrir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
)

target_link_libraries(Fenrir PUBLIC caer)
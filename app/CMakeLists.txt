cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(Fenrir VERSION 1.0)
set(Fenrir_VERSION 1.0)

function(PREPEND var )
   set(listVar "")
   foreach(f ${${var}})
       list(APPEND listVar "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
   endforeach(f)
   set(${var} "${listVar}" PARENT_SCOPE)
endfunction(PREPEND)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

add_subdirectory(src)
add_subdirectory(inc)

add_library(Fenrir STATIC ${FenrirSRC} ${FenrirINC})

target_link_libraries(Fenrir caer)

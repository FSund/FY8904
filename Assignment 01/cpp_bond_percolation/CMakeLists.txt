cmake_minimum_required(VERSION 3.6)

project(bondperc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# don't use extended variants of language dialects (like -std=gnu++11 instead of -std=c++11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as no build type was specified")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Debug/Release)" FORCE)
endif (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

set(SOURCES
#    src/main_old.cpp
    src/main.cpp
    src/sites.cpp
    src/logbinom.cpp
    src/results.cpp
)

# manually add header files to project, for them to show up in QtCreator without
# having to restart each time a new header/source is added
# https://stackoverflow.com/a/33630604
# TODO: This is only done since QtCreator is a bit slow to update headers, so
# can probably remove this when the project is closer to finished
FILE(GLOB_RECURSE LibFiles "*.hpp")
add_custom_target(headers SOURCES ${LibFiles})

add_executable(${PROJECT_NAME} ${SOURCES}) # The main executable

find_package(Armadillo REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

find_package(HDF5 REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})  # not used
target_link_libraries(${PROJECT_NAME} ${HDF5_LIBRARIES})

# had to adjust armadillo_bits/config.hpp, and comment out the line
#     #define ARMA_SUPERLU_INCLUDE_DIR /
# to get SuperLU to work properly
find_package(SuperLU REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${SUPERLU_INCLUDE_DIRS})  # not used
target_link_libraries(${PROJECT_NAME} ${SUPERLU_LIBRARIES})
#message("SUPERLU_LIBRARIES: ${SUPERLU_LIBRARIES}")
#message("SUPERLU_INCLUDE_DIRS: ${SUPERLU_INCLUDE_DIRS}")


find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES})

target_link_libraries(${PROJECT_NAME} stdc++fs) # filesystem

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ARMA_DONT_USE_WRAPPER
    ARMA_USE_SUPERLU
    ARMA_USE_HDF5
    ARMA_USE_LAPACK
    ARMA_USE_BLAS
    ARMA_USE_ARPACK
    ARMA_PRINT_ERRORS
)

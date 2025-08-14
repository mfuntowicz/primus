cmake_minimum_required(VERSION 3.20)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 23)
project(primus LANGUAGES C CXX VERSION 1.0.0)

option(PRIMUS_BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
set(PRIMUS_STABLEHLO_BUILD_DIR CACHE STRING "Path to StableHLO build")

if(${PRIMUS_BUILD_WITH_CCACHE})
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    else()
        find_program(SCCACHE_PROGRAM sccache)
        if(SCCACHE_PROGRAM)
            set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
            set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
        else()
            message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
        endif ()
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Default build type: Release")
    set(CMAKE_BUILD_TYPE "ReleaseWithDebInfo")
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++23")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()

# #########
# LLVM
# #########
if(NOT MLIR_DIR)
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

if(NOT LLD_DIR)
    set(LLD_DIR ${LLVM_LIBRARY_DIR}/cmake/lld)
endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# StableHLO
if(PRIMUS_STABLEHLO_BUILD_DIR)
    include_directories("${PRIMUS_STABLEHLO_BUILD_DIR}/..")
    include_directories("${PRIMUS_STABLEHLO_BUILD_DIR}")
    link_directories("${PRIMUS_STABLEHLO_BUILD_DIR}/lib")
else()
    message(FATAL_ERROR "StableHLO build not specified. Please relaunch with PRIMUS_STABLEHLO_BUILD_DIR=../..")
endif()

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}) # Tablegen'd files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(PRIMUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PRIMUS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PRIMUS_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(primus)
cmake_minimum_required(VERSION 3.25)
project(Primus)

set(CMAKE_CXX_STANDARD 23)

# Build variables
set(PRIMUS_LLVM_BUILD_DIR CACHE STRING "Path to LLVM build")
option(PRIMUS_ENABLE_BINDINGS_PYTHON "Enable building the python binding" OFF)
option(PRIMUS_ENABLE_LLD "Use LLD as linker if available" OFF)

set(CMAKE_CXX_FLAGS "-fno-rtti")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Third-party dependencies
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# LLVM
#-------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/llvm-project/llvm/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/llvm-project/mlir/cmake/modules")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# StableHLO
#-------------------------------------------------------------------------------
set(STABLEHLO_BUILD_EMBEDDED ON)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON ${PRIMUS_ENABLE_BINDINGS_PYTHON})
add_subdirectory("third-party/stablehlo")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third-party/stablehlo")
include_directories("${CMAKE_BINARY_DIR}/third-party/stablehlo")

#-------------------------------------------------------------------------------
# spdlog
#-------------------------------------------------------------------------------
add_subdirectory("third-party/spdlog")

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------
if (PRIMUS_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif ()

add_subdirectory(unicron)

#-------------------------------------------------------------------------------
# Performance configuration
#-------------------------------------------------------------------------------
include(HandleLLVMOptions)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
if (PRIMUS_ENABLE_LLD)
    message(STATUS "Enabling LLD as the linker")
    add_link_options("-fuse-ld=lld")
endif ()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------
set(PRIMUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PRIMUS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PRIMUS_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(primus)
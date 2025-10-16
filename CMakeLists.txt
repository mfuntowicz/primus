cmake_minimum_required(VERSION 3.25)
project(primus)

set(CMAKE_CXX_STANDARD 23)

# Look around for LLVM dependencies
find_package(LLVM REQUIRED CONFIG)

# Include LLVM CMake modules
include(AddLLVM)
include(HandleLLVMOptions)

add_subdirectory(examples)
add_subdirectory(tlang)


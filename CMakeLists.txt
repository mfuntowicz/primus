cmake_minimum_required(VERSION 3.25)
project(primus)
set(CMAKE_CXX_STANDARD 23)

option(TLANG_BUILD_TESTS OFF "Enable building tlang unit-tests")

include(FetchContent)

# Look around for LLVM dependencies
find_package(LLVM REQUIRED CONFIG)

# Include LLVM CMake modules
include(AddLLVM)
include(HandleLLVMOptions)

if (${TLANG_BUILD_TESTS})
    message("Enabling unit-tests")
    fetchcontent_declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2
            GIT_TAG v3.11.0
    )
    fetchcontent_makeavailable(Catch2)
endif ()

add_subdirectory(examples)
add_subdirectory(tlang)
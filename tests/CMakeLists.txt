message(STATUS "Building Primus Tests Suite")
add_executable(primus-tests analysis.cpp)
target_link_libraries(primus-tests PRIVATE Catch2::Catch2WithMain PrimusOpsAnalysis)

include(CTest)
include(Catch)
catch_discover_tests(primus-tests)
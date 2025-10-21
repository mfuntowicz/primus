add_subdirectory(frontend)

if (${TLANG_BUILD_TESTS})
    add_subdirectory(tests)
endif ()
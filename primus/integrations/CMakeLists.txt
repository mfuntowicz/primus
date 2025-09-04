add_subdirectory(c)


if (PRIMUS_ENABLE_BINDINGS_PYTHON)
    add_subdirectory(python)
endif ()
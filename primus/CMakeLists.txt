
# Heavily inspired from Triton's repository: https://github.com/triton-lang/triton/blob/main/CMakeLists.txt
function(add_primus_object name)
    cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
    add_library(${name} OBJECT)
    target_sources(${name}
            PRIVATE ${ARG_UNPARSED_ARGUMENTS}
            INTERFACE $<TARGET_OBJECTS:${name}>
    )


    # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    if(ARG_LINK_LIBS)
        target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
    endif()
endfunction()

set_property(GLOBAL PROPERTY PRIMUS_LIBS "")
function(add_primus_library name)
    set_property(GLOBAL APPEND PROPERTY PRIMUS_LIBS ${name})
    add_primus_object(${name} ${ARGN})
    llvm_update_compile_flags(${name})
endfunction()

add_subdirectory(conversions)
add_subdirectory(dialect)
add_subdirectory(tools)
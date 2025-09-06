set(LLVM_OPTIONAL_SOURCES
        PrimusOptMain.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
message(STATUS "Conversion libs: ${conversion_libs}")
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRRegisterAllDialects
        MLIRRegisterAllExtensions
        MLIRRegisterAllPasses
        StablehloRegister
        StablehloPasses
        PrimusRegister
        PrimusStablehloTransforms
)
add_llvm_executable(primus-opt PrimusOptMain.cpp)
llvm_update_compile_flags(primus-opt)
target_link_libraries(primus-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(primus-opt)
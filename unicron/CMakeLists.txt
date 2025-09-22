add_library(Unicron STATIC
        errors.hpp
        runtime.cpp
        runtimes/cpu.cpp
        artifact.cpp
        artifact.hpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(UNICRON_MLIR_LIBS
        MLIRBuiltinToLLVMIRTranslation
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRSupport
)

llvm_update_compile_flags(Unicron)
llvm_map_components_to_libnames(UNICRON_LLVM_LIBS core native orcjit support)
target_link_libraries(Unicron
        ${UNICRON_LLVM_LIBS}
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        ${UNICRON_MLIR_LIBS}
        MLIRJitRunner
        MLIRExecutionEngineUtils
        mlir_c_runner_utils
        PrimusOps
        #        PrimusOpsTransforms
        PrimusRegister
        PrimusLinalgTransforms
        spdlog::spdlog
)


add_executable(UnicronRunner runner.cpp)
target_link_libraries(UnicronRunner PRIVATE Unicron mlir_c_runner_utils)
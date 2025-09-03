set(LLVM_TARGET_DEFINITIONS Base.td)
mlir_tablegen(BaseAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(BaseAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(PrimusBaseIncGen)
add_dependencies(mlir-headers PrimusBaseIncGen)

add_mlir_library(PrimusBase
        PARTIAL_SOURCES_INTENDED
        Base.cpp

        DEPENDS
        PrimusBaseIncGen

        LINK_LIBS PUBLIC
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRShapeDialect
        MLIRSupport
)


set(LLVM_TARGET_DEFINITIONS PrimusOps.td)
mlir_tablegen(PrimusOps.h.inc -gen-op-decls)
mlir_tablegen(PrimusOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(PrimusOpsIncGen)
add_dependencies(mlir-headers PrimusOpsIncGen)

add_mlir_dialect_library(PrimusOps
        PARTIAL_SOURCES_INTENDED
        PrimusOps.cpp

        DEPENDS
        PrimusOpsIncGen

        LINK_LIBS PUBLIC
        PrimusBase
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRQuantDialect
        MLIRShapeDialect
        MLIRSupport
        MLIRTensorDialect
)

target_include_directories(PrimusOps INTERFACE
        $<BUILD_INTERFACE:${PRIMUS_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PRIMUS_BINARY_DIR}>
)
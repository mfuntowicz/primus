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

add_mlir_dialect_library(PrimusRegister
        PARTIAL_SOURCES_INTENDED
        Register.cpp

        DEPENDS
        PrimusOpsIncGen

        LINK_LIBS PUBLIC
        PrimusOps
        MLIRFuncDialect
        MLIRBufferizationDialect
        MLIRIndexDialect
        MLIRShapeDialect
        #        StablehloOps
)

set(LLVM_TARGET_DEFINITIONS PrimusOps.td)
mlir_tablegen(PrimusOps.h.inc -gen-op-decls)
mlir_tablegen(PrimusOps.cpp.inc -gen-op-defs)
mlir_tablegen(PrimusAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(PrimusAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(PrimusOpsIncGen)
add_dependencies(mlir-headers PrimusOpsIncGen)

add_mlir_dialect_library(PrimusOps
        PARTIAL_SOURCES_INTENDED
        PrimusOps.cpp

        DEPENDS
        PrimusOpsIncGen

        LINK_LIBS PUBLIC
        PrimusBase
        #        StablehloAssemblyFormat
        #        StablehloOps
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


set(LLVM_TARGET_DEFINITIONS OptimusOps.td)
mlir_tablegen(OptimusOps.h.inc -gen-op-decls)
mlir_tablegen(OptimusOps.cpp.inc -gen-op-defs)
mlir_tablegen(OptimusAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(OptimusAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(OptimusOpsIncGen)
add_dependencies(mlir-headers OptimusOpsIncGen)


add_mlir_dialect_library(OptimusOps
        PARTIAL_SOURCES_INTENDED
        OptimusOps.cpp

        DEPENDS
        OptimusOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
)

target_include_directories(OptimusOps INTERFACE
        $<BUILD_INTERFACE:${PRIMUS_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PRIMUS_BINARY_DIR}>
)

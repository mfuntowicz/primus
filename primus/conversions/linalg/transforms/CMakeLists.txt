set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PrimusLinalgTransforms)
add_public_tablegen_target(PrimusLinalgTransformsPassIncGen)

add_mlir_library(PrimusLinalgTransforms
        PrimusLegalizeToLinalg.cpp
        Rewriters.h

        DEPENDS
        PrimusOpsIncGen
        PrimusLinalgTransformsPassIncGen

        LINK_LIBS PUBLIC
        PrimusBase
        PrimusOps
        MLIRArithDialect
        MLIRBufferizationDialect
        MLIRComplexDialect
        MLIRFuncDialect
        MLIRIR
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLinalgUtils
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRPass
        MLIRSCFDialect
        MLIRShapeDialect
        MLIRSparseTensorDialect
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        MLIRVectorDialect
)
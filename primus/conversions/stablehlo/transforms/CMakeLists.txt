set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PrimusStablehloTransforms)
add_public_tablegen_target(PrimusStablehloTransformsPassIncGen)

add_mlir_library(PrimusStablehloTransforms
        PrimusLegalizeToStablehlo.cpp
        PrimusToStablehloElementwise.cpp
        Rewriters.h

        DEPENDS
        PrimusOpsIncGen
        PrimusStablehloTransformsPassIncGen

        LINK_LIBS PUBLIC
        StablehloBase
        StablehloOps
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
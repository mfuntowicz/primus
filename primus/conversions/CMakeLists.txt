set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name PrimusToStablehlo)
add_public_tablegen_target(
    PrimusStablehloTransformsPassIncGen
)

add_primus_library(PrimusToStablehlo
    Rewriters.hpp
    PrimusLegalizeToStablehlo.cpp

    DEPENDS
    PrimusStablehloTransformsPassIncGen

    LINK_LIBS PUBLIC
    StablehloBase
    ChloOps
    StablehloOps
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRShapeDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    MLIRVectorDialect
    PrimusOps
)
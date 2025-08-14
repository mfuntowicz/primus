set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name PrimusToStablehlo)
add_public_tablegen_target(
    PrimusStablehloTransformsPassIncGen
)

add_primus_library(PrimusToStablehlo
    Passes.hpp
    StablehloUtils.hpp
    StablehloUtils.cpp
    ElementwiseOpToStablehlo.cpp
    PrimusLegalizeToStablehlo.cpp

    DEPENDS
    PrimusStablehloTransformsPassIncGen

    LINK_LIBS PUBLIC
    ChloOps
#    StablehloBase
    StablehloOps
#    StablehloTypeConversion
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
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PrimusStablehloTransforms)
add_public_tablegen_target(PrimusStablehloTransformsPassIncGen)

add_mlir_library(PrimusStablehloTransforms
        PrimusLegalizeToStablehlo.cpp

        DEPENDS
        PrimusOpsIncGen
        PrimusStablehloTransformsPassIncGen
)
#ifndef PRIMUS_CONVERSIONS_PASSES
#define PRIMUS_CONVERSIONS_PASSES

include "mlir/Pass/PassBase.td"

def PrimusLegalizeToStablehlo: Pass<"primus-legalize-to-stablehlo"> {
    let summary = "Convert high-level Primus operators to StableHLO";
    let dependentDialects = [
        "mlir::math::MathDialect",
//        "mlir::stablehlo::StablehloDialect",
    ];
    let options = [];
}

#endif //PRIMUS_CONVERSIONS_PASSES
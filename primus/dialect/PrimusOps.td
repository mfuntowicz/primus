#ifndef PRIMUS_OP
#define PRIMUS_OP

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "primus/dialect/PrimusTypes.td"

def Primus_Dialect : Dialect {
    let name = "primus";
    let description = [{
        Primus is a high-level, LLM focused, set of operators lowering to StableHLO.
        Original intent targets generating high-performance kernels for common operations involved in modern Large
        Language Models (LLM).
    }];

    let cppNamespace = "mlir::primus";
    let dependentDialects = [
        "linalg::LinalgDialect"
        "stablehlo:StableHloDialect"
    ];
}

class Primus_Op<string mnemonic, list<Trait> traits = []> :
    Op<Primus_Dialect, mnemonic, traits>;

def Primus_SwiGluOp: Primus_Op<"swiglu", [NoMemoryEffect, Elementwise, SameOperandsAndResultShape, Pure]> {
    let summary = "Swish activation followed by a Gated Linear Unit (GLU)";
    let arguments = (
        ins Primus_FloatTensor:$x
    );

    let results = (outs Primus_FloatTensor:$out);
}

#endif  // PRIMUS_OP
#ifndef PRIMUS_TO_LINALG_PASSES
#define PRIMUS_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def StablehloLegalizeToLinalgPass
    : Pass<"stablehlo-legalize-to-linalg"> {
  let summary = "Legalize StableHLO to LinAlg";
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::complex::ComplexDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::shape::ShapeDialect",
    "mlir::sparse_tensor::SparseTensorDialect",
  ];
  let options = [Option<"enablePrimitiveOps", "enable-primitive-ops", "bool",
                        /*default=*/"false",
                        "Lower to primitive Linalg ops (map, reduce and "
                        "transpose) when possible, instead of linalg.generic">,
                 Option<"enableSparseOps", "enable-sparse-ops", "bool",
                        /*default=*/"false",
                        "Lower to Sparse Tensor ops (sparse_tensor.concatenate)"
                        "when possible, instead of linalg.generic">];
}

#endif  // PRIMUS_TO_LINALG_PASSES
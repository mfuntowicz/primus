#ifndef PRIMUS_TO_LINALG_PASSES
#define PRIMUS_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def PrimusLegalizeToStablehloPass
    : Pass<"primus-legalize-to-stablehlo"> {
  let summary = "Legalize Primus to Stablehlo";
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::complex::ComplexDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::shape::ShapeDialect",
    "mlir::sparse_tensor::SparseTensorDialect",
    "mlir::stablehlo::StablehloDialect"
  ];
  let options = [];
}

#endif  // PRIMUS_TO_LINALG_PASSES
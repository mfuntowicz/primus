#ifndef PRIMUS_TO_LINALG_PASSES
#define PRIMUS_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def PrimusLegalizeToLinalgPass
    : Pass<"primus-legalize-to-linalg"> {
  let summary = "Legalize Primus to Linalg";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::index::IndexDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::shape::ShapeDialect",
    "mlir::tensor::TensorDialect"
  ];
  let options = [];
}

#endif //PRIMUS_TO_LINALG_PASSES
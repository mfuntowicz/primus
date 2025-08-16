#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "primus/dialect/Register.hpp"

int main(int argc, char **argv) {
    mlir::primus::registerAllPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);

    mlir::primus::registerAllDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);

    return failed(
        mlir::MlirOptMain(argc, argv, "Primus optimizer driver\n", registry));
}
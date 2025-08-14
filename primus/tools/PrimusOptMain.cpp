#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "primus/conversions/Passes.hpp"
#include "primus/dialect/Register.hpp"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::primus::registerPrimusToStablehloPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);

    mlir::primus::registerAllDialects(registry);

    return failed(
        mlir::MlirOptMain(argc, argv, "Primus optimizer driver\n", registry));
}
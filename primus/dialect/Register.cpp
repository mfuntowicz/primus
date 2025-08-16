//
// Created by mfuntowicz on 8/14/25.
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

#include "primus/dialect/PrimusDialect.hpp"
#include "primus/dialect/Register.hpp"
#include "primus/conversions/Passes.hpp"


namespace mlir::primus
{
    void registerAllDialects(DialectRegistry &registry) {
        // clang-format off
        registry.insert<func::FuncDialect, PrimusDialect>();
        // registry.insert<
        //     chlo::ChloDialect,
        //     stablehlo::StablehloDialect,
        //     vhlo::VhloDialect
        // >();
        // clang-format on
    }

    void registerAllPasses() {
        registerPrimusLegalizeToStablehloPass();

        stablehlo::registerStablehloLegalizeToLinalgPass();
        stablehlo::registerStablehloAggressiveSimplificationPass();
        stablehlo::registerStablehloRefineShapesPass();
        stablehlo::registerStablehloConvertToSignlessPass();
        stablehlo::registerShapeLegalizeToStablehloPass();
        stablehlo::registerStablehloLegalizeDeprecatedOpsPass();
    }

}
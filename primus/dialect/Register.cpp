//
// Created by mfuntowicz on 8/14/25.
//

#include "primus/dialect/Register.hpp"


#include "PrimusDialect.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include <stablehlo/dialect/StablehloOps.h>
#include "primus/dialect/PrimusOps.hpp"

namespace mlir::primus
{
    void registerAllDialects(DialectRegistry &registry)
    {
        registry.insert<func::FuncDialect, stablehlo::StablehloDialect>();
        registry.insert<PrimusDialect>();
    }

}
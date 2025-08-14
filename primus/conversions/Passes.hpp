//
// Created by mfuntowicz on 8/13/25.
//

#ifndef PRIMUS_PASSES_HPP
#define PRIMUS_PASSES_HPP

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
    class ModuleOp;

    namespace primus
    {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "primus/conversions/Passes.h.inc"

    }
}  // namespace mlir

#endif //PRIMUS_PASSES_HPP
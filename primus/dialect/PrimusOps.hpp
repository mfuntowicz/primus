//
// Created by mfuntowicz on 8/13/25.
//

#ifndef PRIMUS_PRIMUSOPS_HPP
#define PRIMUS_PRIMUSOPS_HPP

#include <optional>

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "primus/dialect/Ops.h.inc"

#endif //PRIMUS_PRIMUSOPS_HPP
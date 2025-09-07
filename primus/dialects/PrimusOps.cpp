// Copyright Morgan Funtowicz (c) 2025.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*

//
// Created by Morgan Funtowicz on 9/2/2025.
//

#include "PrimusOps.h"

#include <mlir/IR/OpDefinition.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include "primus/dialects/AssemblyFormat.h"

namespace mlir::primus {
    /**
     * Ensure the following requirements are met:
     * - `x` input dimensions can be dynamic except the last one (fastest moving)
     * - `x` latest dimension is known and even
     * - `x`, `cos` and `sin` have the same last dimension
     * @return
     */
    LogicalResult RotaryOp::verify() {
        const auto x = dyn_cast<RankedTensorType>(getX().getType());
        if (x.isDynamicDim(x.getRank() - 1)) {
            emitOpError("operand `x` last dimension should be static");
            return failure();
        }

        if (x.getShape().back() % 2 != 0) {
            emitOpError("operand `x` last dimension should be divisible by 2");
            return failure();
        }

        const auto cos = dyn_cast<RankedTensorType>(getCos().getType());
        const auto sin = dyn_cast<RankedTensorType>(getSin().getType());
        if (!llvm::all_equal({x.getShape().back(), cos.getShape().back(), sin.getShape().back()})) {
            emitOpError("operands `x`, `cos` and `sin` should have the same trailing dimension value");
            return failure();
        }

        return success();
    }

    PrimusDialect::PrimusDialect(MLIRContext *context)
        : Dialect(getDialectNamespace(), context, TypeID::get<PrimusDialect>()) {
        addOperations<
#define GET_OP_LIST
#include "primus/dialects/PrimusOps.cpp.inc"
        >();
    }
}

#define GET_OP_CLASSES
#include "primus/dialects/PrimusOps.cpp.inc"

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

#include "llvm/ADT/TypeSwitch.h"
#include <llvm/Support/Format.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include "primus/dialects/AssemblyFormat.h"

#define GET_ATTRDEF_CLASSES
#include "primus/dialects/PrimusAttrs.cpp.inc"

namespace mlir::primus {
    /**
     * Ensure the following requirements are met:
     * - `x` input dimensions can be dynamic except the last one (fastest moving)
     * - `x` latest dimension is known and even
     * - `x`, `cos` and `sin` have the same last dimension
     * @return
     */
    LogicalResult RotaryOp::verify() {
        const auto xTy = cast<RankedTensorType>(getX().getType());
        if (!(xTy.getRank() == 4 && !xTy.isDynamicDim(3))) {
            emitOpError("operand `x` should be a 4D tensor with static last dimension");
            return failure();
        }

        if (xTy.getDimSize(3) % 2 != 0) {
            emitOpError("operand `x` last dimension should be divisible by 2");
            return failure();
        }

        const auto cos = dyn_cast<RankedTensorType>(getCos().getType());
        const auto sin = dyn_cast<RankedTensorType>(getSin().getType());
        if (cos.getShape() != sin.getShape()) {
            emitOpError("operands `cos` and `sin` should have the same shape");
            return failure();
        }

        // Attributes
        if (xTy.getDimSize(3) != static_cast<int64_t>(getHeadSize())) {
            emitOpError("operand `x` last dimension should match attribute `head_size`");
            return failure();
        }

        if (cos.getDimSize(cos.getRank() - 1) != static_cast<int64_t>(getHeadSize()) / 2) {
            emitOpError("operand `cos` last dimension should match attribute `head_size`");
            return failure();
        }

        // This one should not be necessary as we are checking shape equality between cos and sin
        // But it would provide a proper op error in case of ...
        if (sin.getDimSize(sin.getRank() - 1) != static_cast<int64_t>(getHeadSize()) / 2) {
            emitOpError("operand `sin` last dimension should match attribute `head_size`");
            return failure();
        }

        if (getHeadSize() % 2 != 0) {
            emitOpError("attribute `head_size` should be divisible by 2");
            return failure();
        }

        return success();
    }

    PrimusDialect::PrimusDialect(MLIRContext *context)
        : Dialect(getDialectNamespace(), context, TypeID::get<PrimusDialect>()) {
        addOperations <

#define GET_OP_LIST
#include "primus/dialects/PrimusOps.cpp.inc"
    >
    (
    
    );
}

}

#define GET_OP_CLASSES
#include "primus/dialects/PrimusOps.cpp.inc"
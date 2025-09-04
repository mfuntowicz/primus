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

#include <mlir/Dialect/Vector/IR/VectorOps.h>


namespace mlir::primus {
    LogicalResult RotaryOp::verify() {
        auto adaptor = RotaryOp::Adaptor(getOperands());

        // Inputs
        // Cosine and sinus tensors are of the same shape as defined in the PrimusOps.td
        const auto xTy = dyn_cast<RankedTensorType>(adaptor.getX().getType());
        const auto cosTy = dyn_cast<RankedTensorType>(adaptor.getCos().getType());

        // Both `x` and `cosinus` tensors trailing dimension should match
        if (xTy.getShape().back() != cosTy.getShape().back())
            return failure();

        return success();
    }
}

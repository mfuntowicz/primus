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

#ifndef PRIMUS_BASE_H
#define PRIMUS_BASE_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include order matters
#include "primus/dialects/BaseAttrInterfaces.h.inc"

namespace mlir::primus {
    // Inspired from StableHLO SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
    // href: https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Base.h#L494C8-L494C63
    template<typename ConcreteType>
    struct SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
            : OpTrait::TraitBase<ConcreteType, SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait> {
        // An elementwise op is not speculatable if a dimension of the result
        // type is static while the corresponding dimension in the input type is
        // dynamic. Indeed, the input dimension could differ at runtime.
        // If the output dimension is dynamic, there is no expectation, so there
        // cannot be a mismatch.
        // If the input dimension is static, the output dimension can be inferred from
        // it, so there cannot be a mismatch.

        Speculation::Speculatability getSpeculatability() {
            auto op = this->getOperation();
            auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());

            for (auto resultType = cast<RankedTensorType>(op->getResult(0).getType()); size_t i: llvm::seq(
                     resultType.getRank())) {
                if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
                    return Speculation::NotSpeculatable;
            }
            return Speculation::Speculatable;
        }
    };
}

#endif //PRIMUS_BASE_H
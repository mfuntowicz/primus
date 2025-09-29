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
// Created by mfuntowicz on 9/21/25.
//

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <primus/dialects/PrimusOps.h>

namespace mlir::primus
{
    struct RotaryOpInterface : bufferization::BufferizableOpInterface::ExternalModel<RotaryOpInterface, RotaryOp>
    {
        constexpr static int64_t X_OPERAND_INDEX = 0;

        LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                                const bufferization::BufferizationOptions& options) const
        {
            // Your bufferization logic here
            auto rotaryOp = cast<primus::RotaryOp>(op);
            // Convert tensor operands to memrefs
            // Create memref-based version of rotary operation
            // for (auto operand : rotaryOp.getOperands())
            // {
            //     assert(llvm::isa<RankedTensorType>(operand.getType()) && "all operand should be ranked-tensor type");
            //
            //     const auto opTy = cast<RankedTensorType>(operand.getType());
            //
            //     bufferization::replaceOpWithNewBufferizedOp<>()
            //     return success();
            // }
            return success();
        }

        bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                                    const bufferization::AnalysisState& state) const
        {
            return true;
        }

        bool bufferizesToMemoryWrite(Operation* op, OpOperand& opOperand,
                                     const bufferization::AnalysisState& state) const
        {
            return opOperand.getOperandNumber() == X_OPERAND_INDEX;
        }
    };
}
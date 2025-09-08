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
// Created by momo- on 9/5/2025.
//

#include <tuple>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "primus/conversions/stablehlo/transforms/LegalizeToStablehloUtils.h"
#include "primus/dialects/PrimusOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::primus {
    namespace {
        // Helper to build slice parameters for one chunk
        struct SliceParams {
            SmallVector<Value> startIndices;
            SmallVector<Value> limitIndices;
            SmallVector<Value> strides;
        };
        
        // SliceParams buildSliceParams(OpBuilder &builder, Location loc, Value input,
        //                            ArrayRef<int64_t> shape, int64_t chunkStart, int64_t chunkEnd) {
        //     SliceParams params;
        //     int64_t rank = shape.size();
        //
        //     for (int64_t i = 0; i < rank; ++i) {
        //         // Strides are always 1
        //         params.strides.push_back(createI64Constant(builder, loc, 1));
        //
        //         if (i == rank - 1) {
        //             // Last dimension - chunking dimension
        //             params.startIndices.push_back(createI64Constant(builder, loc, chunkStart));
        //             params.limitIndices.push_back(createI64Constant(builder, loc, chunkEnd));
        //         } else {
        //             // Leading dimensions - always start at 0
        //             params.startIndices.push_back(createI64Constant(builder, loc, 0));
        //
        //             // Limit depends on whether the dimension is static or dynamic
        //             if (shape[i] == ShapedType::kDynamic) {
        //                 params.limitIndices.push_back(getDimensionSizeI64(builder, loc, input, i));
        //             } else {
        //                 params.limitIndices.push_back(createI64Constant(builder, loc, shape[i]));
        //             }
        //         }
        //     }
        //
        //     return params;
        // }
        //
        // // Helper function to chunk an SSA value into two equal parts over the last dimension
        // // Handles dynamic dimensions in leading axes
        // std::pair<Value, Value> chunkLastDimension(MlirBuilder& builder, Value input) {
        //     auto inputType = llvm::cast<RankedTensorType>(input.getType());
        //     auto shape = inputType.getShape();
        //
        //     if (shape.empty()) {
        //         return {input, input};
        //     }
        //
        //     int64_t lastDimSize = shape.back();
        //     assert(lastDimSize != ShapedType::kDynamic && "Last dimension must be static for chunking");
        //
        //     int64_t chunkSize = lastDimSize / 2;
        //
        //     // Build parameters for both chunks
        //     auto params1 = buildSliceParams(builder, loc, input, shape, 0, chunkSize);
        //     auto params2 = buildSliceParams(builder, loc, input, shape, chunkSize, lastDimSize);
        //
        //     // Create RealDynamicSlice operations
        //     auto chunk1 = builder.create<stablehlo::RealDynamicSliceOp>(
        //         loc, input, params1.startIndices, params1.limitIndices, params1.strides);
        //     auto chunk2 = builder.create<stablehlo::RealDynamicSliceOp>(
        //         loc, input, params2.startIndices, params2.limitIndices, params2.strides);
        //
        //     return {chunk1.getResult(), chunk2.getResult()};
        // }

        struct RotaryOpConverter final : OpConversionPattern<RotaryOp> {
            using OpConversionPattern<RotaryOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                auto builder = MlirBuilder(rewriter, op.getLoc());
                // auto [xHead, xTail] = chunkLastDimension(builder, adaptor.getX());

                llvm::outs() << "Converting " << op << "\n";

                // auto xDim0 = getDimensionSizeI64(builder, op, 0);
                auto op_ = MlirOp(builder, adaptor.getX());
                auto xDim0 = stablehlo::GetDimensionSize(op_, 0);
                rewriter.replaceOp(op, xDim0.getValue());
                return success();
            }
        };
    }

    namespace detail {
        void populateElementwisePrimusToStablehloConversionPatterns(
            MLIRContext *context, RewritePatternSet *patterns) {
            patterns->add<RotaryOpConverter>(context);
        }
    }
}
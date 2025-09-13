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

#include <ranges>
#include <tuple>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

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
            using OpConversionPattern::OpConversionPattern;

            static llvm::SmallVector<int64_t, 4> getChunkedShape(const llvm::ArrayRef<int64_t> &shape) {
                return {shape[0], shape[1], shape[2], shape[3] / 2};
            }

            // LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
            //     auto builder = MlirBuilder(rewriter, op.getLoc());
            //     auto xOp = MlirOp(builder, op.getX());
            //     auto xTy = cast<RankedTensorType>(xOp.getType());
            //
            //     // Get dimensions of heading dimensions
            //     auto xDims = \
            //         std::views::iota(0, xTy.getRank())
            //         | std::views::transform([&](auto rank) { return getDimensionAsTensor(xOp, rank, rewriter.getI64Type()); })
            //         | std::ranges::to<llvm::SmallVector<MlirOp, 4>>();
            //
            //     auto rankOneTy = RankedTensorType::get({1}, rewriter.getI64Type());
            //     auto xDim3 = stablehlo::Constant(builder, DenseIntElementsAttr::get(rankOneTy, {xTy.getDimSize(3)}));
            //     auto xDim3Half = stablehlo::Constant(builder, DenseIntElementsAttr::get(rankOneTy, {xTy.getDimSize(3) / 2}));
            //
            //     // Constants
            //     auto xZero = stablehlo::Constant(builder, DenseIntElementsAttr::get(rankOneTy, { 0l }));
            //     auto xSlices = stablehlo::Constant(builder, {1, 1, 1, 1});
            //     auto xChunkedTy = RankedTensorType::get(getChunkedShape(xTy.getShape()), xTy.getElementType());
            //
            //     // [0, 0, 0, 0], [xDim0, xDim1, xDim2, xDim3Half], [1, 1, 1, 1]
            //     auto xHeadStarts = stablehlo::Constant(builder, {0, 0, 0, 0});
            //     auto xHeadLimits = stablehlo::Concatenate(builder, {xDims[0], xDims[1], xDims[2], xDim3Half}, 0);
            //     auto xHead = stablehlo::RealDynamicSlice(xChunkedTy, xOp, xHeadStarts, xHeadLimits, xSlices);
            //
            //     // [0, 0, 0, xDim3Half], [xDim0, xDim1, xDim2, xDim3], [1, 1, 1, 1]
            //     auto xTailStarts = stablehlo::Concatenate(builder, {xZero, xZero, xZero, xDim3Half}, 0);
            //     auto xTailLimits = stablehlo::Concatenate(builder, {xDims[0], xDims[1], xDims[2], xDim3}, 0);
            //     auto xTail = stablehlo::RealDynamicSlice(xChunkedTy, xOp, xTailStarts, xTailLimits, xSlices);
            //
            //     // Rotate
            //     auto xCos = MlirOp(builder, op.getCos());
            //     auto xSin = MlirOp(builder, op.getSin());
            //
            //     auto xCosSinTy = cast<RankedTensorType>(xCos.getType());
            //     auto xCosSinShape = xCosSinTy.getShape();
            //
            //     // Expand cos and sin (b, s, heads) -> (b, 1, s, heads)
            //     auto xCosSinDims = \
            //         std::views::iota(0, xCosSinTy.getRank())
            //         | std::views::transform([&](auto rank) { return getDimensionAsTensor(xCos, rank, rewriter.getI64Type()); })
            //         | std::ranges::to<llvm::SmallVector<MlirOp, 4>>();
            //
            //     auto xFreqsExpTy = RankedTensorType::get({xCosSinShape[0], xTy.getShape()[1], xCosSinShape[1], xCosSinShape[2]}, xCosSinTy.getElementType());
            //     auto xFreqsExp = stablehlo::Concatenate(builder, {xCosSinDims[0], xDims[1], xCosSinDims[1], xCosSinDims[2]}, 0);
            //
            //     auto xCosExp = stablehlo::DynamicReshape(xFreqsExpTy, xCos, xFreqsExp);
            //     auto xSinExp = stablehlo::DynamicReshape(xFreqsExpTy, xSin, xFreqsExp);
            //
            //     auto xHeadCos = stablehlo::Mul(xHead, xCosExp);
            //     auto xTailSin = stablehlo::Mul(xTail, xSinExp);
            //     auto xHeadRotated = stablehlo::Subtract(xHeadCos, xTailSin);
            //
            //     auto xTailCos = stablehlo::Mul(xTail, xCosExp);
            //     auto xHeadSin = stablehlo::Mul(xHead, xSinExp);
            //     auto xTailRotated = stablehlo::Add(xTailCos, xHeadSin);
            //
            //     auto xRotated = stablehlo::Concatenate(builder, {xHeadRotated, xTailRotated}, 3);
            //     rewriter.replaceOp(op, xRotated.getValue());
            //     return success();
            // }

            LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const override {
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
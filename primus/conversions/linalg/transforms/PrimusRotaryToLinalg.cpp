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
// Created by momo- on 9/13/2025.
//

#include "Rewriters.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "primus/dialects/PrimusOps.h"

namespace mlir::primus {
    namespace
    {
        // Function to split the last dimension of a tensor into two equal parts
        std::pair<Value, Value> splitLastDimensionInTwo(
            ConversionPatternRewriter &rewriter, const Location loc, Value src, const RankedTensorType srcTy) {
            const auto shape = srcTy.getShape();

            // Assume all dimensions are static
            const int64_t lastDimSize = shape[3];
            const int64_t halfSize = lastDimSize / 2;

            // Create offsets and sizes for extract_slice - all static
            SmallVector<OpFoldResult> offsets, sizes, strides;

            for (int64_t i = 0; i < srcTy.getRank(); ++i) {
                if (i == 3) {
                    // For dimension 3 (last dimension we're splitting)
                    sizes.push_back(rewriter.getIndexAttr(halfSize));
                } else {
                    // All other dimensions are static now
                    sizes.push_back(rewriter.getIndexAttr(shape[i]));
                }
                offsets.push_back(rewriter.getIndexAttr(0)); // Start at 0 for the first half
                strides.push_back(rewriter.getIndexAttr(1));
            }

            // Create the result type for both halves
            SmallVector<int64_t> halfShape(shape.begin(), shape.end());
            halfShape[3] = halfSize;
            auto halfType = RankedTensorType::get(halfShape, srcTy.getElementType());

            // Extract first half
            Value firstHalf = rewriter.create<tensor::ExtractSliceOp>(
                loc, halfType, src, offsets, sizes, strides);

            // Update offset for the second half (starts at halfSize in dimension 3)
            offsets[3] = rewriter.getIndexAttr(halfSize);

            // Extract second half
            Value secondHalf = rewriter.create<tensor::ExtractSliceOp>(
                loc, halfType, src, offsets, sizes, strides);

            return {firstHalf, secondHalf};
        }

        // Function to apply rotary embedding using linalg.generic
        Value applyRotary(
            ConversionPatternRewriter& rewriter, const Location loc, const Value x, const Value cos, const Value sin,
            const RankedTensorType xTy)
        {
            // Chunk `x` in two equal parts
            auto [xHeadVal, xTailVal] = splitLastDimensionInTwo(rewriter, loc, x, xTy);

            // Get types - all static now
            auto halfType = cast<RankedTensorType>(xHeadVal.getType());

            // Create output tensors - no dynamic sizes needed since all dimensions are static
            Value outputHead = rewriter.create<tensor::EmptyOp>(
                loc, halfType.getShape(), halfType.getElementType());
            Value outputTail = rewriter.create<tensor::EmptyOp>(
                loc, halfType.getShape(), halfType.getElementType());

            const auto rank = halfType.getRank();

            // Create affine maps
            auto* context = rewriter.getContext();

            // Identity map for input/output tensors (all dimensions)
            auto inputAffineMap = AffineMap::getMultiDimIdentityMap(rank, context);

            // Broadcast map for cos/sin tensors (3D -> 4D: [batch, seq, head_dim])
            auto cosAffineMap = AffineMap::get(rank, 0, {
                                                   rewriter.getAffineDimExpr(0), // batch
                                                   rewriter.getAffineDimExpr(2), // sequence
                                                   rewriter.getAffineDimExpr(3) // head dimension
                                               }, context);

            SmallVector<AffineMap> indexingMaps = {
                inputAffineMap, // xHeadVal
                inputAffineMap, // xTailVal
                cosAffineMap, // cos (3D -> 4D broadcast)
                cosAffineMap,   // sin (3D -> 4D broadcast)
                inputAffineMap, // outputHead
                inputAffineMap // outputTail
            };

            SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

            // Create a single linalg.generic operation that yields both rotated values
            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc,
                TypeRange{halfType, halfType},
                ValueRange{xHeadVal, xTailVal, cos, sin},
                ValueRange{outputHead, outputTail},
                indexingMaps,
                iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value xHead = args[0];
                    Value xTail = args[1];
                    Value cosVal = args[2];
                    Value sinVal = args[3];

                    // Compute rotation formulas
                    // First half: xHead * cos - xTail * sin
                    Value headCos = b.create<arith::MulFOp>(loc, xHead, cosVal);
                    Value tailSin = b.create<arith::MulFOp>(loc, xTail, sinVal);
                    Value rotatedHead = b.create<arith::SubFOp>(loc, headCos, tailSin);

                    // Second half: xTail * cos + xHead * sin
                    Value tailCos = b.create<arith::MulFOp>(loc, xTail, cosVal);
                    Value headSin = b.create<arith::MulFOp>(loc, xHead, sinVal);
                    Value rotatedTail = b.create<arith::AddFOp>(loc, tailCos, headSin);

                    // Yield both results
                    b.create<linalg::YieldOp>(loc, ValueRange{rotatedHead, rotatedTail});
                });

            // Concatenate the two halves back together
            SmallVector<OpFoldResult> concatOffsets, concatSizes, concatStrides;
            const auto shape = xTy.getShape();
            const int64_t halfSize = shape[3] / 2;

            // Initialize for concatenation - all static
            for (int64_t i = 0; i < xTy.getRank(); ++i)
            {
                if (i == 3)
                {
                    concatSizes.push_back(rewriter.getIndexAttr(halfSize));
                } else {
                    concatSizes.push_back(rewriter.getIndexAttr(shape[i]));
                }
                concatOffsets.push_back(rewriter.getIndexAttr(0));
                concatStrides.push_back(rewriter.getIndexAttr(1));
            }

            // Create the final output tensor
            Value finalOutput = rewriter.create<tensor::EmptyOp>(
                loc, xTy.getShape(), xTy.getElementType());

            // Insert first half (rotatedHead)
            Value result = rewriter.create<tensor::InsertSliceOp>(
                loc, genericOp.getResult(0), finalOutput,
                concatOffsets, concatSizes, concatStrides);

            // Update offset for second half
            concatOffsets[3] = rewriter.getIndexAttr(halfSize);

            // Insert second half (rotatedTail)
            result = rewriter.create<tensor::InsertSliceOp>(
                loc, genericOp.getResult(1), result,
                concatOffsets, concatSizes, concatStrides);

            return result;
        }

        /**
         * Define the lowering from `RotaryOp` aka `primus.rotary` to a combination of `linalg` operations
         */
        struct RotaryOpConverter final : OpConversionPattern<RotaryOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override
            {
                const auto loc = op.getLoc();

                // Apply rotary embedding
                const Value result =
                    applyRotary(rewriter, loc, op.getX(), op.getCos(), op.getSin(),
                                cast<RankedTensorType>(op.getX().getType()));

                rewriter.replaceOp(op, result);
                return success();
            }
        };
    }

    namespace detail {
        void populatePrimusRotaryToLinalgConversionPatterns(MLIRContext *context, RewritePatternSet *patterns) {
            // Add the basic rotary converter
            patterns->add<RotaryOpConverter>(context);
        }
    }
}
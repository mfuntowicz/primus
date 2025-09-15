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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "primus/dialects/PrimusOps.h"

namespace mlir::primus {

    namespace {
        // Function to split the last dimension of a tensor into two equal parts
        std::pair<Value, Value> splitLastDimensionInTwo(
            ConversionPatternRewriter &rewriter, const Location loc, Value src, const RankedTensorType srcTy) {
            const auto shape = srcTy.getShape();

            // The last dimension (axis 3) is always static
            const int64_t lastDimSize = shape[3];
            const int64_t halfSize = lastDimSize / 2;
            
            // Create offsets and sizes for extract_slice
            SmallVector<OpFoldResult> offsets, sizes, strides;
            
            for (int64_t i = 0; i < srcTy.getRank(); ++i) {
                if (i == 3) {
                    // For dimension 3 (last dimension we're splitting)
                    sizes.push_back(rewriter.getIndexAttr(halfSize));
                } else {
                    // Other dimensions may be dynamic
                    if (ShapedType::isDynamic(shape[i])) {
                        const Value dimSize = rewriter.create<tensor::DimOp>(loc, src, i);
                        sizes.push_back(dimSize);
                    } else {
                        sizes.push_back(rewriter.getIndexAttr(shape[i]));
                    }
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
            
            return { firstHalf, secondHalf };
        }

        // Function to apply rotary embedding using linalg.generic
        Value applyRotary(
            ConversionPatternRewriter &rewriter, 
            const Location loc, 
            Value x, 
            Value cos, 
            Value sin, 
            const RankedTensorType xTy) {

            // Chunk `x` in two equal parts
            auto [xHeadVal, xTailVal] = splitLastDimensionInTwo(rewriter, loc, x, xTy);

            // Get dynamic dimensions for EmptyOp
            auto halfType = cast<RankedTensorType>(xHeadVal.getType());
            SmallVector<Value> halfDynamicSizes;
            SmallVector<Value> fullDynamicSizes;
            
            for (int64_t i = 0; i < halfType.getRank(); ++i) {
                if (ShapedType::isDynamic(halfType.getShape()[i])) {
                    Value dimSize = rewriter.create<tensor::DimOp>(loc, xHeadVal, i);
                    halfDynamicSizes.push_back(dimSize);
                }
            }
            
            for (int64_t i = 0; i < xTy.getRank(); ++i) {
                if (ShapedType::isDynamic(xTy.getShape()[i])) {
                    Value dimSize = rewriter.create<tensor::DimOp>(loc, x, i);
                    fullDynamicSizes.push_back(dimSize);
                }
            }

            // Create output tensors with proper dynamic sizes
            Value outputHead = rewriter.create<tensor::EmptyOp>(
                loc, halfType.getShape(), halfType.getElementType(), halfDynamicSizes);
            Value outputTail = rewriter.create<tensor::EmptyOp>(
                loc, halfType.getShape(), halfType.getElementType(), halfDynamicSizes);

            const auto rank = halfType.getRank();
            
            // Create affine maps
            // For input tensors (xHeadVal, xTailVal, outputHead, outputTail)
            SmallVector<AffineExpr> inputExprs;
            for (unsigned i = 0; i < rank; ++i) {
                inputExprs.push_back(rewriter.getAffineDimExpr(i));
            }
            auto inputAffineMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());

            // For cos/sin tensors (3D -> 4D broadcast)
            SmallVector<AffineExpr> cosExprs;
            cosExprs.push_back(rewriter.getAffineDimExpr(0)); // batch
            cosExprs.push_back(rewriter.getAffineDimExpr(2)); // sequence
            cosExprs.push_back(rewriter.getAffineDimExpr(3)); // head dimension
            auto cosAffineMap = AffineMap::get(rank, 0, cosExprs, rewriter.getContext());

            SmallVector<AffineMap> indexingMaps = {
                inputAffineMap,  // xHeadVal
                inputAffineMap,  // xTailVal
                cosAffineMap,    // cos (shared affine_map)
                cosAffineMap,    // sin (shared affine_map)
                inputAffineMap,  // outputHead
                inputAffineMap   // outputTail
            };

            SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

            // Create single linalg.generic operation that yields both rotated values
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

            // Initialize for concatenation
            for (int64_t i = 0; i < xTy.getRank(); ++i) {
                if (i == 3) {
                    concatSizes.push_back(rewriter.getIndexAttr(halfSize));
                } else {
                    if (ShapedType::isDynamic(shape[i])) {
                        Value dimSize = rewriter.create<tensor::DimOp>(loc, x, i);
                        concatSizes.push_back(dimSize);
                    } else {
                        concatSizes.push_back(rewriter.getIndexAttr(shape[i]));
                    }
                }
                concatOffsets.push_back(rewriter.getIndexAttr(0));
                concatStrides.push_back(rewriter.getIndexAttr(1));
            }

            // Create final output tensor with proper dynamic sizes
            Value finalOutput = rewriter.create<tensor::EmptyOp>(
                loc, xTy.getShape(), xTy.getElementType(), fullDynamicSizes);

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

        struct RotaryOpConverter final : OpConversionPattern<RotaryOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();

                // Apply rotary embedding
                const Value result =
                    applyRotary(rewriter, loc, op.getX(), op.getCos(), op.getSin(), op.getX().getType());

                rewriter.replaceOp(op, result);
                return success();
            }
        };
    }

    namespace detail {
        void populatePrimusRotaryToLinalgConversionPatterns(MLIRContext *context, RewritePatternSet *patterns) {
            patterns->add<RotaryOpConverter>(context);
        }
    }
}
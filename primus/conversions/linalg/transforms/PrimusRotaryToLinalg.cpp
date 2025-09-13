//
// Created by momo- on 9/13/2025.
//

#include <ranges>

#include "Rewriters.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "primus/dialects/PrimusOps.h"

namespace mlir::primus {

    namespace {
        struct RotaryOpConverter final : OpConversionPattern<RotaryOp> {
            using OpConversionPattern::OpConversionPattern;

            template<size_t Chunks>
            static std::array<OpFoldResult, Chunks> chunkTensorAlong(
                OpBuilder& rewriter, const Location loc, const Value& operand, const ShapedType& source, size_t dim) {
                std::array<OpFoldResult, Chunks> results;

                // Create the resulting shape and type
                auto chunkShape = llvm::SmallVector<int64_t>(source.getShape());
                chunkShape.back() /= Chunks;

                const auto chunkTy = RankedTensorType::get(chunkShape, source.getElementType());

                // SSA definition
                auto vOffsets = SmallVector<OpFoldResult>{0, 0, 0, 0};
                const auto vHead = tensor::ExtractSliceOp::create(
                    rewriter, loc, chunkTy, operand, vOffsets, );

                return results;
            }

            LogicalResult matchAndRewrite(RotaryOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();

                // Inputs
                const auto vX = adaptor.getX();
                const auto vCos = adaptor.getCos();
                const auto vSin = adaptor.getSin();

                // Validation
                if (const auto vXShape = cast<ShapedType>(vX); vXShape.isStaticDim(vXShape.getRank() - 1)) {
                    auto [vXHead, vXTail] = \
                        chunkTensorAlong<2>(rewriter, loc, vX, vXShape, vXShape.getRank() - 1);

                } else {
                    op.emitOpError("last dimension of operand `x` should be static.");
                    return failure();
                }

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
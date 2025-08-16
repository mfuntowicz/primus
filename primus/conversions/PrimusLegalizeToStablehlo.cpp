//
// Created by mfuntowicz on 8/13/25.
//
#include <memory>
#include <llvm/Support/Format.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "primus/conversions/Rewriters.hpp"
#include "primus/conversions/StablehloUtils.hpp"
#include "primus/dialect/PrimusOps.hpp"

#define DEBUG_TYPE "primus-legalize-to-stablehlo"


namespace mlir::primus
{
#define GEN_PASS_DEF_PRIMUSLEGALIZETOSTABLEHLO
#include "primus/conversions/Passes.h.inc"

    namespace
    {
        struct SiluOpConverter final : OpConversionPattern<SiluOp>
        {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(SiluOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
            {
                // Input references
                const auto xRef = adaptor.getX();

                // Ensure we operate on RankedTensor
                if (!isa<RankedTensorType>(xRef.getType()))
                {
                    op.emitError("SwiGluOp: `x` must be a ranked tensor");
                    return failure();
                }

                // Retrieve the shape of the `x` input
                const auto xTy = cast<RankedTensorType>(xRef.getType());
                const auto xShape = xTy.getShape();
                if (xShape.empty())
                {
                    op.emitError("SwiGluOp: `x` cannot be a scalar (0-rank tensor)");
                    return failure();
                }

                auto xLogistic = rewriter.create<stablehlo::LogisticOp>(op.getLoc(), op.getResult().getType(), xRef);
                rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, op.getResult().getType(), xRef, xLogistic);

                return success();
            }
        };

        struct PrimusLegalizeToStablehloPass final: impl::PrimusLegalizeToStablehloBase<PrimusLegalizeToStablehloPass> {
        private:
            std::shared_ptr<ConversionTarget> target;
            FrozenRewritePatternSet patterns;

            using PrimusLegalizeToStablehloBase::PrimusLegalizeToStablehloBase;

            LogicalResult initialize(MLIRContext* context) override
            {
                target = std::make_shared<ConversionTarget>(*context);
                target->addLegalDialect<stablehlo::StablehloDialect>();

                RewritePatternSet patterns_(context);
                populatePrimusLegalizeToStablehloPatterns(context, patterns_);
                patterns = std::move(patterns_);

                return success();
            }

            void runOnOperation() override
            {
                if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
                    return signalPassFailure();
                }
            }

            void getDependentDialects(DialectRegistry& registry) const override
            {
                registry.insert<chlo::ChloDialect, stablehlo::StablehloDialect>();
            }
        };
    }

    void populatePrimusLegalizeToStablehloPatterns(MLIRContext* context, RewritePatternSet &patterns)
    {
        patterns.add<SiluOpConverter>(context);
    }
}

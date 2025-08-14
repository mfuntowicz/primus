//
// Created by mfuntowicz on 8/13/25.
//
#include <memory>
#include <primus/dialect/PrimusOps.hpp>

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "primus/conversions/StablehloUtils.hpp"

namespace mlir::primus
{
    struct SwigluOpConverter final : OpConversionPattern<SwiGluOp>
    {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(SwiGluOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
        {
            // Ensure we have 2 operands
            if (adaptor.getOperands().size() != 2)
            {
                return failure();
            }

            // Create the decomposition body
            // rewriter.create(swiglu.getLoc(), adaptor.getOperands(), [&](OpBuilder &b, Location loc, ValueRange args)
            // {
            //     // Get the type of `x`
            //     auto dtype = args[0].getType();
            //
            //     // Compute `swish(x, beta) = x * sigmoid(beta * x)
            //     const auto one = b.create<stablehlo::ConstantOp>(loc, adaptor.getOperands(), dtype);
            //     // const auto
            //     // const auto negX = b.create<stablehlo::NegOp>(loc, adaptor.getOperands(), dtype);
            //     // const auto expNegX = b.create<stablehlo::ExpOp>(loc, negX, dtype);
            //     // const auto onePlusExpNegX = b.create<stablehlo::AddOp>(loc, one, expNegX, dtype);
            //     // const auto swish = b.create<stablehlo::DivOp>(loc, one, onePlusExpNegX, dtype);
            // });
            //
            // // Create the stablehlo::composite call
            // rewriter.create<stablehlo::CompositeOp>(swiglu.getLoc(), adaptor.getOperands(), "primus.swiglu", "swiglu", 1);

            if (const auto constTensor = toStableHloConstTensor(rewriter, op, llvm::ArrayRef({1.0f}), {1}); constTensor.has_value())
            {
                rewriter.replaceOp(op, constTensor.value());
                return success();
            }

            return failure();
        }
    };


#define GEN_PASS_DEF_PRIMUSLEGALIZETOSTABLEHLO
#include "primus/conversions/Passes.h.inc"

    void populatePrimusLegalizeToStablehloPatterns(
        MLIRContext* context, RewritePatternSet &patterns)
    {
        patterns.add<SwigluOpConverter>(context);
    }

    struct PrimusLegalizeToStablehloPass final: impl::PrimusLegalizeToStablehloBase<PrimusLegalizeToStablehloPass> {
    private:
        std::shared_ptr<ConversionTarget> target;
        FrozenRewritePatternSet patterns;

        using PrimusLegalizeToStablehloBase::PrimusLegalizeToStablehloBase;
        using OpAdaptor = SwiGluOp::Adaptor;

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
    };
}

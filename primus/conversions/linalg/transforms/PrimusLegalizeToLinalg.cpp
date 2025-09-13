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
// Created by momo- on 9/12/2025.
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "primus/conversions/linalg/transforms/Passes.h"
#include "primus/conversions/linalg/transforms/Rewriters.h"

namespace mlir::primus {
#define GEN_PASS_DEF_PRIMUSLEGALIZETOLINALGPASS
#include "primus/conversions/linalg/transforms/Passes.h.inc"

    namespace {
        class PrimusLegalizeToLinalgPass final : public impl::PrimusLegalizeToLinalgPassBase<PrimusLegalizeToLinalgPass> {
            std::shared_ptr<ConversionTarget> target;
            FrozenRewritePatternSet patterns;

        public:
            using PrimusLegalizeToLinalgPassBase::PrimusLegalizeToLinalgPassBase;

            LogicalResult initialize(MLIRContext *context) override {
                target = std::make_shared<ConversionTarget>(*context);
                target->addLegalDialect<
                    arith::ArithDialect,
                    bufferization::BufferizationDialect,
                    index::IndexDialect,
                    math::MathDialect,
                    shape::ShapeDialect,
                    linalg::LinalgDialect,
                    tensor::TensorDialect
                >();

                RewritePatternSet patterns_(context);
                populatePrimusToLinalgConversionPatterns(context, &patterns_);
                patterns = std::move(patterns_);

                return success();
            }

            void runOnOperation() override {
                if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
                    return signalPassFailure();
                }
            }
        };
    }


    void populatePrimusToLinalgConversionPatterns(MLIRContext *ctx, RewritePatternSet *patterns) {
        detail::populatePrimusRotaryToLinalgConversionPatterns(ctx, patterns);
    }
}
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
// Created by momo- on 9/4/2025.
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "primus/conversions/stablehlo/transforms/Passes.h"
#include "primus/conversions/stablehlo/transforms/Rewriters.h"

namespace mlir::primus {
#define GEN_PASS_DEF_PRIMUSLEGALIZETOSTABLEHLOPASS
#include "primus/conversions/stablehlo/transforms/Passes.h.inc"

    namespace {
        class PrimusLegalizeToStablehloPass final : public impl::PrimusLegalizeToStablehloPassBase<PrimusLegalizeToStablehloPass> {
            std::shared_ptr<ConversionTarget> target;
            FrozenRewritePatternSet patterns;
            TypeConverter converter;

        public:
            using PrimusLegalizeToStablehloPassBase::PrimusLegalizeToStablehloPassBase;

            LogicalResult initialize(MLIRContext *context) override {
                target = std::make_shared<ConversionTarget>(*context);

                RewritePatternSet patterns_(context);
                populatePrimusToStablehloConversionPatterns(context, converter, &patterns_);
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


    void populatePrimusToStablehloConversionPatterns(MLIRContext *ctx, TypeConverter &converter, RewritePatternSet *patterns) {
        detail::populateElementwisePrimusToStablehloConversionPatterns(ctx, converter, patterns);
    }
}

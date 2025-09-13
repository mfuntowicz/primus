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


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "primus/conversions/stablehlo/transforms/LegalizeToStablehloUtils.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace mlir::primus {
    /**
     * Helper to create a constant i64 value
     */
    MlirOp createI64Constant(MlirBuilder &builder, const int64_t value) {
        return stablehlo::Constant(builder, value);
    }

    MlirOp getDimensionAs(MlirOp &input, const int64_t dim, const IntegerType as) {
        auto builder = input.getBuilder().getOpBuilder();
        const auto loc = input.getValue().getLoc();

        // Use tensor.dim (returns index)
        auto dimConstant = arith::ConstantIndexOp::create(builder, loc, dim);
        auto tensorDim = tensor::DimOp::create(builder, loc, input.getValue(), dimConstant);

        // Convert index to i64 using arith.index_cast
        const auto i64Type = builder.getI64Type();
        auto dimI64 = arith::IndexCastOp::create(builder, loc, i64Type, tensorDim);

        // If the target type is not i64, convert again
        Value finalValue = dimI64;
        if (as != i64Type) {
            if (as.getWidth() < 64) {
                finalValue = builder.create<arith::TruncIOp>(loc, as, dimI64);
            } else {
                // This shouldn't happen for integer types, but handle it
                finalValue = dimI64;
            }
        }

        // Convert to tensor
        auto emptyTensor = tensor::FromElementsOp::create(builder, loc, finalValue);
        return MlirOp(input.getBuilder(), emptyTensor);
    }

    MlirOp getDimensionAsTensor(MlirOp &input, const int64_t dim, const IntegerType as) {
        const auto size = getDimensionAs(input, dim, as);
        return stablehlo::Reshape(size, {1});
    }
}
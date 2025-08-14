//
// Created by mfuntowicz on 8/14/25.
//

#ifndef PRIMUS_STABLEHLOUTILS_HPP
#define PRIMUS_STABLEHLOUTILS_HPP

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"


namespace mlir::primus
{
    template <typename T>
    Value toStableHloConstant(OpBuilder &rewriter, Location loc, T constant, Value val) {
        Type ty = getElementTypeOrSelf(val.getType());
        auto getAttr = [&]() -> Attribute {
            if (isa<IntegerType>(ty))
                return rewriter.getIntegerAttr(ty, constant);
            if (isa<FloatType>(ty))
                return rewriter.getFloatAttr(ty, constant);
            if (auto complexTy = dyn_cast<ComplexType>(ty))
                return complex::NumberAttr::get(complexTy, constant, 0);
            llvm_unreachable("unhandled element type");
        };

        return rewriter.create<chlo::ConstantLikeOp>(loc, cast<TypedAttr>(getAttr()), val);
    }

    template <typename T>
    std::optional<Value> toStableHloConstTensor(
        PatternRewriter &rewriter, Operation *op, ArrayRef<T> vec, const ArrayRef<int64_t> shape) {
        uint64_t num_total_elements = 1;
        for (const int64_t a : shape) {
            num_total_elements *= a;
        }

        if (vec.size() != num_total_elements) {
            op->emitOpError("getConstTensor(): number of elements mismatch.");
            return std::nullopt;
        }

        RankedTensorType const_type;
        if constexpr (std::is_same_v<T, APInt>) {
            const_type = RankedTensorType::get(
                shape, rewriter.getIntegerType(vec[0].getBitWidth()));
        } else if constexpr (std::is_same_v<T, float>) {
            const_type = RankedTensorType::get(shape, rewriter.getF32Type());
        } else if constexpr (std::is_same_v<T, double>) {
            const_type = RankedTensorType::get(shape, rewriter.getF64Type());
        } else {
            const_type =
                RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
        }
        auto const_attr = DenseElementsAttr::get(const_type, vec);

        auto const_op = rewriter.create<stablehlo::ConstantOp>(
            op->getLoc(), const_type, const_attr);
        return const_op.getResult();
    }
}

#endif //PRIMUS_STABLEHLOUTILS_HPP
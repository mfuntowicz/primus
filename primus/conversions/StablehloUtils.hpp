//
// Created by mfuntowicz on 8/14/25.
//

#ifndef PRIMUS_STABLEHLOUTILS_HPP
#define PRIMUS_STABLEHLOUTILS_HPP

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir::primus
{
    /**
     *
     * @tparam T
     * @param rewriter
     * @param loc
     * @param constant
     * @param val
     * @return
     */
    template <typename T> Value toStableHloConstant(OpBuilder &rewriter, Location loc, T constant, Value val);


    /**
     * Templated function to create constant op for a given type and shape in C-order.
     * @tparam T
     * @param rewriter
     * @param op
     * @param vec
     * @param shape
     * @return
     */
    template <typename T>
    std::optional<Value> toStableHloConstTensor(
        PatternRewriter &rewriter,
        Operation *op,
        ArrayRef<T> vec,
        ArrayRef<int64_t> shape
    );
}

#endif //PRIMUS_STABLEHLOUTILS_HPP
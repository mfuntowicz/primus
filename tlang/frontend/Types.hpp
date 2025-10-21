//
// Created by mfuntowicz on 10/17/25.
//

#ifndef PRIMUS_TYPES_HPP
#define PRIMUS_TYPES_HPP

#include <variant>

#include <llvm/ADT/SmallVector.h>

namespace tlang
{
    struct InferTy
    {
        bool operator==(const InferTy&) const = default;
    };

    static_assert(sizeof(InferTy) == 1);

    struct IntegerTy
    {
        uint8_t width;

        bool operator==(const IntegerTy other) const
        {
            return other.width == width;
        }
    };

    static_assert(sizeof(IntegerTy) == 1);

    struct SignedIntegerTy
    {
        uint8_t width;
    };

    static_assert(sizeof(SignedIntegerTy) == 1);

    struct FloatTy
    {
        uint8_t width;
        uint8_t mantissa;
        uint8_t exponent;
    };

    static_assert(sizeof(FloatTy) == 3);

    /**
     *
     */
    using ScalarTy = std::variant<IntegerTy, SignedIntegerTy, FloatTy>;

    constexpr int32_t DIMENSION_DYNAMIC = std::numeric_limits<int32_t>::min();

    /**
     *
     */
    struct TensorTy
    {
        llvm::SmallVector<int32_t, 4> shapes;
        ScalarTy dtype;

        [[nodiscard]] size_t Rank() const;
    };

    static_assert(sizeof(TensorTy) <= 64);

    /**
     *
     */
    using TensorOrScalarTy = std::variant<ScalarTy, TensorTy>;
    using InferrableTensorOrScalarTy = std::variant<InferTy, ScalarTy, TensorTy>;
}

#endif //PRIMUS_TYPES_HPP

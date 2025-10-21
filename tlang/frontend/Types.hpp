//
// Created by mfuntowicz on 10/17/25.
//

#ifndef PRIMUS_TYPES_HPP
#define PRIMUS_TYPES_HPP

#include <variant>
#include <llvm/ADT/SmallVector.h>
#include "Token.hpp"


namespace tlang
{
    /**
     * Helper type to represent a variable which type is not defined during parsing and should
     * be inferred at a later stage.
     */
    struct InferTy
    {
        constexpr bool operator==(const InferTy&) const { return true; }
    };

    static_assert(sizeof(InferTy) == 1);

    /**
     * Represent an unsigned, fixed-width, integer type,
     */
    struct IntegerTy
    {
        uint8_t width;

        constexpr bool operator==(const IntegerTy& other) const = default;
    };

    static_assert(sizeof(IntegerTy) == 1);

    struct SignedIntegerTy
    {
        uint8_t width;

        constexpr bool operator==(const SignedIntegerTy& other) const = default;
    };

    static_assert(sizeof(SignedIntegerTy) == 1);

    struct FloatTy
    {
        uint8_t width;
        uint8_t mantissa;
        uint8_t exponent;

        constexpr bool operator==(const FloatTy& other) const = default;
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

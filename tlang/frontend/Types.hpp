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


    /**
     * Helper type to represent a variable which type is not defined during parsing and should
     * be inferred at a later stage.
     */
    template <typename T>
    struct InferableTy
    {
        InferableTy() : type(std::nullopt)
        {
        }

        explicit InferableTy(T&& ty) : type(ty)
        {
        }

        /**
         *
         *
         * @return
         */
        bool NeedsInference() const
        {
            return !type.has_value();
        }

        /**
         *
         * @return
         */
        T Type() const
        {
            assert(!NeedsInference() && "No actual type defined, NeedInference() == true");
            return type.value();
        }

    private:
        std::optional<ScalarTy> type;
    };

    using InferableScalarTy = InferableTy<ScalarTy>;
}

#endif //PRIMUS_TYPES_HPP

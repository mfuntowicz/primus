//
// Created by mfuntowicz on 10/17/25.
//

#ifndef PRIMUS_TYPES_HPP
#define PRIMUS_TYPES_HPP

#include <variant>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/SmallVector.h>

namespace tlang
{
    struct IntegerTy
    {
        uint8_t width;

        bool operator==(const IntegerTy other) const
        {
            return other.width == width;
        }
    };

    struct SignedIntegerTy
    {
        uint8_t width;
    };

    struct FloatTy
    {
        uint8_t width;
        uint8_t mantissa;
        uint8_t exponent;
    };

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
    using TensorOrScalarTy = std::variant<TensorTy, ScalarTy>;
}

#endif //PRIMUS_TYPES_HPP

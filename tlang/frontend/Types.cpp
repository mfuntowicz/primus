//
// Created by mfuntowicz on 10/17/25.
//

#include "Types.hpp"

#include <expected>

namespace tlang
{
    size_t TensorTy::Rank() const
    {
        return shapes.size();
    }
}

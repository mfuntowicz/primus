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
// Created by mfuntowicz on 9/29/25.
//
#include "MemoryLayoutAnalysis.h"

#include <ranges>
#include <llvm/Support/FormatVariadic.h>

namespace mlir::primus
{
    bool greater_equal_and_not_dynamic(const int64_t lhs, const int64_t rhs)
    {
        return ShapedType::isStatic(lhs) && ShapedType::isStatic(rhs) && std::greater_equal<int64_t>()(lhs, rhs);
    }

    bool isContiguous(const MemRefType& type)
    {
        return isContiguous(type, 0);
    }


    bool isContiguous(const MemRefType& type, const uint64_t dim)
    {
        // Scalar
        if (!type.hasRank() || type.getRank() == 0)
            return true;

        // Ensure we are in-bound
        if (dim >= static_cast<uint64_t>(type.getRank()))
            return false;

        // Ensure the dim is not dynamic
        if (type.isDynamicDim(dim))
            return false;

        // If we ask for the last dimension of the tensor, check it has stride = 1
        const auto [strides, _] = type.getStridesAndOffset();
        if (dim == static_cast<uint64_t>(type.getRank() - 1))
            return strides.back() == 1;

        // Check contiguity by verifying each stride equals the product of subsequent dimensions
        const auto shape = type.getShape();
        for (auto i = type.getRank() - 1; i >= static_cast<int64_t>(dim); --i)
        {
            const auto expected_stride = std::ranges::fold_left(
                shape | std::views::drop(i + 1),
                1L,
                std::multiplies<int64_t>{}
            );

            if (strides[i] != expected_stride)
                return false;
        }

        return true;
    }
}

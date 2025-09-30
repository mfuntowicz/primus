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
// Created by mfuntowicz on 9/30/25.
//

#include <mlir-c/IR.h>

#include "catch2/catch_test_macros.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "primus/analysis/MemoryLayoutAnalysis.h"

TEST_CASE("Scalar", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);
    const auto buffer = mlir::MemRefType::get({}, f32);

    REQUIRE(mlir::primus::isContiguous(buffer));
    REQUIRE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE(mlir::primus::isContiguous(buffer, 1));
}

TEST_CASE("Contiguous Vector", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);
    const auto buffer = mlir::MemRefType::get({1}, f32);

    REQUIRE(mlir::primus::isContiguous(buffer));
    REQUIRE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 1));
}

TEST_CASE("Contiguous Tensor", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);
    const auto buffer = mlir::MemRefType::get({5, 7, 3}, f32);

    REQUIRE(mlir::primus::isContiguous(buffer));
    REQUIRE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE(mlir::primus::isContiguous(buffer, 1));
}

TEST_CASE("Strided Tensor (contiguous)", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);

    const auto strides = mlir::StridedLayoutAttr::get(&context, 0, {21, 3, 1});
    const auto buffer = mlir::MemRefType::get({5, 7, 3}, f32, strides);

    REQUIRE(mlir::primus::isContiguous(buffer));
    REQUIRE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE(mlir::primus::isContiguous(buffer, 1));
    REQUIRE(mlir::primus::isContiguous(buffer, 2));
}

TEST_CASE("Strided Tensor (non contiguous)", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);

    const auto strides = mlir::StridedLayoutAttr::get(&context, 0, {21, 1, 7});
    const auto buffer = mlir::MemRefType::get({5, 7, 3}, f32, strides);

    REQUIRE_FALSE(mlir::primus::isContiguous(buffer));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 1));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 2));
}

TEST_CASE("Strided Tensor (partially contiguous)", "[memory][layout]")
{
    mlir::MLIRContext context;
    const auto f32 = mlir::Float32Type::get(&context);

    const auto strides = mlir::StridedLayoutAttr::get(&context, 0, {30, 210, 10, 1});
    const auto buffer = mlir::MemRefType::get({5, 7, 3, 10}, f32, strides);

    REQUIRE_FALSE(mlir::primus::isContiguous(buffer));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 0));
    REQUIRE_FALSE(mlir::primus::isContiguous(buffer, 1));
    REQUIRE(mlir::primus::isContiguous(buffer, 2));
    REQUIRE(mlir::primus::isContiguous(buffer, 3));
}

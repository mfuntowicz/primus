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
// Created by momo- on 9/17/2025.
//

#ifndef UNICRON_ARTIFACT_HPP
#define UNICRON_ARTIFACT_HPP

#include <concepts>
#include <expected>
#include <memory>
#include <utility>

#include <mlir/ExecutionEngine/RunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include "spdlog/spdlog.h"

namespace unicron {
    /**
     * Traits to detect specialization of Result<T>
     */
    template <typename>
    struct is_execution_engine_result : std::false_type
    {
    };

    template <typename T>
    struct is_execution_engine_result<mlir::ExecutionEngine::Result<T>> : std::true_type
    {
    };

    /**
     * Evaluate the specialization of `Result<T>`
     * @tparam T Type of the underlying variable returned by the function
     */
    template <typename T>
    inline constexpr bool is_execution_engine_result_v = is_execution_engine_result<T>::value;

    /**
     * Concept to check if a type T is compatible with mlir::ExecutionEngine::Argument<T>::pack
     * and can be used as an argument to ExecutionEngine::invoke
     */
    template <typename... T>
    concept Packable = ((is_execution_engine_result_v<T> || std::convertible_to<T, void*>) && ...);

    struct compilation_artifact_t {
        /**
         *
         * @param name
         * @param dylib
         */
        compilation_artifact_t(const std::string_view name, std::shared_ptr<mlir::ExecutionEngine> dylib)
            : name(std::string(name)), dylib(std::move(dylib))
        {
        }

        template <Packable... Args>
        std::expected<void, std::string> invoke(std::string_view symbol, Args... args)
        {
            spdlog::trace("Invoking function `{}`", symbol);

            // Invoke the function
            if (auto result = dylib->invoke(symbol, args...))
            {
                spdlog::error("JIT {} invocation failed", symbol);
                return std::unexpected(llvm::toString(std::move(result)));
            }

            return {};
        }

    private:
        std::string name;
        std::shared_ptr<mlir::ExecutionEngine> dylib;
    };
}

#endif //UNICRON_ARTIFACT_HPP

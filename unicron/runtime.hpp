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
// Created by momo- on 9/15/2025.
//

#ifndef UNICRON_RUNTIME_H
#define UNICRON_RUNTIME_H

#include <expected>
#include <memory>

#include <llvm/ADT/StringMap.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/MLIRContext.h>

#include "artifact.hpp"
#include "errors.hpp"

namespace unicron
{
    static std::once_flag initialized;

    void initialize_llvm();


    struct target_info_t
    {
        const llvm::Target* target;
        std::unique_ptr<llvm::TargetMachine> machine;
    };

    /**
     *
     * @param triple
     * @return
     */
    std::expected<target_info_t, error_t> create_target_machine(std::string_view triple);

    struct runtime_t
    {
        virtual ~runtime_t() = default;

        /**
         * Create a runtime running on the current "native" machine
         */
        runtime_t();

        // We can't copy an execution engine
        runtime_t(runtime_t&&) = delete;

        /**
         * Compile and make the provided operation ready for its execution through this runtime
         * @param op
         * @param with_debug
         * @return
         */
        std::expected<std::shared_ptr<compilation_artifact_t>, error_t>
        register_module(mlir::Operation* op, bool with_debug = false);

        /**
         *
         * @param name
         * @return
         */
        std::optional<std::shared_ptr<compilation_artifact_t>> lookup(std::string_view name);

        /**
         *
         * @param op The input module representing the intermediate representation to be lowered.
         * @param context The context in which the LLVM IR will be generated.
         * @return A boolean indicating whether the lowering to LLVM was successful.
         */
        virtual std::unique_ptr<llvm::Module> lower_to_llvm(mlir::Operation* op, llvm::LLVMContext& context) = 0;

    private:
        std::string host_native_triple;
        llvm::StringMap<std::shared_ptr<compilation_artifact_t>> modules;
    };

    using FailureOr = llvm::FailureOr<runtime_t>;
} // unicron

#endif //UNICRON_RUNTIME_H
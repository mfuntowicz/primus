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

#include "runtime.hpp"

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/IR/Operation.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Transforms/DialectConversion.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include "artifact.hpp"

namespace unicron {
    void initialize_llvm() {
        std::call_once(initialized, []() {
            spdlog::trace("Initializing LLVM");
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
        });
    }

    runtime_t::runtime_t() : host_native_triple(llvm::sys::getProcessTriple())
    {
    }

    std::expected<target_info_t, error_t> create_target_machine(std::string_view sTriple) {
        const auto triple = llvm::Triple(sTriple.data());

        std::string onError;
        if (const auto* target = llvm::TargetRegistry::lookupTarget(triple, onError); target)
        {
            const auto cpu = llvm::sys::getHostCPUName();

            auto options = llvm::TargetOptions();
            options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
            options.ApproxFuncFPMath = true;
            options.UnsafeFPMath = true;

            auto* machine = target->createTargetMachine(
                triple,
                cpu,
                "",
                options,
                llvm::Reloc::PIC_,
                llvm::CodeModel::Medium,
                llvm::CodeGenOptLevel::Aggressive,
                true
            );

            auto pMachine = std::unique_ptr<llvm::TargetMachine>(machine);
            return target_info_t{target, std::move(pMachine)};
        }

        return std::unexpected(error_t::host_target_detection_failed(onError));
    }

    std::expected<std::shared_ptr<compilation_artifact_t>, error_t>
    runtime_t::register_module(mlir::Operation* const op, const bool with_debug) {
        const auto name = op->getName().getStringRef();

        if (modules.contains(name))
            return std::unexpected(
                error_t::module_already_exist(fmt::format(FMT_STRING("Module `{}` already exists"), name))
            );

        // Create the execution engine from the module definition
        const auto target_info = create_target_machine(host_native_triple);
        if (target_info) {
            spdlog::trace("Creating execution engine targeting architecture `{}`", host_native_triple);
            auto lowering_f = [&](mlir::Operation* op_, llvm::LLVMContext& ctx) { return lower_to_llvm(op_, ctx); };
            auto optimizer_f = mlir::makeOptimizingTransformer(3, 0, target_info->machine.get());
            const auto options = mlir::ExecutionEngineOptions(
                lowering_f,
                optimizer_f,
                llvm::CodeGenOptLevel::Aggressive,
                {},
                nullptr,
                true,
                with_debug,
                true
            );

            auto engine = mlir::ExecutionEngine::create(op, options);
            if (!engine) {
                auto error = engine.takeError();
                return std::unexpected(error_t::module_compilation_failed(llvm::toString(std::move(error))));
            }

            auto shared_engine = std::shared_ptr(std::move(*engine));
            auto artifact = std::make_shared<compilation_artifact_t>(name, shared_engine);

            modules.insert({name, artifact});

            return artifact;
        } else {
            return std::unexpected(target_info.error());
        }
    }
} // unicron

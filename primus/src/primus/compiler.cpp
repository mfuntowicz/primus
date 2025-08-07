#include <filesystem>
#include <fstream>
#include <utility>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "primus.hpp"

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace primus
{
    Compiler::Compiler(std::shared_ptr<mlir::MLIRContext> context, const mlir::ModuleOp module): context(std::move(context)), module(module)
    {
        if constexpr (IS_DEBUG)
        {
            SPDLOG_DEBUG("Initialized compiler with module:");
            module->dumpPretty();
        }
    }

    Compiler Compiler::FromFile(const std::filesystem::path& file)
    {
        SPDLOG_INFO("Processing kernel from {}", file.string());
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(file.string(), true);
        if (const std::error_code ec = fileOrErr.getError()) {
            throw std::runtime_error(spdlog::fmt_lib::format("Could not open input file {}: {}", file.string(), ec.message()));
        }

        // Register dialects
        mlir::DialectRegistry registry;
        mlir::register registerAllDialects(registry);
        mlir::stablehlo::registerAllDialects(registry);
        mlir::stablehlo::registerPasses();
        mlir::stablehlo::registerOptimizationPasses();
        mlir::stablehlo::registerStablehloLinalgTransformsPasses();

        // Parse the input mlir.
        auto context = std::make_shared<mlir::MLIRContext>(registry);
        llvm::SourceMgr sourceMgr;

        sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        mlir::OwningOpRef<mlir::ModuleOp> owningModule = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &*context);
        if (!owningModule) {
            throw std::runtime_error(spdlog::fmt_lib::format("Parsing of input file {} failed", file.string()));
        }

        return { context, owningModule.release() };
    }

    void Compiler::LowerTo(const CompilationTarget& target) const
    {
        mlir::ModuleOp ir(module);
        mlir::PassManager pm(context.get());

        target.LoweringPipeline(pm);
        if (const auto status = pm.run(ir); status.succeeded())
        {
            ir.dump();
        } else
        {
            SPDLOG_ERROR("Failed to lower to linalg");
        }
    }
}

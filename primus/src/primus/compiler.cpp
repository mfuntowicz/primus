#include <filesystem>
#include <fstream>

#include "primus/compiler.hpp"

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"


#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"

namespace primus
{
    Compiler::Compiler(const mlir::ModuleOp module):  module(module)
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
        mlir::stablehlo::registerAllDialects(registry);

        // Parse the input mlir.
        mlir::MLIRContext context(registry);
        llvm::SourceMgr sourceMgr;

        sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        mlir::OwningOpRef<mlir::ModuleOp> owningModule = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
        if (!owningModule) {
            throw std::runtime_error(spdlog::fmt_lib::format("Parsing of input file {} failed", file.string()));
        }

        return Compiler(owningModule.release());
    }
}
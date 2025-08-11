#include <filesystem>
#include <fstream>
#include <utility>
#include <mlir/Dialect/GPU/Pipelines/Passes.h>

#include "primus.hpp"

#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
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
        registry.insert<
            mlir::affine::AffineDialect,
            mlir::arith::ArithDialect,
            mlir::math::MathDialect,
            mlir::tensor::TensorDialect,
            mlir::gpu::GPUDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::ub::UBDialect
        >();
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerAllDialectInterfaceImplementations(registry);
        mlir::arith::registerConvertArithToLLVMInterface(registry);
        mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
        mlir::registerConvertNVVMToLLVMInterface(registry);

        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerNVVMDialectTranslation(registry);

        mlir::registerConvertComplexToLLVMInterface(registry);
        mlir::registerConvertFuncToLLVMInterface(registry);
        mlir::registerConvertMathToLLVMInterface(registry);
        mlir::registerConvertMemRefToLLVMInterface(registry);
        mlir::registerConvertNVVMToLLVMInterface(registry);
        mlir::vector::registerConvertVectorToLLVMInterface(registry);
        mlir::ub::registerConvertUBToLLVMInterface(registry);
        mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);

        mlir::stablehlo::registerAllDialects(registry);
        mlir::stablehlo::registerPasses();
        mlir::stablehlo::registerOptimizationPasses();
        mlir::stablehlo::registerStablehloLinalgTransformsPasses();

        // Parse the input mlir.
        auto context = std::make_shared<mlir::MLIRContext>(registry, GetThreadingMode<IS_DEBUG>());
        context->loadAllAvailableDialects();
        llvm::SourceMgr sourceMgr;

        sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        mlir::OwningOpRef<mlir::ModuleOp> owningModule = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &*context);
        if (!owningModule) {
            throw std::runtime_error(spdlog::fmt_lib::format("Parsing of input file {} failed", file.string()));
        }

        return { context, owningModule.release() };
    }

    void Compiler::LowerFor(const CompilationTarget& target) const
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

    template <bool isDebug>
    mlir::MLIRContext::Threading constexpr Compiler::GetThreadingMode()
    {
        if constexpr (isDebug)
        {
            return mlir::MLIRContext::Threading::DISABLED;
        } else
        {
            return mlir::MLIRContext::Threading::ENABLED;
        }
    }
}

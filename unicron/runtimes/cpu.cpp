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
// Created by momo- on 9/16/2025.
//

#include "unicron/runtimes/cpu.hpp"
#include "primus/conversions/linalg/transforms/Passes.h"
#include "primus/conversions/linalg/transforms/Rewriters.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Affine/Passes.h>
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include <mlir/Dialect/Linalg/Passes.h>
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Operation.h>
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <spdlog/spdlog.h>


namespace unicron {
    std::unique_ptr<llvm::Module> cpu_runtime_t::lower_to_llvm(mlir::Operation *op, llvm::LLVMContext &ctx) {
        spdlog::trace("Lowering module to LLVM IR (cpu)", op->getName().getStringRef());

        mlir::MLIRContext *context = op->getContext();

        // Register the BufferizableOpInterface implementations
        mlir::DialectRegistry registry;
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        context->appendDialectRegistry(registry);

        // Create a PassManager for the multi-steps lowering
        mlir::PassManager pm(context);
        pm.enableVerifier(true);

        // Step 1: Lower Primus dialect to Linalg
        pm.addPass(mlir::primus::createPrimusLegalizeToLinalgPass());

        // Step 2: Bufferization - convert tensors to memrefs
        pm.addPass(mlir::bufferization::createOneShotBufferizePass({
            .bufferizeFunctionBoundaries = true,
            .functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::InferLayoutMap
        }));

        // Step 6: Prepare memref operations for LLVM conversion
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::memref::createExpandReallocPass());
        pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());

        // Step 3: Lower high-level operations to loops
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopNormalizePass(true));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass());

        // Step 4: Convert affine to standard dialect
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // Step 5: Convert SCF to control flow
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToControlFlowPass());

        // Step 7: Convert to LLVM dialect - ORDER MATTERS!
        pm.addPass(mlir::createConvertIndexToLLVMPass({64}));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass({64}));
        pm.addPass(mlir::createConvertControlFlowToLLVMPass({64}));
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass({true, 64, false}));
        pm.addPass(mlir::createConvertFuncToLLVMPass());

        // Step 8: Final cleanup
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // Run the complete pass pipeline
        if (mlir::failed(pm.run(op))) {
            spdlog::error("Failed to run lowering passes");
            return nullptr;
        }

        // Register translation to LLVM IR
        mlir::registerBuiltinDialectTranslation(*context);
        mlir::registerLLVMDialectTranslation(*context);

        // Convert MLIR LLVM dialect to LLVM IR
        auto llvmModule = mlir::translateModuleToLLVMIR(op, ctx);
        if (!llvmModule) {
            spdlog::error("Failed to translate MLIR to LLVM IR");
            op->dumpPretty();
            return nullptr;
        }

        llvmModule->dump();

        return llvmModule;
    }
}

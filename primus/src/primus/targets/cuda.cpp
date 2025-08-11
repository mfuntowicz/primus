#include "primus/targets/cuda.hpp"

#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/GPU/Pipelines/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>
#include <stablehlo/conversions/linalg/transforms/Passes.h>
#include <stablehlo/transforms/optimization/Passes.h>

namespace primus
{
    void CudaTarget::LoweringPipeline(mlir::PassManager& pm) const
    {
        if constexpr (IS_DEBUG)
        {
            pm.enableIRPrinting();
            pm.enableStatistics();
        }
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        pm.addNestedPass<mlir::func::FuncOp>(mlir::stablehlo::createStablehloTargetIndependentOptimizationPass());
        pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());

        // linalg
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createConvertElementwiseToLinalgPass());
        pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
        pm.addPass(mlir::bufferization::createOneShotBufferizePass({
            .bufferizeFunctionBoundaries = true,
            .functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap,
        }));
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

        // affine
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopNormalizePass( /* promoteSingleIter = */ true));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass(0, 0, true));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertAffineForToGPUPass());

        // gpu
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        pm.addPass(mlir::createGpuKernelOutliningPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createGpuDecomposeMemrefsPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::memref::createNormalizeMemRefsPass());

        auto cuda_target_ops = mlir::gpu::GPUToNVVMPipelineOptions();
        cuda_target_ops.cubinChip = "sm_100";
        // cuda_target_ops.cubinFeatures = "+ptx";
        mlir::gpu::buildLowerToNVVMPassPipeline(pm, cuda_target_ops);

        // pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps({.indexBitwidth = 0, .useBarePtrCallConv = true}));
        // pm.addPass(mlir::createGpuNVVMAttachTarget({.chip = "sm_89", .features = "+ptx", .optLevel = 3, .fastFlag = true}));
    }
}
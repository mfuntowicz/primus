#include "primus/targets/cuda.hpp"

#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/Dialect/Tensor/Utils/Utils.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Transforms/Passes.h>
#include <stablehlo/dialect/StablehloOps.h>
#include <stablehlo/conversions/linalg/transforms/Passes.h>
#include <stablehlo/transforms/optimization/Passes.h>

namespace primus
{
    void CudaTarget::LoweringPipeline(mlir::PassManager& pm) const
    {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        // pm.addPass(mlir::stablehlo::createStablehloAggressiveFolderPass({true, 1, true }));
        pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass({true, true }));
        pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
        pm.addPass(mlir::bufferization::createOneShotBufferizePass({.bufferizeFunctionBoundaries = true }));
        pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    }

}

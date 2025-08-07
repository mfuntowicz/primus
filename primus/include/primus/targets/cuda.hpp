#ifndef PRIMUS_TARGETS_CUDA_HEADER
#define PRIMUS_TARGETS_CUDA_HEADER
#include "primus/targets/base.hpp"

namespace primus {

    class CudaTarget final : public CompilationTarget
    {
        void LoweringPipeline(mlir::PassManager& pm) const override;
    };

}
#endif // PRIMUS_TARGETS_CUDA_HEADER

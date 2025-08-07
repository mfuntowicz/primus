#ifndef PRIMUS_TARGET_BASE_HEADER
#define PRIMUS_TARGET_BASE_HEADER

#include "mlir/Pass/PassManager.h"

namespace primus
{
    class CompilationTarget
    {
    public:
        virtual ~CompilationTarget() = default;
        virtual void LoweringPipeline(mlir::PassManager& pm) const = 0;
    };
}
#endif // PRIMUS_TARGET_BASE_HEADER

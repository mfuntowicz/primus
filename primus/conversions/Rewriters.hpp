//
// Created by mfuntowicz on 8/14/25.
//

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::primus
{
    void populatePrimusLegalizeToStablehloPatterns(MLIRContext* context, RewritePatternSet &patterns);
}
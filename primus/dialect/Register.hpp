#ifndef PRIMUS_DIALECT_REGISTER_H
#define PRIMUS_DIALECT_REGISTER_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
    namespace primus {

        // Add primus dialect to the provided registry.
        void registerAllDialects(DialectRegistry &registry);

    }
}

#endif //PRIMUS_DIALECT_REGISTER_H
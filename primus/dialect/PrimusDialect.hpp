//
// Created by mfuntowicz on 8/13/25.
//

#ifndef PRIMUS_PRIMUSDIALECT_HPP
#define PRIMUS_PRIMUSDIALECT_HPP

#include <mlir/IR/Dialect.h>

namespace mlir::primus
{
    class PrimusDialect final : public Dialect
    {
    public:
        explicit PrimusDialect(MLIRContext *context);
        static constexpr StringRef getDialectNamespace() { return "primus"; }
    };
}


#endif //PRIMUS_PRIMUSDIALECT_HPP
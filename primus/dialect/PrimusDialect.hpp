//
// Created by mfuntowicz on 8/13/25.
//

#ifndef PRIMUS_PRIMUSDIALECT_HPP
#define PRIMUS_PRIMUSDIALECT_HPP

#include <mlir/IR/Dialect.h>

namespace mlir::primus
{
    class PrimusDialect final : Dialect
    {
    public:
        explicit PrimusDialect(MLIRContext *context);
        static constexpr StringRef getDialectNamespace() { return "primus"; }

        void getCanonicalizationPatterns(RewritePatternSet& results) const override;
        void* getRegisteredInterfaceForOp(TypeID interfaceID, OperationName opName) override;
    };
}


#endif //PRIMUS_PRIMUSDIALECT_HPP
//
// Created by mfuntowicz on 8/13/25.
//
#include "PrimusOps.hpp"
#include "PrimusDialect.hpp"


namespace mlir::primus
{
    PrimusDialect::PrimusDialect(MLIRContext* context): Dialect(getDialectNamespace(), context, TypeID::get<PrimusDialect>())
    {
        addOperations<
#define GET_OP_LIST
#include "primus/dialect/Ops.cpp.inc"
        >();
    }

}


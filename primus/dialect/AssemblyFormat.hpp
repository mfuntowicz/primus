//
// Created by mfuntowicz on 8/15/25.
//

#ifndef PRIMUS_ASSEMBLYFORMAT_HPP
#define PRIMUS_ASSEMBLYFORMAT_HPP

#include <cstdint>
#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"



namespace mlir::primus
{
    namespace detail {
        void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                                TypeRange operands, Type result);

        ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                                       ArrayRef<Type*> operands,
                                                       Type& result);
    }  // namespace detail

    template <class... OpTypes>
    void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                        OpTypes... types) {
        static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
        SmallVector<Type> typesVec{types...};
        ArrayRef<Type> typesRef = ArrayRef(typesVec);
        return detail::printSameOperandsAndResultTypeImpl(
            p, op, typesRef.drop_back(1), typesRef.back());
    }

    template <class... OpTypes>
    ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                               OpTypes&... types) {
        static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
        SmallVector<Type*> typesVec{&types...};
        ArrayRef<Type*> typesRef = ArrayRef(typesVec);
        return detail::parseSameOperandsAndResultTypeImpl(
            parser, typesRef.drop_back(1), *typesRef.back());
    }
}

#endif //PRIMUS_ASSEMBLYFORMAT_HPP
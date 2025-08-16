//
// Created by mfuntowicz on 8/15/25.
//

#include "AssemblyFormat.hpp"

namespace mlir::primus
{
    namespace {
        // Utility function, used by printSelectOpType and
        // printSameOperandsAndResultType. Given a FunctionType, assign the types
        // to operands and results, erroring if any mismatch in number of operands
        // or results occurs.
        ParseResult assignFromFunctionType(OpAsmParser& parser, llvm::SMLoc loc,
                                           ArrayRef<Type*> operands, Type& result,
                                           FunctionType& fnType) {
            assert(fnType);
            if (fnType.getInputs().size() != operands.size())
                return parser.emitError(loc)
                       << operands.size() << " operands present, but expected "
                       << fnType.getInputs().size();

            // Set operand types to function input types
            for (auto [operand, input] : llvm::zip(operands, fnType.getInputs()))
                *operand = input;

            // Set result type
            if (fnType.getResults().size() != 1)
                return parser.emitError(loc, "expected single output");
            result = fnType.getResults()[0];

            return success();
        }
    }  // namespace

    namespace detail {
        void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                                TypeRange operands, Type result) {
            // Handle zero operand types `() -> a` prints `a`
            if (operands.empty()) {
                p.printType(result);
                return;
            }

            // Handle all same type `(a,a,...) -> a` prints `a`
            bool allSameType =
                llvm::all_of(operands, [&result](auto t) { return t == result; });
            if (allSameType) {
                p.printType(result);
                return;
            }

            // Fall back to generic
            p.printFunctionalType(op);
        }

        ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                                       ArrayRef<Type*> operands,
                                                       Type& result) {
            llvm::SMLoc loc = parser.getCurrentLocation();
            Type type;
            if (parser.parseType(type)) return failure();

            // Handle if function type, all operand types did not match result type.
            if (auto fnType = dyn_cast<FunctionType>(type))
                return assignFromFunctionType(parser, loc, operands, result, fnType);

            // Handle bare types. ` : type` indicating all input/output types match.
            for (Type* t : operands) *t = type;
            result = type;
            return success();
        }
    } // namespace primus::detail
}
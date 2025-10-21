//
// Created by mfuntowicz on 10/16/25.
//

#ifndef PRIMUS_EXPRESSION_HPP
#define PRIMUS_EXPRESSION_HPP

#include <string_view>
#include <variant>

#include <llvm/Support/FormatProviders.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>

#include "Types.hpp"

namespace tlang
{
    /**
     *
     */
    struct VariableDecl
    {
        std::string_view name;
        TensorOrScalarTy type;
        std::string_view initializer;
    };


    struct DeclStmt
    {
    };


    // struct FunctionArgDecl
    // {
    //     std::string_view name;
    // };
    //
    // using FuncArgumentsDecl = llvm::SmallVector<FunctionArgDecl, 2>;
    //
    // /**
    //  *
    //  */
    // struct FunctionDecl
    // {
    //     std::string_view name;
    //     FuncArgumentsDecl args;
    // };
    //
    // using ModelOrFuncDecl = std::variant<FunctionDecl>;
} // tlang

namespace llvm
{
    // template <>
    // struct format_provider<tlang::FunctionArgDecl>
    // {
    //     static void format(const tlang::FunctionArgDecl& T, raw_ostream& OS, const StringRef Style)
    //     {
    //         OS << formatv("FunctionArg{name={0}}", T.name);
    //     }
    // };
    //
    // template <>
    // struct format_provider<tlang::FunctionDecl>
    // {
    //     static void format(const tlang::FunctionDecl& T, raw_ostream& OS, const StringRef Style)
    //     {
    //         OS << formatv("FunctionDecl{name={0}", T.name);
    //         for (const auto& arg : T.args)
    //         {
    //             OS << formatv("\n\tFunctionArg{name={0}}, ", arg.name);
    //         }
    //
    //         OS << "}\n";
    //     }
    // };
}

#endif //PRIMUS_EXPRESSION_HPP

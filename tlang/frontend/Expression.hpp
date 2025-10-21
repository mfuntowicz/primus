//
// Created by mfuntowicz on 10/16/25.
//

#ifndef PRIMUS_EXPRESSION_HPP
#define PRIMUS_EXPRESSION_HPP

#include <string_view>
#include <variant>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>

#include "Types.hpp"

namespace tlang
{
    /**
     * Represent a variable declaration with type, which might be later inferred, and optional initializer.
     *
     * This is transcribed in tlang as:
     * ```
     * a: int32 = 1
     * ```
     *
     * Or without type information:
     * ```
     * a = 1
     * ```
     *
     * In the later case, `type` will be defined as `InferTy` for further refinement at a later stage.
     */
    struct VariableDecl
    {
        std::string_view name;
        InferrableTensorOrScalarTy type;
        std::string_view initializer;
    };

    template <typename T>
    concept VisitableDecl = requires(const T& t)
    {
        { t.Visit() } -> std::same_as<void>;
    };

    template <typename T>
    concept CompositeDecl = VisitableDecl<T> && requires(T t, VariableDecl&& decl)
    {
        /**
         * Add a new declaration to the composite `T`
         */
        { t.AddDecl(std::move(decl)) } -> std::same_as<void>;
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


#include <llvm/Support/FormatProviders.h>

namespace llvm
{
    template <>
    struct format_provider<tlang::InferTy>
    {
        static void format(const tlang::InferTy& T, raw_ostream& OS, StringRef Style)
        {
            OS << "[InferType]";
        }
    };

    template <>
    struct format_provider<tlang::IntegerTy>
    {
        static void format(const tlang::IntegerTy& T, raw_ostream& OS, StringRef Style)
        {
            OS << formatv("[UInt{0}]", T.width);
        }
    };

    template <>
    struct format_provider<tlang::SignedIntegerTy>
    {
        static void format(const tlang::SignedIntegerTy& T, raw_ostream& OS, StringRef Style)
        {
            OS << formatv("[Int{0}]", T.width);
        }
    };

    template <>
    struct format_provider<tlang::FloatTy>
    {
        static void format(const tlang::FloatTy& T, raw_ostream& OS, StringRef Style)
        {
            OS << formatv("[Float{0}]", T.width);
        }
    };

    template <>
    struct format_provider<tlang::ScalarTy>
    {
        static void format(const tlang::ScalarTy& T, raw_ostream& OS, StringRef Style)
        {
            std::visit([=, &OS](auto ty) { OS << formatv("{0}", ty); }, T);
        }
    };

    template <>
    struct format_provider<tlang::TensorTy>
    {
        static void format(const tlang::TensorTy& T, raw_ostream& OS, StringRef Style)
        {
            OS << formatv("[Tensor{shape={0}]", T.shapes.begin(), T.shapes.end());
        }
    };

    template <>
    struct format_provider<tlang::VariableDecl>
    {
        static void format(const tlang::VariableDecl& V, raw_ostream& OS, StringRef Style)
        {
            OS << formatv("VariableDecl{name = {0}, ", V.name);
            std::visit([=, &OS](auto ty) { OS << formatv("type = {0}, ", ty); }, V.type);
            OS << formatv("initializer = {0}}", V.initializer);
        }
    };

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

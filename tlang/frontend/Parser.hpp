//
// Created by mfuntowicz on 10/16/25.
//

#ifndef PRIMUS_PARSER_HPP
#define PRIMUS_PARSER_HPP

#include <expected>
#include <list>
#include <optional>
#include <string_view>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/FormatProviders.h>

#include "Expression.hpp"
#include "Lexer.hpp"
#include "Token.hpp"

namespace tlang
{
    static std::string FUNCTION_DECLARATION_CONTEXT = "Error while parsing function declaration";
    static std::string VARIABLE_DECLARATION_CONTEXT = "Error while parsing variable declaration";
    static std::string TYPE_PARSING_CONTEXT = "Error while parsing type declaration";

    static std::array<std::string, 3> BUILTIN_TYPE_NAMES = {"float", "int", "uint"};

    namespace errors
    {
        /**
         *
         */
        enum DiagnosticKind
        {
            kUnexpectedToken,
            kInvalidType,
        };

        /**
         *
         */
        struct Diagnostic
        {
            bool isError;
            DiagnosticKind kind;
            llvm::SmallString<32> what;

            static Diagnostic
            UnexpectedToken(const TokenKind expected, const Token& actual, const std::string_view context)
            {
                return {
                    true,
                    kUnexpectedToken,
                    llvm::formatv("{0}: Expected {1} but got {2} (line: {3})", context, expected, actual.kind,
                                  actual.line)
                };
            }

            static Diagnostic
            UnexpectedToken(const llvm::SmallVector<TokenKind>& expected, const Token& actual,
                            const std::string_view context)
            {
                return {
                    true,
                    kUnexpectedToken,
                    llvm::formatv("{0}: Expected {1} but got {2} (line: {3})", context, expected.begin(),
                                  expected.end(), actual.kind,
                                  actual.line)
                };
            }

            static Diagnostic
            InvalidType(const Token& token, const std::string_view context)
            {
                return {
                    true,
                    kInvalidType,
                    llvm::formatv("{0}: Invalid type {1} (line: {2})", context, token.value, token.line)
                };
            }
        };

        using Diagnostics = llvm::SmallVector<Diagnostic>;
    }


    /**
     *
     */
    class TranslationUnit
    {
        std::list<VariableDecl> decls;

    public:
        void AddDecl(VariableDecl&&);
        void Visit() const;

        const std::list<VariableDecl>& Declarations() const { return decls; }
    };

    static_assert(CompositeDecl<TranslationUnit>);

    /**
     * Parser is responsible for aggregating lexed tokens into more semantically meaningful representations.
     *
     */
    class Parser
    {
        Lexer lexer;
        /**
                *
                * @param token
                * @return
                */
        static constexpr bool IsBuiltinTypeLiteral(const Token& token)
        {
            return std::ranges::any_of(BUILTIN_TYPE_NAMES, [&](const std::string& type)
            {
                return token.value.value().starts_with(type);
            });
        }

        /**
         *
         * @param ty
         * @param diagnostics
         * @return
         */
        static std::expected<InferableScalarTy, errors::Diagnostics>
        ParseType(const Token& ty, errors::Diagnostics& diagnostics);

        std::expected<TensorTy, errors::Diagnostics>
        ParserTensorDefinition(const ScalarTy dtype, errors::Diagnostics& diagnostics);

        /**
         * @param name
         * @param diagnostics
         * @return
         */
        std::expected<VariableDecl, errors::Diagnostics>
        ParseVariableDecl(Token name, errors::Diagnostics& diagnostics);

        /**
         *
         * @param name
         * @param diagnostics
         * @return
         */
        std::expected<VariableDecl, errors::Diagnostics>
        ParseVariableDeclInferType(Token name, errors::Diagnostics& diagnostics);

    public:
        Parser(std::string_view source, const std::optional<std::string>& file);
        Parser(const llvm::MemoryBufferRef& buffer, const std::optional<std::string>& file);

        /**
         *
         * @return
         */
        std::expected<TranslationUnit, llvm::SmallVector<errors::Diagnostic>> Parse();
    };
} // tlang

namespace llvm
{
    template <>
    struct format_provider<tlang::TranslationUnit>
    {
        static void format(const tlang::TranslationUnit& T, raw_ostream& OS, StringRef Style)
        {
            OS << "{\n";
            for (const auto& decl : T.Declarations())
            {
                OS << formatv("\t{0}\n", decl);
            }
            OS << "}";
        }
    };
}

#endif //PRIMUS_PARSER_HPP

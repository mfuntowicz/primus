//
// Created by mfuntowicz on 10/16/25.
//

#ifndef PRIMUS_PARSER_HPP
#define PRIMUS_PARSER_HPP

#include <expected>
#include <optional>
#include <string_view>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>

#include "Expression.hpp"
#include "Lexer.hpp"
#include "Token.hpp"

namespace tlang
{
    static std::string FUNCTION_DECLARATION_CONTEXT = "Error while parsing function declaration";
    static std::string VARIABLE_DECLARATION_CONTEXT = "Error while parsing variable declaration";

    namespace errors
    {
        /**
         *
         */
        enum DiagnosticKind
        {
            kUnexpectedToken
        };

        /**
         *
         */
        struct Diagnostic
        {
            bool isError;
            DiagnosticKind kind;
            llvm::SmallString<32> what;

            static Diagnostic UnexpectedToken(const TokenKind expected, const Token& actual,
                                              const std::string_view context)
            {
                return {
                    true,
                    kUnexpectedToken,
                    llvm::formatv("{0}: Expected {1} but got {2} (line: {3})", context, expected, actual.kind,
                                  actual.line)
                };
            }
        };

        using Diagnostics = llvm::SmallVector<Diagnostic>;
    }


    class Context
    {
    };


    /**
     *
     */
    class Parser
    {
        Lexer lexer;

        /**
         * @param diagnostics
         * @return
         */
        // std::expected<FunctionDecl, llvm::SmallVector<errors::Diagnostic>>
        // ParseFunctionDeclaration(llvm::SmallVector<errors::Diagnostic>& diagnostics);

        /**
         *
         * @param diagnostics
         * @return
         */
        // std::optional<FuncArgumentsDecl> ParseArgumentList(errors::Diagnostics& diagnostics);

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
        std::expected<VariableDecl, llvm::SmallVector<errors::Diagnostic>> Parse();
    };
} // tlang

#endif //PRIMUS_PARSER_HPP

//
// Created by mfuntowicz on 10/16/25.
//

#include "Parser.hpp"

#include <optional>
#include <llvm/Support/FormatVariadic.h>

#define EXTRACT(varname, x) \
    auto varname = (x); \
    if (!varname) return std::unexpected(diagnostics);

#define CONSUME(x) \
    if (!(x)) return std::unexpected(diagnostics);

namespace tlang
{
    template <typename... Args>
    std::optional<Token> Expect(Token token, TokenKind kind, errors::Diagnostics& diagnostics, Args&&... args)
    {
        auto result = MatchTokenOrDiagnostic(token, kind, diagnostics, std::forward<Args>(args)...);
        if (!result) return std::nullopt;
        return result;
    }

    Parser::Parser(const std::string_view source, const std::optional<std::string>& file) : lexer(source, file)
    {
    }

    Parser::Parser(const llvm::MemoryBufferRef& buffer, const std::optional<std::string>& file) : lexer(buffer, file)
    {
    }

    bool IsTokenA(const Token& token, const TokenKind expected)
    {
        return token.kind == expected;
    }

    std::optional<Token> MatchTokenOrDiagnostic(
        const Token& token,
        const TokenKind expected,
        errors::Diagnostics& diagnostics,
        const std::string_view context
    )
    {
        if (IsTokenA(token, expected)) return token;

        diagnostics.push_back(errors::Diagnostic::UnexpectedToken(expected, token, context));
        return std::nullopt;
    }

    // std::optional<FuncArgumentsDecl> Parser::ParseArgumentList(errors::Diagnostics& diagnostics)
    // {
    //     FuncArgumentsDecl args;
    //     auto token = lexer.Lex();
    //
    //     while (!IsTokenA(token, kParenthesisClose))
    //     {
    //         auto argName = Expect(lexer.Lex(), kLiteral, diagnostics, FUNCTION_DECLARATION_CONTEXT);
    //
    //         if (!MatchTokenOrDiagnostic(lexer.Lex(), kColon, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //             return std::nullopt;
    //
    //         const auto argType = MatchTokenOrDiagnostic(
    //             lexer.Lex(), kLiteral, diagnostics, FUNCTION_DECLARATION_CONTEXT);
    //         if (!argType) return std::nullopt;
    //
    //         token = lexer.Lex();
    //
    //         if (IsTokenA(token, kComma)) token = lexer.Lex();
    //         else if (!IsTokenA(token, kParenthesisClose))
    //         {
    //             diagnostics.push_back(
    //                 errors::Diagnostic::UnexpectedToken(kParenthesisClose, token, FUNCTION_DECLARATION_CONTEXT)
    //             );
    //             return std::nullopt;
    //         }
    //     }
    //
    //     return args;
    // }
    //
    // std::expected<FunctionDecl, errors::Diagnostics> Parser::ParseFunctionDeclaration(errors::Diagnostics& diagnostics)
    // {
    //     BIND(name, MatchTokenOrDiagnostic(lexer.Lex(), kLiteral, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //     CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kParenthesisOpen, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //     BIND(args, ParseArgumentList(diagnostics))
    //     CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kMinus, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //     CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kArrowRight, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //     BIND(returnType, MatchTokenOrDiagnostic(lexer.Lex(), kLiteral, diagnostics, FUNCTION_DECLARATION_CONTEXT))
    //
    //     return FunctionDecl{*name.value().value, *args};
    // }

    std::expected<VariableDecl, errors::Diagnostics>
    Parser::ParseVariableDecl(Token name, errors::Diagnostics& diagnostics)
    {
        EXTRACT(type, MatchTokenOrDiagnostic(lexer.Lex(), kLiteral, diagnostics, VARIABLE_DECLARATION_CONTEXT));
        CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kAssign, diagnostics, VARIABLE_DECLARATION_CONTEXT));
        EXTRACT(value, MatchTokenOrDiagnostic(lexer.Lex(), kInteger, diagnostics, VARIABLE_DECLARATION_CONTEXT));

        return VariableDecl{
            std::string_view(name.value.value()),
            IntegerTy{32},
            std::string_view(value.value().value.value())
        };
    }

    std::expected<VariableDecl, errors::Diagnostics>
    Parser::ParseVariableDeclInferType(Token name, errors::Diagnostics& diagnostics)
    {
        EXTRACT(value, MatchTokenOrDiagnostic(lexer.Lex(), kInteger, diagnostics, VARIABLE_DECLARATION_CONTEXT));

        return VariableDecl{
            std::string_view(name.value.value()),
            InferTy{},
            std::string_view(value.value().value.value())
        };
    }

    std::expected<VariableDecl, errors::Diagnostics> Parser::Parse()
    {
        errors::Diagnostics diagnostics;
        for (auto token = Token::Begin(0); token.kind != kEndOfStream; token = lexer.Lex())
        {
            switch (token.kind)
            {
            case kBeginOfStream: continue;
            case kLiteral:
                if (const auto next = lexer.Lex(); next.kind == kColon)
                {
                    return ParseVariableDecl(token, diagnostics);
                }
                else if (next.kind == kAssign)
                {
                    return ParseVariableDeclInferType(token, diagnostics);
                }
            }

            llvm_unreachable("");
        }
    }
} // tlang

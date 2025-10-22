//
// Created by mfuntowicz on 10/16/25.
//

#include "Parser.hpp"

#include <charconv>
#include <optional>
#include <llvm/Support/FormatVariadic.h>

#define EXTRACT(varname, x) \
    auto varname = (x); \
    if (!varname) return std::unexpected(diagnostics);

#define CONSUME(x) \
    if (!(x)) return std::unexpected(diagnostics);


using namespace tlang::errors;

namespace tlang
{
    template <typename... Args>
    std::optional<Token> Expect(Token token, TokenKind kind, Diagnostics& diagnostics, Args&&... args)
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
        Diagnostics& diagnostics,
        const std::string_view context
    )
    {
        if (IsTokenA(token, expected)) return token;

        diagnostics.push_back(Diagnostic::UnexpectedToken(expected, token, context));
        return std::nullopt;
    }

    std::optional<Token> MatchTokenOrDiagnostic(
        const Token& token,
        const llvm::SmallVector<TokenKind>& kinds,
        Diagnostics& diagnostics,
        const std::string_view context
    )
    {
        for (const auto& expected : kinds)
            if (IsTokenA(token, expected)) return token;

        diagnostics.push_back(Diagnostic::UnexpectedToken(kinds, token, context));
        return std::nullopt;
    }

    void TranslationUnit::AddDecl(VariableDecl&& vardecl)
    {
        decls.push_back(std::move(vardecl));
    }

    void TranslationUnit::Visit() const
    {
    }


    // std::optional<FuncArgumentsDecl> Parser::ParseArgumentList(Diagnostics& diagnostics)
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
    //                 Diagnostic::UnexpectedToken(kParenthesisClose, token, FUNCTION_DECLARATION_CONTEXT)
    //             );
    //             return std::nullopt;
    //         }
    //     }
    //
    //     return args;
    // }
    //
    // std::expected<FunctionDecl, Diagnostics> Parser::ParseFunctionDeclaration(Diagnostics& diagnostics)
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

    std::expected<InferableScalarTy, Diagnostics>
    Parser::ParseType(const Token& ty, Diagnostics& diagnostics)
    {
        if (IsTokenA(ty, kLiteral))
        {
            const auto type = ty.value.value();
            if (type.starts_with("uint"))
            {
                auto dtype = IntegerTy();
                auto [_, ec] = std::from_chars(type.cbegin() + 4, type.cend(), dtype.width, 10);
                if (ec != std::errc())
                    diagnostics.push_back(Diagnostic::InvalidType(ty, TYPE_PARSING_CONTEXT));

                return InferableScalarTy(dtype);
            }

            if (type.starts_with("int"))
            {
                auto dtype = SignedIntegerTy();
                auto [_, ec] = std::from_chars(type.cbegin() + 3, type.cend(), dtype.width, 10);
                if (ec != std::errc())
                    diagnostics.push_back(Diagnostic::InvalidType(ty, TYPE_PARSING_CONTEXT));

                return InferableScalarTy(dtype);
            }

            if (type.starts_with("float"))
            {
                auto dtype = FloatTy();
                auto [_, ec] = std::from_chars(type.cbegin() + 5, type.cend(), dtype.width, 10);
                if (ec != std::errc())
                    diagnostics.push_back(Diagnostic::InvalidType(ty, TYPE_PARSING_CONTEXT));

                return InferableScalarTy(dtype);
            }
        }

        return std::unexpected(diagnostics);
    }

    std::expected<TensorTy, Diagnostics>
    Parser::ParserTensorDefinition(const ScalarTy dtype, Diagnostics& diagnostics)
    {
        CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kSquareOpen, diagnostics, "Tensor declaration"));
        CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kSquareClose, diagnostics, "Tensor declaration"));
        return TensorTy{{}, dtype};
    }


    std::expected<VariableDecl, Diagnostics>
    Parser::ParseVariableDecl(Token name, Diagnostics& diagnostics)
    {
        EXTRACT(literal, MatchTokenOrDiagnostic(lexer.Lex(), kLiteral, diagnostics, VARIABLE_DECLARATION_CONTEXT));
        if (const auto identifier = literal.value(); IsBuiltinTypeLiteral(identifier))
        {
            if (const auto dtype = ParseType(identifier, diagnostics); dtype.has_value())
            {
                const auto scalarTy = dtype.value();
                if (const auto next = lexer.Peek(); next.kind == kSquareOpen)
                {
                    if (scalarTy.NeedsInference())
                        diagnostics.push_back(Diagnostic::InvalidType(literal.value(), VARIABLE_DECLARATION_CONTEXT));

                    auto tensorTy = ParserTensorDefinition(scalarTy.Type(), diagnostics);
                    return VariableDecl{name.value.value(), tensorTy.value(), std::nullopt};
                }
                else
                {
                    CONSUME(MatchTokenOrDiagnostic(lexer.Lex(), kAssign, diagnostics, VARIABLE_DECLARATION_CONTEXT));
                    EXTRACT(value,
                            MatchTokenOrDiagnostic(lexer.Lex(), {kInteger, kFloat}, diagnostics,
                                VARIABLE_DECLARATION_CONTEXT));

                    return VariableDecl{
                        std::string_view(name.value.value()),
                        InferableScalarTy{scalarTy},
                        std::string_view(value.value().value.value())
                    };
                }
            }
        }

        return std::unexpected(diagnostics);
    }

    std::expected<VariableDecl, Diagnostics>
    Parser::ParseVariableDeclInferType(Token name, Diagnostics& diagnostics)
    {
        EXTRACT(value, MatchTokenOrDiagnostic(lexer.Lex(), kInteger, diagnostics, VARIABLE_DECLARATION_CONTEXT));

        return VariableDecl{
            std::string_view(name.value.value()),
            InferableScalarTy(),
            std::string_view(value.value().value.value())
        };
    }

    std::expected<TranslationUnit, Diagnostics> Parser::Parse()
    {
        TranslationUnit unit;
        Diagnostics diagnostics;
        for (auto token = Token::Begin(0); token.kind != kEndOfStream; token = lexer.Lex())
        {
            switch (token.kind)
            {
            case kBeginOfStream: continue;
            case kLiteral:
                if (const auto next = lexer.Lex(); next.kind == kColon)
                {
                    if (auto decl = ParseVariableDecl(token, diagnostics); decl.has_value())
                        unit.AddDecl(std::move(*decl));
                }
                else if (next.kind == kAssign)
                {
                    if (auto decl = ParseVariableDeclInferType(token, diagnostics); decl.has_value())
                        unit.AddDecl(std::move(*decl));
                }
            }
        }

        return unit;
    }
} // tlang

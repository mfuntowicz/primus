//
// Created by mfuntowicz on 10/15/25.
//

#ifndef PRIMUS_TOKEN_HPP
#define PRIMUS_TOKEN_HPP

#include <optional>
#include <string_view>

#include <llvm/Support/FormatProviders.h>

namespace tlang
{
    enum TokenKind
    {
        kBeginOfStream,
        kEndOfStream,
        kLiteral,
        kInteger,
        kFloat,
        kAssign,
        kAdd,
        kMinus,
        kMultiply,
        kDivide,
        kSemiColon,
        kFunctionDecl,
        kModelDecl,
        kPass,
    };


    struct Token
    {
        TokenKind kind;
        uint32_t line;
        std::optional<std::string_view> value = std::nullopt;

        /**
         * Create a new Token initialized as the beginning of the stream with specified-line (default to 0)
         * @param line The line at which the token starts. Defaults to 0.
         * @return
         */
        Token static Begin(const uint32_t line = 0)
        {
            return {kBeginOfStream, line};
        }

        Token static End(const uint32_t line)
        {
            return {kEndOfStream, line};
        }

        Token static Assign(const uint32_t line)
        {
            return {kAssign, line};
        }

        Token static Add(const uint32_t line)
        {
            return {kAdd, line};
        }

        Token static Minus(const uint32_t line)
        {
            return {kMinus, line};
        }

        Token static Multiply(const uint32_t line)
        {
            return {kMultiply, line};
        }

        Token static Divide(const uint32_t line)
        {
            return {kDivide, line};
        }

        Token static Semicolon(const uint32_t line)
        {
            return {kSemiColon, line};
        }

        Token static Def(const uint32_t line)
        {
            return {kFunctionDecl, line};
        }

        Token static Model(const uint32_t line)
        {
            return {kModelDecl, line};
        }

        Token static Literal(const uint32_t line, std::string_view value)
        {
            return {kLiteral, line, value};
        }

        Token static Integer(const uint32_t line, std::string_view value)
        {
            return {kInteger, line, value};
        }

        Token static Float(const uint32_t line, std::string_view value)
        {
            return {kFloat, line, value};
        }
    };
}

namespace llvm
{
    template <>
    struct format_provider<tlang::TokenKind>
    {
        static void format(const tlang::TokenKind& K, raw_ostream& OS, StringRef)
        {
            switch (K)
            {
            case tlang::kBeginOfStream: OS << "BeginOfStream";
                break;
            case tlang::kEndOfStream: OS << "EndOfStream";
                break;
            case tlang::kLiteral: OS << "Literal";
                break;
            case tlang::kInteger: OS << "Integer";
                break;
            case tlang::kFloat: OS << "Float";
                break;
            case tlang::kAssign: OS << "Assign";
                break;
            case tlang::kAdd: OS << "Add";
                break;
            case tlang::kMinus: OS << "Minus";
                break;
            case tlang::kMultiply: OS << "Multiply";
                break;
            case tlang::kDivide: OS << "Divide";
                break;
            case tlang::kSemiColon: OS << "SemiColon";
                break;
            case tlang::kFunctionDecl: OS << "FunctionDecl";
                break;
            case tlang::kModelDecl: OS << "ModelDecl";
                break;
            case tlang::kPass: OS << "Pass";
                break;
            }
        }
    };

    template <>
    struct format_provider<tlang::Token>
    {
        static void format(const tlang::Token& T, raw_ostream& OS, const StringRef Style)
        {
            OS << "Token{kind=";
            format_provider<tlang::TokenKind>::format(T.kind, OS, Style);
            OS << ", line=" << T.line;
            if (T.value.has_value())
            {
                OS << ", value='" << *T.value << "'";
            }
            OS << "}";
        }
    };
}


#endif //PRIMUS_TOKEN_HPP

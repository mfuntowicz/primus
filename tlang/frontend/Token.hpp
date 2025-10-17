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
        kArrowRight,
        kComma,
        kColon,
        kParenthesisOpen,
        kParenthesisClose,
        kSquareOpen,
        kSquareClose,
        kFunctionDecl,
        kModelDecl,
        kPass,
    };

    // Macro for tokens without any payload (no value field)
#define EMPTY_TOKEN(Name, Kind) \
    Token static Name(const uint32_t line) \
    { \
        return {Kind, line}; \
    }

    // Macro for tokens with payload (with value field)
#define PAYLOAD_TOKEN(Name, Kind) \
    Token static Name(const uint32_t line, std::string_view value) \
    { \
        return {Kind, line, value}; \
    }

    struct Token
    {
        TokenKind kind;
        uint32_t line;
        std::optional<std::string_view> value = std::nullopt;

        EMPTY_TOKEN(Begin, kBeginOfStream)
        EMPTY_TOKEN(End, kEndOfStream)
        EMPTY_TOKEN(Assign, kAssign)
        EMPTY_TOKEN(Add, kAdd)
        EMPTY_TOKEN(Minus, kMinus)
        EMPTY_TOKEN(Multiply, kMultiply)
        EMPTY_TOKEN(Divide, kDivide)
        EMPTY_TOKEN(ArrowRight, kArrowRight)
        EMPTY_TOKEN(Comma, kComma)
        EMPTY_TOKEN(Colon, kColon)
        EMPTY_TOKEN(ParenthesisOpen, kParenthesisOpen)
        EMPTY_TOKEN(ParenthesisClose, kParenthesisClose)
        EMPTY_TOKEN(SquareOpen, kSquareOpen)
        EMPTY_TOKEN(SquareClose, kSquareClose)
        EMPTY_TOKEN(Def, kFunctionDecl)
        EMPTY_TOKEN(Model, kModelDecl)

        PAYLOAD_TOKEN(Literal, kLiteral)
        PAYLOAD_TOKEN(Integer, kInteger)
        PAYLOAD_TOKEN(Float, kFloat)
    };

#undef EMPTY_TOKEN
#undef PAYLOAD_TOKEN
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
                using enum tlang::TokenKind;
            case kBeginOfStream: OS << "BeginOfStream";
                break;
            case kEndOfStream: OS << "EndOfStream";
                break;
            case kLiteral: OS << "Literal";
                break;
            case kInteger: OS << "Integer";
                break;
            case kFloat: OS << "Float";
                break;
            case kAssign: OS << "Assign";
                break;
            case kAdd: OS << "Add";
                break;
            case kMinus: OS << "Minus";
                break;
            case kMultiply: OS << "Multiply";
                break;
            case kDivide: OS << "Divide";
                break;
            case kArrowRight: OS << "ArrowRight";
                break;
            case kComma: OS << "Comma";
                break;
            case kColon: OS << "Colon";
                break;
            case kParenthesisOpen: OS << "ParenthesisOpen";
                break;
            case kParenthesisClose: OS << "ParenthesisClose";
                break;
            case kSquareOpen: OS << "SquareOpen";
                break;
            case kSquareClose: OS << "SquareClose";
                break;
            case kFunctionDecl: OS << "FunctionDecl";
                break;
            case kModelDecl: OS << "ModelDecl";
                break;
            case kPass: OS << "Pass";
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

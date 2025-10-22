//
// Created by Morgan Funtowicz on 10/15/2025.
//

#include "Lexer.hpp"

#include <charconv>

#include "Token.hpp"

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormatVariadic.h>

namespace tlang
{
    inline bool IsInteger(const std::string_view sv)
    {
        if (sv.empty()) return false;

        // Optional leading sign is allowed by from_chars for integers
        int64_t value; // type doesn't matter for validation; choose a wide enough type
        auto [ptr, ec] = std::from_chars(sv.cbegin(), sv.cend(), value, 10);
        return ec == std::errc{} && ptr == sv.cend();
    }

    inline bool IsFloat(const std::string_view sv)
    {
        if (sv.empty()) return false;

        double value;
        auto [ptr, ec] = std::from_chars(sv.cbegin(), sv.cend(), value, std::chars_format::general);
        return ec == std::errc{} && ptr == sv.cend();
    }

    inline bool IsAcceptedLiteralChar(const uint8_t c)
    {
        return c == '_' || c == '.' || std::isalnum(c);
    }

    void Lexer::MoveOnIfNewLine()
    {
        if (*current == '\n' || *current == '\r')
        {
            ++line;
            ++current;
        }
    }

    std::pair<const char*, const char*> Lexer::Consume()
    {
        // Safety check - ensure offset is within bounds
        if (current >= end)
        {
            return {current, current};
        }

        // Read alphanumeric characters while staying within bounds
        const char* from = current;
        const char* to = current;
        while (to < end && IsAcceptedLiteralChar(static_cast<unsigned char>(*to)))
        {
            ++to;
        }

        current = to - 1;
        return {from, to};
    }

    Lexer::Lexer(const std::string_view source, const std::optional<std::string>& file)
        : source(source), line(0), current(source.cbegin()), end(source.cend())
    {
    }

    Lexer::Lexer(const llvm::MemoryBufferRef& source, const std::optional<std::string>& file)
        : Lexer(std::string_view(source.getBuffer()), file)
    {
    }

    Token Lexer::Lex()
    {
        // Skip any leading space(s) and count newlines
        while (current < end && isspace(*current))
        {
            if (*current == '\n' || *current == '\r')
                ++line;
            ++current;
        }

        // Check we are not at the end
        if (current >= end) return Token::End(line);

        auto token = Token::Begin(line);
        switch (*current)
        {
        // case '\n':
        // case '\r':
        //     ++line;
        //     ++current;
        //     [[fallthrough]];

        // Skip comments
        case '#':
            do { ++current; }
            while (*current != '\n');
            break;

        case '>':
            token = Token::ArrowRight(line);
            break;

        case ',':
            token = Token::Comma(line);
            break;

        case ':':
            token = Token::Colon(line);
            break;

        case '(':
            token = Token::ParenthesisOpen(line);
            break;

        case ')':
            token = Token::ParenthesisClose(line);
            break;

        case '[':
            token = Token::SquareOpen(line);
            break;

        case ']':
            token = Token::SquareClose(line);
            break;

        case '=':
            token = Token::Assign(line);
            break;

        case '+':
            token = Token::Add(line);
            break;

        case '-':
            token = Token::Minus(line);
            break;

        case '*':
            token = Token::Multiply(line);
            break;

        case '/':
            token = Token::Divide(line);
            break;

        default:
            const auto [from, to] = Consume();
            const auto buffer = std::string_view(from, to);
            if (IsInteger(buffer))
                token = Token::Integer(line, buffer);
            else if (IsFloat(buffer))
                token = Token::Float(line, buffer);
            else if (buffer == "model")
                token = Token::Model(line);
            else if (buffer == "def")
                token = Token::Def(line);
            else
                token = Token::Literal(line, buffer);
        }

        ++current;
        return token;
    }

    Token Lexer::Peek()
    {
        // Store current position
        const auto* originalCurrent = current;
        const auto originalLine = line;

        // Get the next token
        auto token = Lex();

        // Restore position
        current = originalCurrent;
        line = originalLine;

        return token;
    }
} // htl

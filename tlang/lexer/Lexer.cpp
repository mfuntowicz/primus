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
    inline bool is_integer(const std::string_view sv)
    {
        if (sv.empty()) return false;

        // Optional leading sign is allowed by from_chars for integers
        int64_t value; // type doesn't matter for validation; choose a wide enough type
        auto [ptr, ec] = std::from_chars(sv.cbegin(), sv.cend(), value, 10);
        return ec == std::errc{} && ptr == sv.cend();
    }

    inline bool is_float(const std::string_view sv)
    {
        if (sv.empty()) return false;

        double value;
        auto [ptr, ec] = std::from_chars(sv.cbegin(), sv.cend(), value, std::chars_format::general);
        return ec == std::errc{} && ptr == sv.cend();
    }

    std::pair<const char*, const char*> Lexer::Consume()
    {
        // Safety check - ensure offset is within bounds
        if (current >= end)
        {
            return {current, current};
        }

        const char* start = current;

        // Read alphanumeric characters while staying within bounds
        while (current < end && std::isalnum(static_cast<unsigned char>(*current)))
        {
            ++current;
        }

        return {start, current};
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
        if (current >= end) return Token::End(line);

        // New line, increment current line number
        // TODO: Increment and continue processing

        auto token = Token::Begin();
        switch (*current)
        {
        // Skip comments
        case '#':
            do { ++current; }
            while (*current != '\n');
            break;

        case ':':
            token = Token::Semicolon(line);
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
            // Skip spaces
            while (std::isspace(*current) && current != source.end()) ++current;

            const auto [from, to] = Consume();
            const auto buffer = std::string_view(from, to);
            if (is_integer(buffer))
                token = Token::Integer(line, buffer);
            else if (is_float(buffer))
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
} // htl

//
// Created by mfuntowicz on 10/16/25.
//

#include "Parser.hpp"

#include <llvm/Support/FormatVariadic.h>

#include "Token.hpp"

namespace tlang
{
    Parser::Parser(const std::string_view source, const std::optional<std::string>& file) : lexer(source, file)
    {
    }

    Parser::Parser(const llvm::MemoryBufferRef& buffer, const std::optional<std::string>& file) : lexer(buffer, file)
    {
    }

    void Parser::Parse()
    {
        for (auto token = lexer.Lex(); token.kind != kEndOfStream; token = lexer.Lex())
        {
            llvm::outs() << llvm::formatv("New token {0}\n", token);
        }
    }
} // tlang

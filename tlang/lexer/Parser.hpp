//
// Created by mfuntowicz on 10/16/25.
//

#ifndef PRIMUS_PARSER_HPP
#define PRIMUS_PARSER_HPP

#include <optional>
#include <string_view>

#include "Lexer.hpp"

namespace tlang
{
    struct ParserError
    {
    };


    class Parser
    {
        Lexer lexer;

    public:
        Parser(std::string_view source, const std::optional<std::string>& file);
        Parser(const llvm::MemoryBufferRef& buffer, const std::optional<std::string>& file);

        void Parse();
    };
} // tlang

#endif //PRIMUS_PARSER_HPP

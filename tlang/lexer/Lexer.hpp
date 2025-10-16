//
// Created by Morgan Funtowicz on 10/15/2025.
//

#ifndef PRIMUS_LEXER_HPP
#define PRIMUS_LEXER_HPP

#include <optional>
#include <string_view>

#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/LogicalResult.h>

namespace tlang
{
    class Lexer
    {
        std::string file;
        std::string_view source;

    public:
        /**
         * Construct a Lexer from the provided buffer without taking ownership of the underlying string.
         * The content is assumed to overlive the lifespan of the Lexer
         * @param source
         * @param file
         */
        Lexer(std::string_view source, const std::optional<std::string>& file);

        /**
         * Construct a Lexer from the provided buffer without taking ownership of the underlying string.
         * The content is assumed to overlive the lifespan of the Lexer
         * @param source
         * @param file
         */
        Lexer(const llvm::MemoryBufferRef& source, const std::optional<std::string>& file);

        Lexer(const Lexer&) = delete;
        Lexer& operator=(const Lexer&) = delete;

        std::pair<const char*, const char*> consume(const char*& offset) const;
        llvm::LogicalResult Lex() const;
    };
} // htl

#endif //PRIMUS_LEXER_HPP

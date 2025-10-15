//
// Created by momo- on 10/15/2025.
//

#ifndef PRIMUS_LEXER_HPP
#define PRIMUS_LEXER_HPP

#include <string_view>

namespace htl {
    class Lexer {
        std::string_view source;

    public:
        /**
         * Construct a Lexer from the provided buffer without taking ownership of the underlying string.
         * The content is assumed to overlive the lifespan of the Lexer
         * @param source
         */
        explicit Lexer(std::string_view source): source(source){}

        Lexer(const Lexer&) = delete;
        Lexer& operator=(const Lexer&) = delete;
    };
} // htl

#endif //PRIMUS_LEXER_HPP
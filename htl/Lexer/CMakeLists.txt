set(HTL_HEADERS Lexer.hpp)
set(HTL_SOURCES Lexer.cpp)

add_library(HtlLexer SHARED ${HTL_HEADERS} ${HTL_SOURCES})
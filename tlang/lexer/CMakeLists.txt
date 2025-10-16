set(TLANG_LEXER_HEADERS Lexer.hpp Token.hpp)
set(TLANG_LEXER_SOURCES Lexer.cpp)

llvm_map_components_to_libnames(LLVM_DEPS core support)

add_library(TLangLexer SHARED ${TLANG_LEXER_HEADERS} ${TLANG_LEXER_SOURCES})
target_link_libraries(TLangLexer ${LLVM_DEPS})
target_compile_options(TLangLexer PUBLIC -fno-rtti)

set(TLANG_PARSER_HEADERS Parser.hpp)
set(TLANG_PARSER_SOURCES Parser.cpp)
add_library(TLangParser SHARED ${TLANG_PARSER_HEADERS} ${TLANG_PARSER_SOURCES})
target_link_libraries(TLangParser ${LLVM_DEPS} TLangLexer)
target_compile_options(TLangParser PUBLIC -fno-rtti)
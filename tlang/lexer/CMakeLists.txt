set(TLANG_HEADERS Lexer.hpp Token.hpp)
set(TLANG_SOURCES Lexer.cpp)

llvm_map_components_to_libnames(LLVM_DEPS core support)

add_library(TLangLexer SHARED ${TLANG_HEADERS} ${TLANG_SOURCES})
target_link_libraries(TLangLexer ${LLVM_DEPS})
target_compile_options(TLangLexer PUBLIC -fno-rtti)
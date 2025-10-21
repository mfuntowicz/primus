add_executable(tlang_tests test_parser.cpp)
target_link_libraries(tlang_tests PUBLIC Catch2::Catch2WithMain TLangParser TLangLexer ${LLVM_DEPS})
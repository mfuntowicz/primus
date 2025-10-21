//
// Created by mfuntowicz on 10/19/25.
//
#include <format>

#include "../frontend/Lexer.hpp"
#include "../frontend/Parser.hpp"

#include "catch2/catch_test_macros.hpp"

TEST_CASE("Parse variable assign statement", "[parser][variable][assign]")
{
    const auto source = "a: int32 = 1;";
    auto parser = tlang::Parser(source, std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto vardecl = decl.value();
        REQUIRE(vardecl.name == "a");
        REQUIRE(vardecl.initializer == "1");

        const auto scalarTy = std::get<tlang::ScalarTy>(vardecl.type);
        REQUIRE(std::get<tlang::IntegerTy>(scalarTy) == tlang::IntegerTy { .width = 32 });
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

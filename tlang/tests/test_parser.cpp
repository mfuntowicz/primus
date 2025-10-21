//
// Created by mfuntowicz on 10/19/25.
//
#include <format>

#include "../frontend/Lexer.hpp"
#include "../frontend/Parser.hpp"

#include "catch2/catch_all.hpp"

TEST_CASE("Parse variable assign statement", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("a: int32 = 1;", std::nullopt);
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

TEST_CASE("Parse variable assign statement without type infos", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("a = 1;", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto vardecl = decl.value();
        REQUIRE(vardecl.name == "a");
        REQUIRE(vardecl.initializer == "1");
        REQUIRE(std::get<tlang::InferTy>(vardecl.type) == tlang::InferTy {});
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

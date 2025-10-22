//
// Created by mfuntowicz on 10/19/25.
//
#include <format>

#include "../frontend/Lexer.hpp"
#include "../frontend/Parser.hpp"

#include "catch2/catch_all.hpp"

TEST_CASE("Parse signed integer variable assign statement", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("a: int32 = 1", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto unit = decl.value();
        const auto decls = unit.Declarations();
        REQUIRE(decls.size() == 1);

        const auto vardecl = decls.front();
        REQUIRE(vardecl.name == "a");
        REQUIRE(vardecl.initializer == "1");

        const auto scalarTy = std::get<tlang::ScalarTy>(vardecl.type);
        REQUIRE(std::get<tlang::SignedIntegerTy>(scalarTy) == tlang::SignedIntegerTy { .width = 32 });
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

TEST_CASE("Parse unsigned integer variable assign statement", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("a: uint64 = 1024", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto unit = decl.value();
        const auto decls = unit.Declarations();
        REQUIRE(decls.size() == 1);

        const auto vardecl = decls.front();
        REQUIRE(vardecl.name == "a");
        REQUIRE(vardecl.initializer == "1024");

        const auto scalarTy = std::get<tlang::ScalarTy>(vardecl.type);
        REQUIRE(std::get<tlang::IntegerTy>(scalarTy) == tlang::IntegerTy { .width = 64 });
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

TEST_CASE("Parse variable assign statement without type infos", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("c = 1", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto unit = decl.value();
        const auto decls = unit.Declarations();
        REQUIRE(decls.size() == 1);

        const auto vardecl = decls.front();
        REQUIRE(vardecl.name == "c");
        REQUIRE(vardecl.initializer == "1");
        REQUIRE(std::get<tlang::InferTy>(vardecl.type) == tlang::InferTy {});
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

TEST_CASE("Parse float32 variable assign statement", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("d: float32 = 32.7", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto unit = decl.value();
        const auto decls = unit.Declarations();
        REQUIRE(decls.size() == 1);

        const auto vardecl = decls.front();
        REQUIRE(vardecl.name == "d");
        REQUIRE(vardecl.initializer == "32.7");

        const auto scalarTy = std::get<tlang::ScalarTy>(vardecl.type);
        REQUIRE(std::get<tlang::FloatTy>(scalarTy) == tlang::FloatTy { .width = 32, .mantissa = 0, .exponent = 0 });
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

TEST_CASE("Parse float8 variable assign statement", "[parser][variable][assign]")
{
    auto parser = tlang::Parser("d: float8 = 32.7", std::nullopt);
    if (const auto decl = parser.Parse(); decl.has_value())
    {
        const auto unit = decl.value();
        const auto decls = unit.Declarations();
        REQUIRE(decls.size() == 1);

        const auto vardecl = decls.front();
        REQUIRE(vardecl.name == "d");
        REQUIRE(vardecl.initializer == "32.7");

        const auto scalarTy = std::get<tlang::ScalarTy>(vardecl.type);
        REQUIRE(std::get<tlang::FloatTy>(scalarTy) == tlang::FloatTy { .width = 8, .mantissa = 0, .exponent = 0 });
    }
    else
    {
        FAIL("Parser returned std::unexpected.");
    }
}

#include <filesystem>
#include <iostream>

#include "cxxopts.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/std.h"

#include "primus.hpp"

void compile(const cxxopts::ParseResult &args)
{
    if (!args.count("input")) {
        SPDLOG_ERROR("no input file provided");
        throw std::runtime_error("No input file provided");
    }

    const auto input = args["input"].as<std::filesystem::path>();
    auto compiler = primus::Compiler::FromFile(input);
}

int main(int argc, char **argv)
{
    // Define CLI args
    cxxopts::Options options("Primus Transformers Compiler");
    options.add_options()
    ("help", "Print usage")
    ("input", "StableHLO file to proceed", cxxopts::value<std::filesystem::path>());
    options.parse_positional({"input"});

    // Parse CLI args and proceed
    auto args = options.parse(argc, argv);
    if (args.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    try
    {
        compile(args);
    } catch (const std::exception &_)
    {
        std::cerr << options.help() << std::endl;
        return 1;
    }
}

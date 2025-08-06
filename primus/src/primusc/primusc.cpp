#include <filesystem>

#include "cxxopts.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/std.h"

#include "primus.hpp"

int main(int argc, char **argv)
{
    cxxopts::Options options("Primus Transformers Compiler");
    options.add_options()
    ("input", "StableHLO file to proceed", cxxopts::value<std::filesystem::path>());
    options.parse_positional({"input"});

    auto args = options.parse(argc, argv);

    if (args.count("input"))
    {
        const auto input = args["input"].as<std::filesystem::path>();
        auto compiler = primus::Compiler::FromFile(input);
    } else
    {
        SPDLOG_ERROR("No input file provided");
        options.help();
    }
}

#include <filesystem>
#include <fstream>

#include "spdlog/spdlog.h"
#include "spdlog/fmt/std.h"

#include "primus/compiler.hpp"

namespace primus
{
    Compiler Compiler::FromFile(std::filesystem::path input)
    {
        SPDLOG_INFO("Processing kernel from {}", input);
        if (std::filesystem::exists(input))
        {
            auto fin = std::ifstream(input);
            return Compiler{};
        }

        throw std::runtime_error("Could not open input file");
    }

}

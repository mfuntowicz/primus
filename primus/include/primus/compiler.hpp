#ifndef PRIMUS_COMPILER_HPP
#define PRIMUS_COMPILER_HPP

#include <filesystem>

namespace primus
{
    class Compiler {
    public:
        /**
         * Create a compiler instance from a file `input`
         * @param input File path to read the kernel definition from
         * @return
         */
        static Compiler FromFile(std::filesystem::path input);
    };
}
#endif // PRIMUS_COMPILER_HPP

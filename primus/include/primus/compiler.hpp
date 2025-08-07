#ifndef PRIMUS_COMPILER_HPP
#define PRIMUS_COMPILER_HPP

#include <filesystem>
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include <mlir/IR/BuiltinOps.h>

namespace primus
{
    class Compiler {
    protected:
        mlir::ModuleOp module;

        explicit Compiler(mlir::ModuleOp module);

    public:
        /**
         * Create a compiler instance from a file `input`
         * @param file File path to read the kernel definition from
         * @return
         */
        static Compiler FromFile(const std::filesystem::path& file);
    };
}
#endif // PRIMUS_COMPILER_HPP

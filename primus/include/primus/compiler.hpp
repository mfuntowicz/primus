#ifndef PRIMUS_COMPILER_HEADER
#define PRIMUS_COMPILER_HEADER

#include <filesystem>
#include <memory>

#include <mlir/IR/BuiltinOps.h>

namespace primus
{
    class Compiler {
    protected:
        std::shared_ptr<mlir::MLIRContext> context;
        mlir::ModuleOp module;

        Compiler(std::shared_ptr<mlir::MLIRContext> context, mlir::ModuleOp module);
        template<bool isDebug> static constexpr mlir::MLIRContext::Threading GetThreadingMode();
    public:
        /**
         * Create a compiler instance from a file `input`
         * @param file File path to read the kernel definition from
         * @return
         */
        static Compiler FromFile(const std::filesystem::path& file);

        void LowerFor(const CompilationTarget& target) const;
    };
}
#endif // PRIMUS_COMPILER_HEADER

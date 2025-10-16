//
// Created by mfuntowicz on 10/15/25.
//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include "../tlang/lexer/Lexer.hpp"

using namespace llvm;


static cl::opt<std::string> file(cl::Positional, cl::desc("Specify input filename"), cl::Required);

int main(const int argc, char** argv)
{
    cl::ParseCommandLineOptions(argc, argv, "tlang exemple lexer printer");
    ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFileOrSTDIN(file);

    if (!MBOrErr)
    {
        errs() << "error: failed to read '" << file << "': "
            << MBOrErr.getError().message() << "\n";
        return 1;
    }

    // Hold onto the owning buffer while the ref is used
    const std::unique_ptr<MemoryBuffer> buffer = std::move(*MBOrErr);
    auto lexer = tlang::Lexer(buffer->getMemBufferRef(), file);
    const auto result = lexer.Lex();
    return 0;
}

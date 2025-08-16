set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS PrimusOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(PrimusOps PrimusOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PrimusTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(PrimusOpTableGen)

add_primus_library(PrimusOps
    AssemblyFormat.cpp
    PrimusDialect.cpp
    PrimusOps.cpp
    Register.cpp

    DEPENDS
    PrimusOpTableGen

    LINK_LIBS PUBLIC
    MLIRIR
    ChloOps
    StablehloOps
    StablehloBase
    StablehloTypeInference
)

target_include_directories(PrimusOps INTERFACE
    $<BUILD_INTERFACE:${PRIMUS_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PRIMUS_BINARY_DIR}>
)
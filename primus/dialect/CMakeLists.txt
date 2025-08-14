set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS PrimusOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(PrimusOps PrimusOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PrimusTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(PrimusTableGen)

add_primus_library(PrimusOps
    PrimusDialect.cpp

    DEPENDS
    PrimusTableGen

    LINK_LIBS PUBLIC
    MLIRIR
    StablehloOps
)
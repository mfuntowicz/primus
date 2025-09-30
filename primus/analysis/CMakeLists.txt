add_mlir_dialect_library(PrimusOpsAnalysis
        MemoryLayoutAnalysis.cpp

        DEPENDS
        PrimusOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
)
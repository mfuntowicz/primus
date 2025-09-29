add_mlir_dialect_library(PrimusOpsTransforms
        BufferizableOpInterfaceImpl.cpp

        DEPENDS
        PrimusOpsIncGen

        LINK_LIBS PUBLIC
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
)
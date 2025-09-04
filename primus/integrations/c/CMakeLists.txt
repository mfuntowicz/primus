add_mlir_public_c_api_library(PrimusCAPI
        PARTIAL_SOURCES_INTENDED
        PrimusDialect.cpp

        LINK_LIBS PUBLIC
        MLIRCAPIIR
        MLIRIR
        MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
        PrimusOps
)
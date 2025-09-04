declare_mlir_python_sources(PrimusPythonSources)
declare_mlir_python_sources(PrimusPythonSources.Dialects
        ADD_TO_PARENT PrimusPythonSources
)

declare_mlir_dialect_python_bindings(
        ADD_TO_PARENT PrimusPythonSources.Dialects
        ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
        TD_FILE dialects/PrimusOps.td
        SOURCES dialects/primus.py
        DIALECT_NAME primus)


###############################################################################
# Generate packages and shared libraries
################################################################################

add_mlir_python_common_capi_library(PrimusPythonCAPI
        INSTALL_COMPONENT PrimusPythonModules
        INSTALL_DESTINATION python_packages/primus/mlir/_mlir_libs
        OUTPUT_DIRECTORY "${PRIMUS_BINARY_DIR}/python_packages/primus/mlir/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../.."
        DECLARED_SOURCES
        MLIRPythonSources
        #        MLIRPythonExtension.RegisterEverything
        PrimusPythonSources
)

add_mlir_python_modules(PrimusPythonModules
        ROOT_PREFIX "${PRIMUS_BINARY_DIR}/python_packages/primus/mlir"
        INSTALL_PREFIX "python_packages/primus/mlir"

        DECLARED_SOURCES
        #        MLIRPythonSources
        #        MLIRPythonExtension.RegisterEverything
        PrimusPythonSources

        COMMON_CAPI_LINK_LIBS
        PrimusPythonCAPI
)
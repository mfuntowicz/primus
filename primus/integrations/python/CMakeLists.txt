include(AddMLIRPython)

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

################################################################################
# Extensions
################################################################################

## The following options are used specifically for compiling nanobind codebase.

declare_mlir_python_sources(PrimusPythonExtensions)
declare_mlir_python_extension(PrimusPythonExtensions.Main
        MODULE_NAME _primus
        ADD_TO_PARENT PrimusPythonExtensions
        PYTHON_BINDINGS_LIBRARY nanobind
        SOURCES
        PrimusModule.cpp
        EMBED_CAPI_LINK_LIBS
        PrimusCAPI
        PRIVATE_LINK_LIBS
        PrimusCAPI
        LLVMSupport
)


###############################################################################
# Generate packages and shared libraries
################################################################################
declare_mlir_python_sources(PrimusWrapperPythonSources
        ADD_TO_PARENT PrimusPythonSources
        ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES
        primus/__init__.py
        primus/module.py
        primus/functional/__init__.py
        primus/torch/__init__.py
        primus/torch/types.py
)

add_mlir_python_common_capi_library(PrimusPythonCAPI
        INSTALL_COMPONENT PrimusPythonModules
        INSTALL_DESTINATION python_packages/primus/mlir/_mlir_libs
        OUTPUT_DIRECTORY "${PRIMUS_BINARY_DIR}/python_packages/primus/mlir/_mlir_libs"
        RELATIVE_INSTALL_ROOT "../../../.."
        DECLARED_SOURCES
        MLIRPythonSources
        MLIRPythonExtension.RegisterEverything
        PrimusPythonSources
        PrimusPythonExtensions
)

add_mlir_python_modules(PrimusPythonModules
        ROOT_PREFIX "${PRIMUS_BINARY_DIR}/python_packages/primus/mlir"
        INSTALL_PREFIX "python_packages/primus/mlir"

        DECLARED_SOURCES
        MLIRPythonSources
        MLIRPythonExtension.RegisterEverything
        PrimusPythonSources
        PrimusPythonExtensions

        COMMON_CAPI_LINK_LIBS
        PrimusPythonCAPI
        Unicron
)
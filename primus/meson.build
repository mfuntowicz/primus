project('primus', ['c', 'cpp'], version : '1.0.0', license : 'Apache 2.0', default_options : ['cpp_std=c++23', 'build.cpp_std=c++23'])

cmake = import('cmake')

# Retrieve compilers
CC = meson.get_compiler('c')
CXX = meson.get_compiler('cpp')

env = environment()
env.prepend('LIBRARY_PATH', '/speedy/Workspace/stablehlo/llvm-build/lib/')

# Global args
if get_option('buildtype') == 'debug'
    add_global_arguments(['-DIS_DEBUG'], language: 'cpp')
endif

# Dependencies
fmt = subproject('fmt')
d_fmt = fmt.get_variable('fmt_dep')

spdlog = subproject('spdlog')
d_spdlog = spdlog.get_variable('spdlog_dep')

cxxopts = subproject('cxxopts')
d_cxxopts = cxxopts.get_variable('cxxopts_dep')

# Retrieve LLVM dependency
d_llvm = dependency(
    'llvm',
    version: '> 21.0',
    static: true,
    modules: ['core', 'codegen', 'linker', 'nvptx', 'option', 'support', 'target', 'passes'],
    cmake_module_path: [
        '/speedy/Workspace/stablehlo/llvm-build/lib/cmake/llvm',
        '/speedy/Workspace/stablehlo/llvm-build/lib/cmake/mlir'
    ],
)

#llvm_config_includes_cmd = run_command('/speedy/Workspace/stablehlo/llvm-build/bin/llvm-config', ['--includedir'], check: true)
#llvm_config_cppflags_cmd = run_command('/speedy/Workspace/stablehlo/llvm-build/bin/llvm-config', ['--cxxflags'], check: true)
#llvm_config_ldargs_cmd = run_command('/speedy/Workspace/stablehlo/llvm-build/bin/llvm-config', ['--ldflags'], check: true)
#llvm_config_libs_cmd = run_command('/speedy/Workspace/stablehlo/llvm-build/bin/llvm-config', ['--libs', 'core', 'support'], check: true)
#llvm_includes = llvm_config_includes_cmd.stdout().strip()
#llvm_cppflags = llvm_config_cppflags_cmd.stdout().strip()
#llvm_ldargs = llvm_config_ldargs_cmd.stdout().strip()
#llvm_libs = llvm_config_libs_cmd.stdout().strip()
#llvm_ldargs = llvm_ldargs + llvm_libs

# Build up MLIR dependencies...
mlir_inc = include_directories('/speedy/Workspace/stablehlo/llvm-project/mlir/include')
d_mlir_targets = [
    'AffineAnalysis',
    'AffineDialect',
    'AffineToStandard',
    'AffineTransformOps',
    'AffineTransforms',
    'AffineUtils',
    'AMDGPUDialect',
    'AMDGPUToROCDL',
    'AMDGPUTransforms',
    'AMDGPUUtils',
    'AMXDialect',
    'AMXToLLVMIRTranslation',
    'AMXTransforms',
    'Analysis',
    'ArithAttrToLLVMConversion',
    'ArithDialect',
    'ArithToAMDGPU',
    'ArithToArmSME',
    'ArithToEmitC',
    'ArithToLLVM',
    'ArithToSPIRV',
    'ArithTransforms',
    'ArithUtils',
    'ArithValueBoundsOpInterfaceImpl',
    'ArmNeon2dToIntr',
    'ArmNeonDialect',
    'ArmNeonToLLVMIRTranslation',
    'ArmNeonTransforms',
    'ArmNeonVectorTransformOps',
    'ArmSMEDialect',
    'ArmSMEToLLVM',
    'ArmSMEToLLVMIRTranslation',
    'ArmSMEToSCF',
    'ArmSMETransforms',
    'ArmSVEDialect',
    'ArmSVEToLLVMIRTranslation',
    'ArmSVETransforms',
    'ArmSVEVectorTransformOps',
    'AsmParser',
    'AsyncDialect',
    'AsyncToLLVM',
    'AsyncTransforms',
    'BufferizationDialect',
    'BufferizationPipelines',
    'BufferizationToMemRef',
    'BufferizationTransformOps',
    'BufferizationTransforms',
    'BuiltinToLLVMIRTranslation',
    'BytecodeOpInterface',
    'BytecodeReader',
    'BytecodeWriter',
    'CallInterfaces',
    'CAPIAMDGPU',
    'CAPIArith',
    'CAPIAsync',
    'CAPIControlFlow',
    'CAPIConversion',
    'CAPIDebug',
    'CAPIEmitC',
    'CAPIExecutionEngine',
    'CAPIExportSMTLIB',
    'CAPIFunc',
    'CAPIGPU',
    'CAPIIndex',
    'CAPIInterfaces',
    'CAPIIR',
    'CAPIIRDL',
    'CAPILinalg',
    'CAPILLVM',
    'CAPIMath',
    'CAPIMemRef',
    'CAPIMLProgram',
    'CAPINVGPU',
    'CAPINVVM',
    'CAPIOpenMP',
    'CAPIPDL',
    'CAPIQuant',
    'CAPIRegisterEverything',
    'CAPIROCDL',
    'CAPISCF',
    'CAPIShape',
    'CAPISMT',
    'CAPISparseTensor',
    'CAPISPIRV',
    'CAPITarget',
    'CAPITensor',
    'CAPITransformDialect',
    'CAPITransformDialectTransforms',
    'CAPITransforms',
    'CAPIVector',
    'CastInterfaces',
    'ComplexDialect',
    'ComplexDivisionConversion',
    'ComplexToLibm',
    'ComplexToLLVM',
    'ComplexToSPIRV',
    'ComplexToStandard',
    'ControlFlowDialect',
    'ControlFlowInterfaces',
    'ControlFlowToLLVM',
    'ControlFlowToSCF',
    'ControlFlowToSPIRV',
    'ControlFlowTransforms',
    'ConvertToEmitC',
    'ConvertToLLVMInterface',
    'ConvertToLLVMPass',
    'CopyOpInterface',
    'DataLayoutInterfaces',
    'Debug',
    'DerivedAttributeOpInterface',
    'DestinationStyleOpInterface',
    'Dialect',
    'DialectUtils',
    'DLTIDialect',
    'DLTITransformOps',
    'EmitCDialect',
    'EmitCTransforms',
    'ExecutionEngine',
    'ExecutionEngineUtils',
    'ExportSMTLIB',
    'FromLLVMIRTranslationRegistration',
    'FuncAllExtensions',
    'FuncDialect',
    'FuncInlinerExtension',
    'FuncMeshShardingExtensions',
    'FunctionInterfaces',
    'FuncToEmitC',
    'FuncToLLVM',
    'FuncToSPIRV',
    'FuncTransformOps',
    'FuncTransforms',
    'FuncUtils',
    'GPUDialect',
    'GPUPipelines',
    'GPUToGPURuntimeTransforms',
    'GPUToLLVMIRTranslation',
    'GPUToLLVMSPV',
    'GPUToNVVMTransforms',
    'GPUToROCDLTransforms',
    'GPUToSPIRV',
    'GPUToVulkanTransforms',
    'GPUTransformOps',
    'GPUTransforms',
    'GPUUtils',
    'IndexDialect',
    'IndexingMapOpInterface',
    'IndexToLLVM',
    'IndexToSPIRV',
    'InferIntRangeCommon',
    'InferIntRangeInterface',
    'InferTypeOpInterface',
    'IR',
    'IRDL',
    'JitRunner',
    'LinalgDialect',
    'LinalgToStandard',
    'LinalgTransformOps',
    'LinalgTransforms',
    'LinalgUtils',
    'LLVMCommonConversion',
    'LLVMDialect',
    'LLVMIRToLLVMTranslation',
    'LLVMIRToNVVMTranslation',
    'LLVMIRTransforms',
    'LLVMToLLVMIRTranslation',
    'LoopLikeInterface',
    'LspServerLib',
    'LspServerSupportLib',
    'MaskableOpInterface',
    'MaskingOpInterface',
    'MathDialect',
    'MathToEmitC',
    'MathToFuncs',
    'MathToLibm',
    'MathToLLVM',
    'MathToROCDL',
    'MathToSPIRV',
    'MathTransforms',
    'MemorySlotInterfaces',
    'MemRefDialect',
    'MemRefToEmitC',
    'MemRefToLLVM',
    'MemRefToSPIRV',
    'MemRefTransformOps',
    'MemRefTransforms',
    'MemRefUtils',
    'MeshDialect',
    'MeshToMPI',
    'MeshTransforms',
    'MlirOptMain',
    'MLProgramDialect',
    'MLProgramTransforms',
    'MPIDialect',
    'MPIToLLVM',
    'NVGPUDialect',
    'NVGPUToNVVM',
    'NVGPUTransformOps',
    'NVGPUTransforms',
    'NVGPUUtils',
    'NVVMDialect',
    'NVVMTarget',
    'NVVMToLLVM',
    'NVVMToLLVMIRTranslation',
    'Observers',
    'OpenACCDialect',
    'OpenACCMPCommon',
    'OpenACCToLLVMIRTranslation',
    'OpenACCToSCF',
    'OpenACCTransforms',
    'OpenMPDialect',
    'OpenMPToLLVM',
    'OpenMPToLLVMIRTranslation',
    'OptLib',
    'ParallelCombiningOpInterface',
    'Parser',
    'Pass',
    'PDLDialect',
    'PDLInterpDialect',
    'PDLLAST',
    'PDLLCodeGen',
    'PdllLspServerLib',
    'PDLLODS',
    'PDLLParser',
    'PDLToPDLInterp',
    'PluginsLib',
    'PolynomialDialect',
    'Presburger',
    'PtrDialect',
    'QuantDialect',
    'QuantTransforms',
    'QuantUtils',
    'Query',
    'QueryLib',
    'QueryMatcher',
    'ReconcileUnrealizedCasts',
    'Reduce',
    'ReduceLib',
    'Rewrite',
    'RewritePDL',
    'ROCDLDialect',
    'ROCDLTarget',
    'ROCDLToLLVMIRTranslation',
    'RuntimeVerifiableOpInterface',
    'SCFDialect',
    'SCFToControlFlow',
    'SCFToEmitC',
    'SCFToGPU',
    'SCFToOpenMP',
    'SCFToSPIRV',
    'SCFTransformOps',
    'SCFTransforms',
    'SCFUtils',
    'ShapeDialect',
    'ShapedOpInterfaces',
    'ShapeOpsTransforms',
    'ShapeToStandard',
    'ShardingInterface',
    'SideEffectInterfaces',
    'SMT',
    'SparseTensorDialect',
    'SparseTensorPipelines',
    'SparseTensorRuntime',
    'SparseTensorTransformOps',
    'SparseTensorTransforms',
    'SparseTensorUtils',
    'SPIRVAttrToLLVMConversion',
    'SPIRVBinaryUtils',
    'SPIRVConversion',
    'SPIRVDeserialization',
    'SPIRVDialect',
    'SPIRVImageInterfaces',
    'SPIRVModuleCombiner',
    'SPIRVSerialization',
    'SPIRVTarget',
    'SPIRVToLLVM',
    'SPIRVToLLVMIRTranslation',
    'SPIRVTransforms',
    'SPIRVTranslateRegistration',
    'SPIRVUtils',
    'SubsetOpInterface',
    'Support',
    'TableGen',
    'TargetCpp',
    'TargetIRDLToCpp',
    'TargetLLVM',
    'TargetLLVMIRExport',
    'TargetLLVMIRImport',
    'TblgenLib',
    'TensorAllExtensions',
    'TensorDialect',
    'TensorInferTypeOpInterfaceImpl',
    'TensorMeshShardingExtensions',
    'TensorTilingInterfaceImpl',
    'TensorToLinalg',
    'TensorToSPIRV',
    'TensorTransformOps',
    'TensorTransforms',
    'TensorUtils',
    'TilingInterface',
    'ToLLVMIRTranslationRegistration',
    'TosaDialect',
    'TosaShardingInterfaceImpl',
    'TosaToArith',
    'TosaToLinalg',
    'TosaToMLProgram',
    'TosaToSCF',
    'TosaToTensor',
    'TosaTransforms',
    'TransformDebugExtension',
    'TransformDialect',
    'TransformDialectInterfaces',
    'TransformDialectIRDLExtension',
    'TransformDialectTransforms',
    'TransformDialectUtils',
    'TransformLoopExtension',
    'TransformPDLExtension',
    'Transforms',
    'TransformUtils',
    'TranslateLib',
    'UBDialect',
    'UBToLLVM',
    'UBToSPIRV',
    'ValueBoundsOpInterface',
    'VCIXDialect',
    'VCIXToLLVMIRTranslation',
    'VectorDialect',
    'VectorInterfaces',
    'VectorToArmSME',
    'VectorToGPU',
    'VectorToLLVM',
    'VectorToLLVMPass',
    'VectorToSCF',
    'VectorToSPIRV',
    'VectorToXeGPU',
    'VectorTransformOps',
    'VectorTransforms',
    'VectorUtils',
    'ViewLikeInterface',
    'X86VectorDialect',
    'X86VectorToLLVMIRTranslation',
    'X86VectorTransforms',
    'XeGPUDialect',
    'XeGPUTransforms',
    'XeGPUUtils',
]
d_mlir = []

foreach target : d_mlir_targets
    d_target = CXX.find_library(
        'MLIR' + target,
        dirs: ['/speedy/Workspace/stablehlo/llvm-build/lib'],
        header_include_directories: mlir_inc,
        static: true
    )

    d_mlir += d_target
endforeach

# Look for StableHLO standalone
#environment({
#  'CMAKE_MODULE_PATH': '/speedy/Workspace/stablehlo/llvm-build/lib/cmake/llvm:/speedy/Workspace/stablehlo/llvm-build/lib/cmake/mlir'
#}, method: 'prepend', separator: ':')
#
#stablehlo_opts = cmake.subproject_options()
#stablehlo_opts.add_cmake_defines({
#   'LLVM_ENABLE_ASSERTIONS': true,
#   'LLVM_DIR': '/speedy/Workspace/stablehlo/llvm-build/lib/cmake/llvm',
#   'MLIR_DIR': '/speedy/Workspace/stablehlo/llvm-build/lib/cmake/mlir',
#   'STABLEHLO_ENABLE_LLD': true,
#   'STABLEHLO_ENABLE_BINDINGS_PYTHON': false,
#   'STABLEHLO_STANDALONE_BUILD': true,
##   'STABLEHLO_BUILD_EMBEDDED': true,
#})

#stablehlo = cmake.subproject('stablehlo', options: stablehlo_opts)
stablehlo_path = '/speedy/Workspace/stablehlo'
stablehlo_build_path = stablehlo_path / 'build'
stablehlo_llvm_build_path = stablehlo_path / 'llvm_build'

stablehlo_includes = [include_directories(stablehlo_path, stablehlo_build_path)]
stablehlo_chlo_ops = CXX.find_library('ChloOps', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_vhlo_ops = CXX.find_library('VhloOps', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)

stablehlo_vhlo_types = CXX.find_library('VhloTypes', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_assembly_format = CXX.find_library('StablehloAssemblyFormat', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_base = CXX.find_library('StablehloBase', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_broadcast_utils = CXX.find_library('StablehloBroadcastUtils', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_ops = CXX.find_library('StablehloOps', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_linalg_transforms = CXX.find_library('StablehloLinalgTransforms', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_optim_passes = CXX.find_library('StablehloOptimizationPasses', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_type_conversion = CXX.find_library('StablehloTypeConversion', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_register = CXX.find_library('StablehloRegister', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_ref_api = CXX.find_library('StablehloReferenceApi', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_passes = CXX.find_library('StablehloPasses', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_passes_utils = CXX.find_library('StablehloPassUtils', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_type_inference = CXX.find_library('StablehloTypeInference', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
stablehlo_version = CXX.find_library('Version', dirs: [stablehlo_build_path / 'lib'], header_include_directories: stablehlo_includes, static: true)
d_stablehlo = [stablehlo_assembly_format, stablehlo_base, stablehlo_broadcast_utils, stablehlo_ops, stablehlo_chlo_ops, stablehlo_optim_passes, stablehlo_linalg_transforms, stablehlo_vhlo_ops, stablehlo_vhlo_types, stablehlo_register, stablehlo_ref_api, stablehlo_type_conversion, stablehlo_passes, stablehlo_passes_utils, stablehlo_type_inference, stablehlo_version]

# Primus and derivates definitions
inc = include_directories('include')
subdir('include')
subdir('src')
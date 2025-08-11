stablehlo_llvm_lib_path = stablehlo_llvm_build_path / 'lib'

# Primus library definition
primus_sources = [
    'primus/compiler.cpp', 'primus/targets/cuda.cpp'
]
primus = shared_library(
    'primus',
    primus_sources,
#    include_directories: [inc] + stablehlo_includes + llvm_includes + mlir_inc,
    include_directories: [inc] + stablehlo_includes + mlir_inc,
    dependencies: [
        d_fmt,
        d_llvm,
        d_mlir,
        d_spdlog,
        d_stablehlo
    ],
    install: true,
#    link_args: [llvm_ldargs, '-lLLVMCore', '-lLLVMRemarks', '-lLLVMBitstreamReader', '-lLLVMBinaryFormat', '-lLLVMTargetParser', '-lLLVMSupport', '-lLLVMDemangle'],
    cpp_args: ['-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1']
)

# Primusc compiler definition
primusc_sources = ['primusc/primusc.cpp']
primusc = executable(
    'primusc',
    primusc_sources,
    include_directories: [inc] + stablehlo_includes,
    dependencies: [d_cxxopts, d_fmt, d_spdlog],
    link_with: primus,
#    link_args: [llvm_ldargs],
    cpp_args: ['-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1']
)
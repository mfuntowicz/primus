# Primus library definition
primus_sources = [
    'primus/compiler.cpp',
]
primus = shared_library(
    'primus',
    primus_sources,
    include_directories: inc,
    dependencies: [d_fmt, d_spdlog],
    install: true
)

# Primusc compiler definition
primusc_sources = ['primusc/primusc.cpp']
primusc = executable(
    'primusc',
    primusc_sources,
    include_directories: inc,
    dependencies: [d_cxxopts, d_fmt, d_spdlog],
    link_with: primus
)
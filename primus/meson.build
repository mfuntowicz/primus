project('primus', 'cpp', version : '1.0.0', license : 'Apache 2.0', default_options : ['cpp_std=c++23', 'build.cpp_std=c++23'])


# Dependencies
fmt = subproject('fmt')
d_fmt = fmt.get_variable('fmt_dep')

spdlog = subproject('spdlog')
d_spdlog = spdlog.get_variable('spdlog_dep')

cxxopts = subproject('cxxopts')
d_cxxopts = cxxopts.get_variable('cxxopts_dep')

# Primus and derivates definitions
inc = include_directories('include')
subdir('include')
subdir('src')
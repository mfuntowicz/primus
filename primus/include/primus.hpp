#ifndef PRIMUS_HPP
#define PRIMUS_HPP

#ifdef IS_DEBUG
#if IS_DEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#endif
#endif

#include "mlir/IR/ImplicitLocOpBuilder.h"

#include "primus/targets/base.hpp"
#include "primus/targets/cuda.hpp"
#include "primus/compiler.hpp"
#endif // PRIMUS_HPP

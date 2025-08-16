//
// Created by mfuntowicz on 8/14/25.
//

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>

#include "primus/dialect/PrimusOps.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"

#include "primus/dialect/AssemblyFormat.hpp"

#define GET_OP_CLASSES
#include "primus/dialect/Ops.cpp.inc"
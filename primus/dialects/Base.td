// Copyright Morgan Funtowicz (c) 2025.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*

#ifndef PRIMUS_DIALECT_BASE
#define PRIMUS_DIALECT_BASE

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------===//
// Primus scalar types
//===----------------------------------------------===//
def Primus_Float : AnyTypeOf<[
        F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
        F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2,
        F8E5M2FNUZ, F8E8M0FNU, F16, F32, F64, BF16
    ], "4/6/8/16/32/64-bit float">;


//===----------------------------------------------===//
// Primus tensor types
//===----------------------------------------------===//
def PrimusFloatTensor: RankedTensorOf<[Primus_Float]>;

#endif //PRIMUS_DIALECT_BASE
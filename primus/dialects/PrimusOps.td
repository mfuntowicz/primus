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

#ifndef PRIMUS_DIALECT_PRIMUS_OPS
#define PRIMUS_DIALECT_PRIMUS_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "primus/dialects/Base.td"


def Primus_Dialect : Dialect {
  let name = "primus";
  let cppNamespace = "::mlir::primus";

  let description = [{
    Primus is a high-level, Large-Language-Models (LLM) kernel primitive library.
  }];
}


class Primus_Op<string mnemonic, list<Trait> traits = []>
    : Op<Primus_Dialect, mnemonic, traits> {}


def Primus_RotaryOp : Primus_Op<"rotary", [Elementwise, InferShapedTypeOpInterface, NoMemoryEffect]> {
 let summary = "Rotary embedding kernel";
  let description = [{
    Performs element-wise rotary operation on `operand` tensor and produces a `result` tensor.

    Example:
    ```mlir
    %result = primus.rotary %operand : tensor<?x?x?xf32>
    ```
  }];
  let arguments = (ins PrimusFloatTensor:$operand, PrimusFloatTensor:$cos, PrimusFloatTensor:$sin);
  let results = (outs PrimusFloatTensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $operand `,` $cos `,` $sin `)` attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($cos), type($sin), type($result))
  }];
}

#endif //PRIMUS_DIALECT_PRIMUS_OPS
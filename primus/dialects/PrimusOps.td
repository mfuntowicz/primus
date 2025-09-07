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

class Primus_ElementwiseOp<string mnemonic>
    : Primus_Op<mnemonic, [Elementwise, InferShapedTypeOpInterface, NoMemoryEffect, Scalarizable, Vectorizable, Tensorizable]> {}


def Primus_RotaryOp : Primus_Op<"rotary", [InferShapedTypeOpInterface, NoMemoryEffect]> {
 let summary = "Rotary embedding kernel";
  let description = [{
    Performs element-wise rotary operation on `operand` tensor according to the cosinus and sinus frequencies table
    provided and write the result to `out` tensor, inplace.

    - `x` operand is expected to be rank-3 tensors of statically or dynamically shaped dimensions.
    - `cos` and `sin` operands are expected to be rank-2 tensors of shape <max_sequence_length x hidden_dim> with statically shaped dimensions.

    Example:
    ```mlir
    %result = primus.rotary %operand : tensor<?x?x?xf32>
    ```
  }];

  let arguments = (ins
    Arg<PrimusRank3FloatTensor, "the source tensor", []>:$x,
    Arg<PrimusStaticRank2FloatTensor, "the cosinus frequencies table", []>:$cos,
    Arg<PrimusStaticRank2FloatTensor, "the sinus frequencies table", []>:$sin
  );

  let results = (outs PrimusFloatTensor:$result);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $x `,` $cos `,` $sin `)` attr-dict `:` custom<SameOperandsAndResultType>(type($x), type($result)) `,` custom<SameOperandsAndResultType>(type($cos), type($sin))
  }];
}

#endif //PRIMUS_DIALECT_PRIMUS_OPS
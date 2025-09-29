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

#ifndef PRIMUS_DIALECT_OPTIMUS_ATTRS
#define PRIMUS_DIALECT_OPTIMUS_ATTRS

include "mlir/IR/OpBase.td"

def Optimus_TargetDescriptor : AttrDef<Primus_Dialect, "TargetDescriptor"> {
    let mnemonic = "target_descriptor";
    let summary = "Attribute that describes target information";
      let parameters = (ins
          StringRefParameter<"Target name">:$name
      );
      let assemblyFormat = "`<` `name` `=` $name `>`";
}


#endif //PRIMUS_DIALECT_OPTIMUS_ATTRS
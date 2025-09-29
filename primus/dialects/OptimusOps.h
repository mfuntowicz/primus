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

//
// Created by mfuntowicz on 9/23/25.
//

#ifndef PRIMUS_OPTIMUSOPS_H
#define PRIMUS_OPTIMUSOPS_H

#define GET_ATTRDEF_CLASSES
#include "primus/dialects/OptimusAttrs.h.inc"

namespace mlir::optimus
{
    class OptimusDialect : public mlir::Dialect
    {
    public:
        explicit OptimusDialect(MLIRContext* context);

        static StringRef getDialectNamespace() { return "optimus"; }
    };
}

#define GET_OP_CLASSES
#include "primus/dialects/OptimusOps.h.inc"


#endif //PRIMUS_OPTIMUSOPS_H
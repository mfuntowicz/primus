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
// Created by mfuntowicz on 9/29/25.
//

#ifndef PRIMUS_MEMORYLAYOUTANALYSIS_H
#define PRIMUS_MEMORYLAYOUTANALYSIS_H

#include "llvm/Support/Error.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::primus
{
    /**
     *
     * @param type
     * @return
     */
    bool isContiguous(const MemRefType& type);

    /**
     *
     * @param type
     * @param dim
     * @return
     */
    bool isContiguous(const MemRefType& type, uint64_t dim);
}


#endif //PRIMUS_MEMORYLAYOUTANALYSIS_H

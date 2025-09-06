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
// Created by momo- on 9/5/2025.
//

#ifndef PRIMUS_REGISTER_H
#define PRIMUS_REGISTER_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::primus {
    /**
     * Add Primus dialects to the provided registry
     * @param registry Registry to add the primus dialects
     */
    void registerAllDialects(DialectRegistry &registry);

}

#endif //PRIMUS_REGISTER_H
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

#ifndef PRIMUS_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H
#define PRIMUS_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::primus {

    //===----------------------------------------------------------------------===//
    // General Primus lowering patterns.
    //===----------------------------------------------------------------------===//

    /// Populates the patterns that convert from Primus to Linalg on tensors.
    void populatePrimusToLinalgConversionPatterns(
        MLIRContext *context, RewritePatternSet *patterns);

    //===----------------------------------------------------------------------===//
    // Fine-grained patterns used by the implementation.
    //===----------------------------------------------------------------------===//
    namespace detail {
        /// Populates the patterns that convert from elementwise Primus ops to Linalg
        /// on tensors.
        void populatePrimusRotaryToLinalgConversionPatterns(MLIRContext *context, RewritePatternSet *patterns);
    }
}



#endif //PRIMUS_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H
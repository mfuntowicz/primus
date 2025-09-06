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
// Created by momo- on 9/4/2025.
//

#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
// #include "primus/dialects/PrimusOps.h"
// #include "primus/conversions/stablehlo/transforms/Passes.h"
#include "primus/dialects/Register.h"

int main(const int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::stablehlo::registerPassPipelines();
    mlir::stablehlo::registerPasses();
    mlir::stablehlo::registerOptimizationPasses();
    mlir::stablehlo::registerStablehloLinalgTransformsPasses();
    // mlir::primus::registerPrimusLegalizeToStablehloPass();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::primus::registerAllDialects(registry);

    return failed(
        mlir::MlirOptMain(argc, argv, "Primus optimizer driver\n", registry));
}

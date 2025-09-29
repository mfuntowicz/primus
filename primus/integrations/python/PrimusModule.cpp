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

#include <llvm/TargetParser/Host.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "primus/integrations/c/PrimusDialect.h"

namespace
{
    NB_MODULE(_primus, m)
    {
        m.doc() = "Primus MLIR binding";

        m.def("register_dialect", [](const MlirContext context, const bool load)
              {
                  const MlirDialectHandle dialect = mlirGetDialectHandle__primus__();
                  mlirDialectHandleRegisterDialect(dialect, context);
                  if (load)
                  {
                      mlirDialectHandleLoadDialect(dialect, context);
                  }
              },
              nanobind::arg("context"), nanobind::arg("load") = true
        );

        m.def("get_host_triple", []()
        {
            return std::string(llvm::sys::getProcessTriple());
        });
    }
}
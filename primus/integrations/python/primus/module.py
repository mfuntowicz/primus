#  Copyright Morgan Funtowicz (c) 2025.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================*/
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================*/
from abc import ABC, abstractmethod
from typing import Protocol

from torch import nn

from mlir._mlir_libs._mlir.ir import *


class Symbol(Protocol):
    """
    Represent an identifiable entity
    """

    NAME: str

    def mangle(self, **kwargs):
        ...


class PrimusModule(nn.Module, Symbol, ABC):
    """

    """

    def __repr__(self):
        return f"@{self.NAME}[]"

    def __str__(self):
        return f"@{self.NAME}"

    @abstractmethod
    def __assembly__(self, module: Module, **kwargs):
        raise NotImplementedError("Method __materialize__ is abstract and should be implemented")

    def __materialize__(self, **kwargs) -> Module:
        with Context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                self.__assembly__(module, **kwargs)

            return module

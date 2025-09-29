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

import torch
# from primus.mlir.dialects import primus as primusops, func
# from primus.mlir._mlir_libs import _mlir as mlir
# from primus.mlir._mlir_libs._mlir import ir
# from primus.mlir._mlir_libs._mlir.ir import *

from mlir.primus.module import PrimusModule
from mlir.primus.torch import buffer_from_torch_tensor, view_buffer_as_tensor
from mlir.dialects import primus as primus
from mlir.dialects import func, bufferization, tensor
from mlir._mlir_libs._mlir.ir import *
from transformers import GptOssConfig, AutoConfig


class GptOssRotary(PrimusModule):
    NAME = "rotary_embedding_fwd"

    __slots__ = ("num_q_attention_heads", "num_k_v_attention_heads", "head_size")

    def __init__(self, config: GptOssConfig):
        self.num_q_attention_heads = config.num_attention_heads
        self.num_k_v_attention_heads = config.num_key_value_heads
        self.head_size = config.head_dim

    def __assembly__(self, mod: "Module", q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        primus.register_dialect(mod.context)

        q_ty = buffer_from_torch_tensor(q)
        k_ty = buffer_from_torch_tensor(k)
        cos_ty = buffer_from_torch_tensor(cos)
        sin_ty = buffer_from_torch_tensor(sin)

        @func.FuncOp.from_py_func(q_ty, k_ty, cos_ty, sin_ty)
        def rotary_embedding_fwd(q_b, k_b, cos_b, sin_b):
            # Extract the necessary attributes
            int64_t = IntegerType.get_signless(64)

            head_size = IntegerAttr.get(int64_t, self.head_size)
            num_q_heads = IntegerAttr.get(int64_t, self.num_q_attention_heads)
            num_k_heads = IntegerAttr.get(int64_t, self.num_k_v_attention_heads)

            q_t, k_t, cos_t, sin_t = (
                view_buffer_as_tensor(q_b, restrict=True, writable=True),
                view_buffer_as_tensor(k_b, restrict=True, writable=True),
                view_buffer_as_tensor(cos_b, restrict=True, writable=False),
                view_buffer_as_tensor(sin_b, restrict=True, writable=False)
            )

            q_rotated = primus.rotary(
                q_t.type, q_t, cos_t, sin_t,
                num_heads=num_q_heads,
                head_size=head_size
            )

            k_rotated = primus.rotary(
                k_t.type, k_t, cos_t, sin_t,
                num_heads=num_k_heads,
                head_size=head_size
            )

            bufferization.materialize_in_destination(q_ty, q_rotated, q_b, restrict=True, writable=True),
            bufferization.materialize_in_destination(k_ty, k_rotated, k_b, restrict=True, writable=True)


if __name__ == '__main__':
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
    rotary = GptOssRotary(config)

    head_size = config.head_dim

    q = torch.randn((1, 128, config.num_attention_heads * head_size), dtype=torch.bfloat16)
    k = torch.randn((1, 128, config.num_key_value_heads * head_size), dtype=torch.bfloat16)
    cos = torch.randn((1, 128, 32), dtype=torch.bfloat16)
    sin = torch.randn((1, 128, 32), dtype=torch.bfloat16)

    module = rotary.__materialize__(q=q.view(1, 128, -1, head_size).transpose(2, 1),
                                    k=k.view(1, 128, -1, head_size).transpose(2, 1), cos=cos, sin=sin)
    print(module)

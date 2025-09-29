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
import functools
import inspect
from typing import Optional, Callable, Any

import torch
from mlir._mlir_libs._mlir.ir import *
from mlir.dialects import bufferization


def as_type(dtype: torch.dtype) -> "Type":
    """Convert a PyTorch dtype to the corresponding MLIR Type.

    This function maps PyTorch data types to their equivalent MLIR type representations,
    enabling interoperability between PyTorch tensors and MLIR operations.
    
    Types are bound to the current MLIRContext.

    Args:
        dtype: A PyTorch dtype (e.g., torch.float32, torch.int8, torch.bool)

    Returns:
        The corresponding MLIR Type object:
        - Float types: F32Type, F64Type, F16Type, BF16Type
        - Integer types: IntegerType with appropriate signedness and bit width
        - Complex types: ComplexType with float element types
        - Quantized types: IntegerType representing the storage format

    Raises:
        ValueError: If the provided dtype is not supported

    Examples:
        >>> as_type(torch.float32)
        F32Type
        >>> as_type(torch.int8)
        IntegerType(signless, 8)
        >>> as_type(torch.complex64)
        ComplexType(F32Type)
    """
    match dtype:
        case torch.float32:
            return F32Type.get()
        case torch.float64:
            return F64Type.get()
        case torch.float16:
            return F16Type.get()
        case torch.bfloat16:
            return BF16Type.get()
        case torch.int8:
            return IntegerType.get_signless(8)
        case torch.int16:
            return IntegerType.get_signless(16)
        case torch.int32:
            return IntegerType.get_signless(32)
        case torch.int64:
            return IntegerType.get_signless(64)
        case torch.uint8:
            return IntegerType.get_unsigned(8)
        case torch.bool:
            return IntegerType.get_signless(1)
        case torch.complex64:
            return ComplexType.get(F32Type.get())
        case torch.complex128:
            return ComplexType.get(F64Type.get())
        case torch.qint8:
            return IntegerType.get_signed(8)
        case torch.quint8:
            return IntegerType.get_unsigned(8)
        case torch.qint32:
            return IntegerType.get_signed(32)
        case torch.quint4x2:
            return IntegerType.get_unsigned(8)
        case _:
            raise ValueError(f"Unsupported torch.dtype: {dtype}")


def get_memory_space(tensor: torch.Tensor) -> Optional["Attribute"]:
    """Determine the MLIR memory space based on PyTorch tensor device.

    This function maps PyTorch device types to corresponding MLIR memory space
    identifiers, enabling proper representation of tensors residing in different
    memory hierarchies (CPU, GPU, etc.).

    Args:
        tensor: A PyTorch tensor with a specific device location

    Returns:
        Memory space attribute:
        - None: Default/host memory space (CPU)
        - IntegerAttr(1): GPU global memory space
        - IntegerAttr(2): Apple Metal Performance Shaders

    Examples:
        >>> cpu_tensor = torch.randn(3, 4)
        >>> get_memory_space(cpu_tensor)
        None

        >>> gpu_tensor = torch.randn(3, 4, device='cuda')
        >>> get_memory_space(gpu_tensor)
        IntegerAttr(1)
    """
    match tensor.device.type:
        case 'cuda':
            return IntegerAttr.get(IntegerType.get_signless(64), 1)  # GPU global memory space
        case 'mps':
            return IntegerAttr.get(IntegerType.get_signless(64), 2)  # Apple Metal Performance Shaders
        case _:
            return None  # Default for cpu/unknown devices


def buffer_from_torch_tensor(t: torch.Tensor) -> "MemRefType":
    """Create an MLIR MemRefType from a PyTorch tensor's shape, dtype, and layout.

    This function converts a PyTorch tensor to its corresponding MLIR MemRefType,
    which represents a multi-dimensional buffer with known shape, element type,
    and memory layout (strides). The MemRefType preserves the tensor's memory
    layout, enabling correct handling of non-contiguous tensors.

    Args:
        t: A PyTorch tensor with defined shape, dtype, and stride information
    Returns:
        An MLIR MemRefType with the tensor's shape, converted element type,
        and stride layout information

    Examples:
        >>> tensor = torch.randn(3, 4, dtype=torch.float32)
        >>> memref_type = buffer_from_torch_tensor(tensor)
        >>> # Returns MemRefType with shape [3, 4], F32Type elements, and strides [4, 1]

        >>> transposed = tensor.t()  # Non-contiguous tensor
        >>> memref_type = from_allocated_tensor(transposed)
        >>> # Returns MemRefType with shape [4, 3], F32Type elements, and strides [1, 4]
    """
    # Create stride-based layout using StridedLayoutAttr
    # Offset is typically 0 for PyTorch tensors
    layout = None if t.is_contiguous() else StridedLayoutAttr.get(0, t.stride())

    return MemRefType.get(t.shape, as_type(t.dtype), memory_space=get_memory_space(t), layout=layout)


def view_buffer_as_tensor(
        buffer: "Value",
        *,
        restrict: bool = False,
        writable: bool = False
) -> "RankedTensorType":
    """Convert an allocated buffer from MLIR MemRefType to a MLIR's Tensor type (RankedTensorType)."""
    ty = RankedTensorType.get(buffer.type.shape, buffer.type.element_type, )

    return bufferization.to_tensor(ty, buffer, restrict=restrict, writable=writable)


def as_tensor(tensor_param: str, *, restrict: bool = True, writable: bool = False,
              desc: Optional[str] = None) -> Callable:
    """Decorator that converts a specific torch.Tensor parameter to MemRefType using buffer_from_torch_tensor.

    This decorator automatically transforms a named torch.Tensor parameter into its corresponding
    MLIR MemRefType representation before passing it to the decorated function. This enables
    seamless interoperability between PyTorch tensors and MLIR operations.

    Args:
        tensor_param: Name of the parameter to convert from torch.Tensor to MemRefType
        restrict: Whether the buffer has restricted aliasing (default: True)
        readable: Whether the buffer can be read from (default: True)
        writable: Whether the buffer can be written to (default: False)
        desc: Optional description of the operand

    Returns:
        A decorator function that performs automatic tensor-to-buffer conversion
        for the specified parameter.

    Examples:
        >>> @as_tensor('input_tensor', readable=True, writable=False, desc='Input data')
        ... def process_tensor(input_tensor, other_param):
        ...     # input_tensor is now MemRefType if it was torch.Tensor
        ...     return input_tensor

        >>> tensor = torch.randn(3, 4)
        >>> memref = process_tensor(tensor, "other")
        >>> # memref is a MemRefType object
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert the specified tensor parameter if it exists and is a torch.Tensor
            if tensor_param in bound_args.arguments:
                value = bound_args.arguments[tensor_param]
                if isinstance(value, torch.Tensor):
                    buffer = buffer_from_torch_tensor(value)
                    bound_args.arguments[tensor_param] = view_buffer_as_tensor(
                        buffer,
                        restrict=restrict,
                        writable=writable
                    )

            # Use only keyword arguments to avoid duplicate parameter issues
            return func(**bound_args.arguments)

        return wrapper

    return decorator

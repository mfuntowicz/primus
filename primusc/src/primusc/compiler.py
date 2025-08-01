from pathlib import Path
from typing import Self

from llvmlite.binding import Target as LlvmTarget
from loguru import logger
from mlir.ir import Context as MlirContext, Module as MlirModule
from mlir.passmanager import PassManager


from .devices import Device
from .llvm import allocate_execution_engine, initialize_llvm


class Compiler:
    """
    Primus Compiler (aka. `primusc`)
    """
    def __init__(self, module: "MlirModule", context: "MlirContext"):
        self._module = module
        self._context = context

    @staticmethod
    def from_file(workdir: Path, module: str) -> "Self":
        if (module_file := (workdir / module).absolute()).exists():
            logger.debug(f"Parsing MLIR file: {module_file}")

            with open(module_file, "r") as f:
                module_def = f.read()
                context = MlirContext()
                module = MlirModule.parse(asm=module_def, context=context)
                return Compiler(module, context)
        else:
            raise IOError(f"{module_file} not found")

    def as_llvmir(self, device: Device) -> "MlirModule":
        """
        Convert the original module to the LLVM IR counterpart
        :param device: The target device to lower for
        :return:
        """
        pm = PassManager(context=self._context)
        logger.debug("Lowering down MLIR module to LLVM IR")

    def jit(self, *, device: Device, target: LlvmTarget | None = None):
        """
        Creates a Just-In-Time artifact to execute the underlying MLIR module
        :param device:
        :param target:
        :return:
        """
        initialize_llvm()

        # Allocate the LLVM JITer
        engine = allocate_execution_engine(target=target)

        # Lower down the current module to a llvm compatible representation
        llvm_ir = self.as_llvmir(device)

        # Register the new module to the JIT engine and compile down to in-memory executable
        engine.add_module(llvm_ir)
        engine.finalize_object()

    def compile(self, dest: Path, *, device: Device, target: LlvmTarget | None = None):
        """
        :param dest:
        :param device:
        :param target:
        :return:
        """
        logger.debug(f"Compiling MLIR module to {dest}")
        jitted = self.jit(device=device, target=target)
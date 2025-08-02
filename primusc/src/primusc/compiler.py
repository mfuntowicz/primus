import subprocess
from pathlib import Path
from typing import Self, TYPE_CHECKING

import llvmlite.binding
from llvmlite.binding import Target as LlvmTarget, parse_assembly as parse_llvm_assembly
from loguru import logger
from mlir.ir import Context as MlirContext, Module as MlirModule
from mlir.passmanager import PassManager
from mlir.extras.runtime.passes import  run_pipeline


from .devices import Device
from .llvm import allocate_execution_engine, initialize_llvm
from .passes import with_llvm_lowering_passes

if TYPE_CHECKING:
    from llvmlite.binding import ModuleRef


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

    @staticmethod
    def translate_to_llvmir(module: str) -> "MlirModule":
        # Run mlir-translate to convert to LLVM IR
        translate_cmd = ["mlir-translate", "--mlir-to-llvmir"]
        try:
            translate_result = subprocess.run(
                translate_cmd,
                input=module,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print("Error running mlir-translate:")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise

        return translate_result.stdout

    def as_llvm(self, device: Device) -> "MlirModule":
        """
        Convert the original module to the LLVM IR counterpart
        :param device: The target device to lower for
        :return:
        """
        logger.debug("Lowering down MLIR module to LLVM IR")

        # Clone the module to keep the original copy
        module = self._module.operation.clone()

        # Build-up the lowering pipeline
        pipeline = with_llvm_lowering_passes()

        # Execute the pipeline
        pm = PassManager.parse(str(pipeline), context=self._context)
        pm.enable_ir_printing(enable_debug_info=False, print_after_change=True)
        pm.enable_verifier(True)
        pm.run(module.operation)

        return module


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
        llvm_ir = self.as_llvm(device)
        llvm_asm = self.translate_to_llvmir(str(llvm_ir.operation).strip())
        llvm_module = parse_llvm_assembly(llvm_asm)
        llvm_module.verify()

        print(str(llvm_module))
        llvmir_module = self.translate_to_llvmir(str(llvm_module))

        # Register the new module to the JIT engine and compile down to in-memory executable
        engine.add_module(llvm_module)
        engine.finalize_object()
        engine.run_static_constructors()

    def compile(self, dest: Path, *, device: Device, target: LlvmTarget | None = None):
        """
        :param dest:
        :param device:
        :param target:
        :return:
        """
        logger.debug(f"Compiling MLIR module to {dest}")
        jitted = self.jit(device=device, target=target)

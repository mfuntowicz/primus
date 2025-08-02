from typing import TYPE_CHECKING

import llvmlite.binding as llvm
from loguru import logger

if TYPE_CHECKING:
    from llvmlite.binding import ExecutionEngine

def initialize_llvm():
    """

    :return:
    """
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


def allocate_execution_engine(target: llvm.Target | None = None) -> "ExecutionEngine":
    """

    :param target:
    :return:
    """
    target = target or llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")

    logger.debug(f"Allocating LLVM execution engine targeting {target.triple}")
    return llvm.create_mcjit_compiler(backing_mod, target_machine)
from mlir.passmanager import PassManager


def with_llvm_lowering_passes(manager: PassManager | None = None, verbose: bool = False) -> PassManager:
    manager = manager or PassManager()
    manager.enable_ir_printing(print_after_change=verbose)

    manager.add("canonicalize")
    manager.add("cse")
    manager.add("linalg-generalize-named-ops")
    manager.add("linalg-fuse-elementwise-ops")
    manager.add("convert-linalg-to-affine-loops")
    manager.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    manager.add("func.func(affine-loop-invariant-code-motion)")
    manager.add("affine-parallelize")
    manager.add("affine-super-vectorize")
    manager.add("affine-scalrep")
    manager.add("affine-loop-normalize")
    manager.add("lower-affine")
    manager.add("buffer-deallocation-pipeline")
    manager.add("convert-bufferization-to-memref")
    manager.add("canonicalize")
    manager.add("convert-arith-to-llvm")
    manager.add("convert-math-to-llvm")
    manager.add("inline")
    manager.add("reconcile-unrealized-casts")
    manager.add("canonicalize")
    return manager
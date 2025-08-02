from mlir.passmanager import PassManager
from mlir.extras.runtime.passes import Pipeline


def with_bufferization_passes(pipeline: Pipeline | None = None) -> Pipeline:
    pipeline = pipeline or Pipeline()

    pipeline \
        .one_shot_bufferize(allow_unknown_ops=True, unknown_type_conversion="identity-layout-map", allow_return_allocs_from_loops=True, bufferize_function_boundaries=True) \
        .buffer_results_to_out_params(True) \
        .canonicalize() \
        .cse()

    return pipeline

def with_affine_passes(pipeline: Pipeline | None = None) -> Pipeline:
    pipeline = pipeline or Pipeline()

    pipeline \
        .convert_tensor_to_linalg() \
        .convert_linalg_to_affine_loops() \
        .Func(
            Pipeline().affine_loop_invariant_code_motion() \
                .affine_loop_normalize() \
                .affine_simplify_structures() \
                .affine_parallelize() \
                .affine_super_vectorize() \
                .affine_scalrep() \
                .affine_loop_fusion()
                .lower_affine()
        ).lower_affine() \
        .canonicalize() \
        .cse()

    return pipeline


def with_linalg_lowering_passes(pipeline: Pipeline | None = None) -> Pipeline:
    pipeline = pipeline or Pipeline()

    pipeline \
        .convert_elementwise_to_linalg() \
        .linalg_inline_scalar_operands() \
        .linalg_generalize_named_ops() \
        .linalg_fuse_elementwise_ops() \
        .one_shot_bufferize(allow_unknown_ops=True, unknown_type_conversion="identity-layout-map", allow_return_allocs_from_loops=True, bufferize_function_boundaries=True) \
        .buffer_results_to_out_params(True) \
        .bufferize() \
        .convert_linalg_to_affine_loops() \
        .convert_linalg_to_loops() \
        .canonicalize() \
        .cse()

    return pipeline


def with_llvm_lowering_passes(pipeline: Pipeline | None = None) -> Pipeline:
    pipeline = pipeline or Pipeline()

    pipeline \
        .canonicalize() \
        .cse()

    # Linalg conversion
    pipeline = with_linalg_lowering_passes(pipeline)

    # Affine conversion
    pipeline = with_affine_passes(pipeline)

    # Bufferization
    pipeline.bufferize()

    # LLVM conversion
    pipeline \
        .inline() \
        .canonicalize() \
        .cse() \
        .lower_to_llvm(use_bare_ptr_memref_call_conv=False) \
        .strip_debuginfo()

    return pipeline
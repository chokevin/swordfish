"""Standalone vectorsum_v2 submission entrypoint.

The evaluator imports ``custom_kernel`` from this file and passes the generated
``(input_tensor, output_tensor)`` tuple. Keep this file self-contained: it should
not depend on the local swordfish package being installed in the evaluation
container.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 8192
PARTIAL_NUM_WARPS = 8
FINAL_NUM_WARPS = 16
PARTIAL_NUM_STAGES = 1

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - evaluator has Triton; local CPU import stays safe.
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _partial_sum_kernel(x_ptr, partials_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        partial = tl.sum(values, axis=0)
        tl.store(partials_ptr + pid, partial)

    @triton.jit
    def _final_sum_kernel(partials_ptr, out_ptr, n_partials: int, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_partials
        values = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        total = tl.sum(values, axis=0)
        tl.store(out_ptr, total)

else:
    _partial_sum_kernel = None
    _final_sum_kernel = None


_PARTIALS = None
_PARTIALS_DEVICE = None
_PARTIALS_N = 0
_N_PARTIALS = 0
_FINAL_BLOCK_SIZE = 0
_GRAPH = None
_GRAPH_X = None
_GRAPH_OUTPUT = None
_GRAPH_DATA = None
_GRAPH_PARTIALS = None
_GRAPH_N = 0
_GRAPH_REPLAY = None
_GRAPH_RESULT = None
_GRAPH_CAPTURE_WARMUP = 0


def _launch_sum(x, output, partials, n_elements: int, n_partials: int, final_block_size: int):
    _partial_sum_kernel[(n_partials,)](
        x,
        partials,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=PARTIAL_NUM_WARPS,
        num_stages=PARTIAL_NUM_STAGES,
    )
    _final_sum_kernel[(1,)](
        partials,
        output,
        n_partials,
        BLOCK_SIZE=final_block_size,
        num_warps=FINAL_NUM_WARPS,
    )


def _make_custom_kernel():
    last_x_obj = None
    last_output_obj = None
    graph_x_obj = None
    graph_obj = None
    replay_fn = None
    result_tensor = None
    partials_obj = None
    graph_output = None
    eager_partials = None
    eager_output = None
    eager_result = None
    eager_x_obj = None
    eager_device = None
    eager_n = 0
    eager_n_partials = 0
    eager_final_block = 0

    def custom_kernel(data):
        nonlocal eager_device, eager_final_block, eager_n, eager_n_partials
        nonlocal eager_output, eager_partials, eager_result, eager_x_obj
        nonlocal graph_obj, graph_output, graph_x_obj
        nonlocal last_output_obj, last_x_obj, partials_obj, replay_fn, result_tensor

        x, output = data
        if replay_fn is not None and graph_x_obj is x:
            replay_fn()
            return result_tensor

        if triton is None or _partial_sum_kernel is None or _final_sum_kernel is None:
            raise RuntimeError("custom_kernel requires Triton")

        if eager_partials is not None and eager_x_obj is x:
            if (
                x.device.type == "cuda"
                and hasattr(torch.cuda, "CUDAGraph")
                and last_x_obj is x
                and last_output_obj is output
            ):
                graph_output = torch.empty((1,), device=x.device, dtype=torch.float32)
                partials = torch.empty((eager_n_partials,), device=x.device, dtype=torch.float32)

                _launch_sum(x, graph_output, partials, eager_n, eager_n_partials, eager_final_block)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _launch_sum(
                        x, graph_output, partials, eager_n, eager_n_partials, eager_final_block
                    )

                replay_fn = graph.replay
                result = graph_output.reshape(-1)[0]
                graph_x_obj = x
                graph_obj = graph
                result_tensor = result
                partials_obj = partials
                for _ in range(_GRAPH_CAPTURE_WARMUP):
                    replay_fn()
                torch.cuda.synchronize()
                replay_fn()
                return result

            _launch_sum(
                x,
                eager_output,
                eager_partials,
                eager_n,
                eager_n_partials,
                eager_final_block,
            )
            last_x_obj = x
            last_output_obj = output
            return eager_result

        n_elements = x.numel()
        device = x.device.index
        if eager_partials is None or eager_n != n_elements or eager_device != device:
            eager_n = n_elements
            eager_device = device
            n_partials = triton.cdiv(n_elements, BLOCK_SIZE)
            eager_n_partials = n_partials
            eager_final_block = triton.next_power_of_2(n_partials)
            eager_partials = torch.empty((n_partials,), device=x.device, dtype=torch.float32)
            eager_output = torch.empty((1,), device=x.device, dtype=torch.float32)
            eager_result = eager_output.reshape(-1)[0]
            replay_fn = None
            graph_x_obj = None

        if (
            x.device.type == "cuda"
            and hasattr(torch.cuda, "CUDAGraph")
            and last_x_obj is x
            and last_output_obj is output
        ):
            graph_output = torch.empty((1,), device=x.device, dtype=torch.float32)
            partials = torch.empty((eager_n_partials,), device=x.device, dtype=torch.float32)

            _launch_sum(x, graph_output, partials, n_elements, eager_n_partials, eager_final_block)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _launch_sum(
                    x, graph_output, partials, n_elements, eager_n_partials, eager_final_block
                )

            replay_fn = graph.replay
            result = graph_output.reshape(-1)[0]
            graph_x_obj = x
            graph_obj = graph
            result_tensor = result
            partials_obj = partials
            for _ in range(_GRAPH_CAPTURE_WARMUP):
                replay_fn()
            torch.cuda.synchronize()
            replay_fn()
            return result

        _launch_sum(
            x,
            eager_output,
            eager_partials,
            n_elements,
            eager_n_partials,
            eager_final_block,
        )
        eager_x_obj = x
        last_x_obj = x
        last_output_obj = output
        return eager_result

    return custom_kernel


custom_kernel = _make_custom_kernel()

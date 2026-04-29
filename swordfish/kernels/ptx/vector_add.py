"""Handwritten raw PTX vector-add spike."""

from __future__ import annotations

import textwrap

import torch


PTX_VECTOR_ADD_F32 = textwrap.dedent(
    r"""
    .version 8.0
    .target sm_80
    .address_size 64

    .visible .entry vector_add_f32(
        .param .u64 a_ptr,
        .param .u64 b_ptr,
        .param .u64 out_ptr,
        .param .u32 n
    )
    {
        .reg .pred p;
        .reg .b32 r_tid, r_ctaid, r_ntid, r_idx, r_n;
        .reg .b64 rd_a, rd_b, rd_out, rd_offset;
        .reg .f32 f_a, f_b, f_out;

        ld.param.u64 rd_a, [a_ptr];
        ld.param.u64 rd_b, [b_ptr];
        ld.param.u64 rd_out, [out_ptr];
        ld.param.u32 r_n, [n];

        mov.u32 r_tid, %tid.x;
        mov.u32 r_ctaid, %ctaid.x;
        mov.u32 r_ntid, %ntid.x;
        mad.lo.u32 r_idx, r_ctaid, r_ntid, r_tid;
        setp.ge.u32 p, r_idx, r_n;
        @p bra done;

        mul.wide.u32 rd_offset, r_idx, 4;
        add.u64 rd_a, rd_a, rd_offset;
        add.u64 rd_b, rd_b, rd_offset;
        add.u64 rd_out, rd_out, rd_offset;

        ld.global.f32 f_a, [rd_a];
        ld.global.f32 f_b, [rd_b];
        add.rn.f32 f_out, f_a, f_b;
        st.global.f32 [rd_out], f_out;

    done:
        ret;
    }
    """
).strip()


def raw_ptx_blocker() -> str:
    return (
        "Raw PTX loader is not implemented yet. Use CUDA driver bindings on a "
        "Linux CUDA host to load PTX_VECTOR_ADD_F32 and launch vector_add_f32; "
        "do not fall back to torch.add under a raw-PTX benchmark label."
    )


def torch_vector_add_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        return a + b
    return torch.add(a, b, out=out)


def ptx_vector_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if a.device.type != "cuda" or b.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("raw PTX vector add requires CUDA tensors")
    raise RuntimeError(raw_ptx_blocker())

"""In-place rotate-half RoPE. One kernel for the query or the key, no intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    stride_xb,
    stride_xh,
    stride_xs,
    stride_xd,
    stride_cb,
    stride_cs,
    stride_cd,
    stride_sb,
    stride_ss,
    stride_sd,
    heads,
    seq,
    half,
    BLOCK: tl.constexpr,
    OUT: tl.constexpr,
):
    """In-place rotate-half RoPE for one [batch, heads, seq, dim] tensor.

    cos and sin are [batch, seq, dim] and broadcast across heads. The pair
    layout matches rotate_half: the two halves of the head dim swap with a
    sign on the first half.
    """
    row = tl.program_id(0)
    token = row % seq
    bh = row // seq
    head = bh % heads
    batch = bh // heads
    offs = tl.arange(0, BLOCK)
    mask = offs < half

    x_row = x_ptr + batch * stride_xb + head * stride_xh + token * stride_xs
    x1 = tl.load(x_row + offs * stride_xd, mask=mask).to(tl.float32)
    x2 = tl.load(x_row + (offs + half) * stride_xd, mask=mask).to(tl.float32)

    cos_row = cos_ptr + batch * stride_cb + token * stride_cs
    sin_row = sin_ptr + batch * stride_sb + token * stride_ss
    c1 = tl.load(cos_row + offs * stride_cd, mask=mask).to(tl.float32)
    c2 = tl.load(cos_row + (offs + half) * stride_cd, mask=mask).to(tl.float32)
    s1 = tl.load(sin_row + offs * stride_sd, mask=mask).to(tl.float32)
    s2 = tl.load(sin_row + (offs + half) * stride_sd, mask=mask).to(tl.float32)

    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2
    tl.store(x_row + offs * stride_xd, out1.to(OUT), mask=mask)
    tl.store(x_row + (offs + half) * stride_xd, out2.to(OUT), mask=mask)


def _broadcast_stride(size, stride, expected):
    if size == expected:
        return stride
    if size == 1:
        return 0
    raise RuntimeError(f"cannot broadcast size {size} onto {expected}")


def rope_inplace(x, cos, sin):
    """Apply rotate-half RoPE to `x` in place. `x` is [batch, heads, seq, dim]."""
    if x.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise RuntimeError(f"rope shapes x={tuple(x.shape)} cos={tuple(cos.shape)}")
    batch, heads, seq, dim = x.shape
    half = dim // 2
    if dim != half * 2 or half > 128 or half == 0:
        raise RuntimeError(f"rope head dim {dim} is outside the fused kernel")
    out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32
    rope_kernel[(batch * heads * seq,)](
        x,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        _broadcast_stride(cos.shape[0], cos.stride(0), batch),
        _broadcast_stride(cos.shape[1], cos.stride(1), seq),
        cos.stride(2),
        _broadcast_stride(sin.shape[0], sin.stride(0), batch),
        _broadcast_stride(sin.shape[1], sin.stride(1), seq),
        sin.stride(2),
        heads,
        seq,
        half,
        BLOCK=128,
        OUT=out_dtype,
    )
    return x

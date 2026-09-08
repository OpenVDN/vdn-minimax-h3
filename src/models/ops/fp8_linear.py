"""fp8 for the DiT's big Linears. OPT-IN, NEVER A DEFAULT.

WHY IT IS HERE. Once the pointwise work is fused, a block's time is dominated by GEMMs
and attention already running near tensor-core peak; fp8 e4m3 is the remaining lever,
roughly doubling each of the block's GEMMs.

WHY IT IS NOT THE DEFAULT. fp8 does not degrade the output -- it CHANGES it. A single
step's velocity prediction stays close to bf16's (cosine ~0.998), but a denoising
trajectory is chaotic: perturb it early and it settles into a different mode, so the
final latents and the decoded clip are a different (not worse) sample of the same
prompt. fp8 IS NOT A DROP-IN: anything that depends on reproducing a previous render
(a regression suite, a checkpoint comparison, a user re-rolling a seed) stops working.

WHAT IS QUANTISED. Only Linears at or above `min_width` on BOTH sides -- the qkv
projections, to_out, the two feed-forward GEMMs and to_out_linear. Everything narrow
stays in bf16 on purpose: the output gates are low-rank, beta_proj is hidden -> heads,
FrameKDAAlpha is a deliberate fp32 island, and adaln_proj reads a tiny table. None of
them is a meaningful share of the time and all of them are places precision matters
more than throughput.

THE QUANTISER IS A TRITON KERNEL, one program per row: absmax, then the scaled cast,
read once from HBM (the second sweep over the row comes back out of L2); the compiled
torch spelling was several times slower than the bandwidth floor. Two callers avoid a
quantisation altogether: `HybridAttention._qkv` quantises x once for the three
projections (`forward_quantized`), and the inference feed-forward fuses the SwiGLU
activation with the quantisation of its output (`swiglu_quantize`), so the bf16
intermediate is never written.

WHERE THE ERROR COMES FROM. The WEIGHTS, not the activations: rowwise activation scaling
is no more accurate than per-tensor, while fp8 weights alone account for most of the
error. So the lever for accuracy, if anyone wants one, is finer weight granularity
(per-block) or keeping outlier channels in bf16 -- not a better activation scale.
Rowwise activation scaling is used on sm90 anyway because it is free there.

`skip_end_blocks` leaves the first and last N blocks in bf16; it keeps most of the
speedup and removes a disproportionate share of the error.

    from src.models.ops.fp8_linear import convert_linear_to_fp8, revert_fp8   # infer.py
    handle = convert_linear_to_fp8(model)      # after any LoRA merge, before the render
    ...
    revert_fp8(handle)                         # exact: the original modules come back

SM100 USES PER-TENSOR SCALES ON BOTH SIDES, and this is a dispatch fact, not a numerics
preference. torch 2.13 routes rowwise-scaled `_scaled_mm` on sm100 to a generic CUTLASS
kernel that is barely faster than bf16 at these shapes, while per-tensor x per-tensor
dispatches to a cuBLAS kernel at the expected ~2x. Mixed scalar-x-vector scale
combinations are refused by this torch build, so the fast path needs per-tensor on BOTH
sides. The accuracy cost is negligible on the real weights, because e4m3's dynamic range
covers their channel outliers without underflow. sm90 keeps rowwise: it is free there.
The granularity follows the capability alone -- it is not a knob.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
MIN_WIDTH = 4096
SKIP_END_BLOCKS = 4


@triton.jit
def _quantize_rows_kernel(X, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)


@triton.jit
def _swiglu_quantize_kernel(H, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(a * g * tl.sigmoid(g)))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(a * g * tl.sigmoid(g) / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)


def _rows(x):
    if not x.is_cuda:
        raise ValueError("the fp8 quantiser is a Triton kernel; the input must be on CUDA")
    if x.dim() != 2:
        raise ValueError(f"expected [M, K], got {tuple(x.shape)}")
    return x.contiguous()


def quantize_rows(x):
    """bf16 [M, K] -> (fp8 [M, K], fp32 scale [M, 1]). One absmax per row."""
    x = _rows(x)
    M, K = x.shape
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    _quantize_rows_kernel[(M,)](x, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=1024, num_warps=4)
    return y, scale


def swiglu_quantize(h):
    """[M, 2K] projection output -> quantize_rows(a * silu(gate)) with a, gate = h.chunk(2),
    without writing the bf16 activation."""
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    y = torch.empty(M, K, device=h.device, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=h.device, dtype=torch.float32)
    _swiglu_quantize_kernel[(M,)](h, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=2048, num_warps=16)
    return y, scale


def quantize_rows_reference(x):
    """The eager spelling of `quantize_rows`, for the tests."""
    scale = (x.float().abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    return (x.float() / scale).to(FP8_DTYPE), scale


_PER_TENSOR = None


def per_tensor_gemm():
    """sm100 uses per-tensor scales (see the header); sm90 keeps rowwise."""
    global _PER_TENSOR
    if _PER_TENSOR is None:
        _PER_TENSOR = (torch.cuda.is_available()
                       and torch.cuda.get_device_capability(0)[0] >= 10)
    return _PER_TENSOR


@triton.jit
def _absmax_kernel(X, OUT, N, BLOCK: tl.constexpr):
    """Partial absmax + one atomic per program: a full reduction in ONE read pass,
    where eager `x.abs().amax()` materialises the abs (a full-size write at the qkv
    shape) before it reduces."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.atomic_max(OUT, tl.max(tl.abs(x), axis=0))


@triton.jit
def _cast_scaled_kernel(X, Y, S, N, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    scale = tl.load(S)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_rowmax_kernel(H, Y, RM, K, BLOCK_K: tl.constexpr):
    """a * silu(gate) per row into bf16 Y, plus the row's absmax -- the per-tensor
    spelling of `_swiglu_quantize_kernel`, split so the global scale can be taken
    before the cast."""
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask = cols < K
        a = tl.load(H + row * 2 * K + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=mask, other=0.0).to(tl.float32)
        act = (a * g * tl.sigmoid(g)).to(Y.dtype.element_ty)
        # amax of the ROUNDED value: the bf16 intermediate is the tensor being
        # quantised, so its own absmax is the scale.
        amax = tl.maximum(amax, tl.abs(act.to(tl.float32)))
        tl.store(Y + row * K + cols, act, mask=mask)
    tl.store(RM + row, tl.max(amax, axis=0))


def quantize_tensor(x):
    """bf16 [M, K] -> (fp8 [M, K], fp32 scale [1, 1]). One absmax for the whole tensor;
    the scale never leaves the device (no sync)."""
    x = _rows(x)
    n = x.numel()
    amax = torch.zeros(1, device=x.device, dtype=torch.float32)
    _absmax_kernel[(triton.cdiv(n, 8192),)](x.view(-1), amax, n, BLOCK=8192,
                                            num_warps=8)
    scale = (amax / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](x.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192,
                                                 num_warps=8)
    return y, scale


def quantize_tensor_reference(x):
    """The eager spelling of `quantize_tensor`, for the tests."""
    scale = (x.float().abs().amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    return (x.float() / scale).to(FP8_DTYPE), scale


def swiglu_quantize_tensor(h):
    """[M, 2K] projection output -> per-tensor quantised a * silu(gate). One extra
    bf16 write against the fused rowwise kernel; the fast GEMM it unlocks on sm100
    repays it many times over."""
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    act = torch.empty(M, K, device=h.device, dtype=h.dtype)
    rowmax = torch.empty(M, device=h.device, dtype=torch.float32)
    _swiglu_rowmax_kernel[(M,)](h, act, rowmax, K, BLOCK_K=2048, num_warps=16)
    scale = (rowmax.amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(act, dtype=FP8_DTYPE)
    n = act.numel()
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](act.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192,
                                                 num_warps=8)
    return y, scale


def quantize_activation(x):
    """What the GEMM callers use: the scale granularity the current card's fast
    `_scaled_mm` path accepts."""
    return quantize_tensor(x) if per_tensor_gemm() else quantize_rows(x)


def swiglu_quantize_activation(h):
    return swiglu_quantize_tensor(h) if per_tensor_gemm() else swiglu_quantize(h)


class Fp8Linear(nn.Module):
    """A bias-optional Linear whose GEMM runs in fp8 e4m3.

    The weight is quantised ONCE, per output channel, at construction; the activation is
    quantised per row on every call, or by the caller (`forward_quantized`) when one
    activation feeds several of these. The original bf16 module is kept, untouched, so
    `revert_fp8` can put it back.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight
        if per_tensor_gemm():
            # sm100: both scales must be scalar for the fast cuBLAS path; on the real
            # weights this costs nothing measurable over per-channel (see the header).
            scale = (weight.abs().amax().float() / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8",
                                 (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, 1).contiguous())
        else:
            scale = (weight.abs().amax(dim=1, keepdim=True).float()
                     / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8",
                                 (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, -1).contiguous())
        self._bias = linear.bias
        self.original = linear

    @property
    def bias(self):
        return self.original.bias if self.original is not None else self._bias

    def forward_quantized(self, x_fp8, x_scale, out_dtype=torch.bfloat16):
        """[M, K] fp8 rows and their scales ([M, 1] rowwise / [1, 1] per-tensor) -> [M, N]."""
        out = torch._scaled_mm(x_fp8, self.weight_fp8.t(),
                               scale_a=x_scale, scale_b=self.weight_scale,
                               out_dtype=out_dtype, use_fast_accum=True)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        out = self.forward_quantized(*quantize_activation(rows), out_dtype=rows.dtype)
        return out.reshape(*shape[:-1], -1)


def _blocks_to_skip(model, skip_end_blocks):
    """ids of every module inside the first and last N transformer blocks."""
    blocks = getattr(model, "transformer_blocks", None)
    if not blocks or skip_end_blocks <= 0:
        return set()
    keep = set()
    for index, block in enumerate(blocks):
        if index < skip_end_blocks or index >= len(blocks) - skip_end_blocks:
            keep.update(id(m) for m in block.modules())
    return keep


def convert_linear_to_fp8(model, min_width=MIN_WIDTH, skip_end_blocks=SKIP_END_BLOCKS,
                          keep_original=True):
    """Swap the wide Linears for fp8 ones. Returns a handle to pass to `revert_fp8`.

    Call it AFTER any LoRA merge -- the merger writes into `Linear.weight`, and this
    reads that weight once to build the fp8 copy. Converting first would quantise the
    unmerged weights and then silently ignore the merge.
    """
    skip = _blocks_to_skip(model, skip_end_blocks)
    swapped = []
    # Keep only containers, not a materialized list of every original Linear.
    # Otherwise compact mode cannot free replaced BF16 weights until the full walk ends.
    parents = [m for m in model.modules()
               if any(isinstance(c, nn.Linear) for c in m.children())]
    for parent in parents:
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child.in_features < min_width or child.out_features < min_width:
                continue
            if id(child) in skip:
                continue
            target_device = child.weight.device
            if not keep_original and target_device.type == "cuda":
                # Free the BF16 layer before allocating its FP8 replacement. This keeps
                # peak memory below the resident hybrid model on 72 GB cards.
                child = child.to("cpu")
                torch.cuda.empty_cache()
                replacement = Fp8Linear(child)
                replacement.original = None
                replacement = replacement.to(target_device)
            else:
                replacement = Fp8Linear(child).to(target_device)
                if not keep_original:
                    replacement.original = None
            setattr(parent, name, replacement)
            swapped.append((parent, name, child if keep_original else None))
    return swapped


def revert_fp8(handle):
    """Put the original Linears back. Exact -- they were never modified."""
    for parent, name, child in handle:
        setattr(parent, name, child)
    return len(handle)

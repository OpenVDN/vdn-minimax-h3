"""flex_attention with the FLASH/CuteDSL backend, the BlockMask builder and
its caches, and the process-global downgrade latch. The latch (_FLEX_CACHE["infer_disabled"])
is PERMANENT for the process by design -- which is exactly why mask geometry is one
process, one geometry (see infer entrypoint).
"""
import collections
import functools
import os
import warnings

import torch
import torch._dynamo.config
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from src.checkpoints.key_mapping import ANCHOR_FRAME_MODES
from src.models.sequence_layout import SequenceLayout
from src.models.softmax_attention.window import window_bounds  # noqa: F401  (re-export for callers)


_FLEX_CACHE = {}                 # the compiled flex_attention, one entry, never evicted

# BlockMasks, bounded. Each one is ~10 MiB of GPU tensors at H3 scale, the same
# regardless of radius — kv_indices is allocated at max width. The key includes seq_len,
# which varies with the caption, AND the window bounds: unbounded, 100 distinct prompt
# lengths would park ~1 GiB on the GPU permanently. A run trains one fixed radius, so
# the working set is just the seq_len buckets the dataset spans; 64 caps it at ~650 MiB.
_MASK_CACHE = collections.OrderedDict()
MAX_CACHED_MASKS = 64



def _flex_attention_fn(inference=False):
    """torch.compile'd flex_attention. Two variants, and the difference is 1.6x.

    TRAINING (inference=False) — dynamic=True, Triton, ONE graph for every sequence
    length. dynamic=False is a landmine here. The packed length varies with the caption,
    so a static compile makes a fresh cache entry per length; torch._dynamo's
    recompile_limit is 8, and on the 9th distinct length it stops compiling and runs
    flex_attention EAGER, which materialises the full score matrix (56 x ~104k^2 x 2 B).
    A short benchmark cannot catch this — 3-4 steps see at most 4 distinct lengths.

    `fail_on_recompile_limit_hit` matters most, because the degradation is silent by
    construction: dynamo logs a warning and keeps running, and what surfaces is an OOM
    with no obvious connection to shapes. As an exception, an unanticipated shape stops
    the run instead of quietly falling off the fast path.

    INFERENCE (inference=True) — dynamic=False and CuteDSL/FlashAttention-4. At the
    real shape and mask this is ~1.6x faster per layer than the Triton path: the
    block-sparse FLASH kernel realises the sparsity almost perfectly, so its throughput
    approaches the dense FA4 kernel's.

    Why the two cannot be one. FLASH refuses a BlockMask under a dynamic compile:

        NYI: score_mod or mask_mod captures a dynamic scalar (SymInt/SymFloat). The
        FLASH backend cannot inline symbolic values into the CuteDSL template.

    and the symbols are inside the BlockMask itself (its Q_LEN, KV_LEN, block sizes), so
    rewriting our mask_mod's captures as device tensors does not help. dynamic=False is
    the only door, and it is a door only inference can walk through: one render process
    renders one prompt, so seq_len is fixed for its whole life, while training brings a
    new caption every batch. The restriction is narrower than "FLASH needs static
    shapes" — the DENSE FA4 teacher runs under a dynamic compile, because with no mask
    there is no template to inline into.

    The static variant keeps the recompile limit that guards the dynamic one, so a
    process that does see 9 distinct lengths raises rather than degrading. Callers
    downgrade on that (see window_softmax_flex).
    """
    key = "infer" if inference else "train"
    if key not in _FLEX_CACHE:
        torch._dynamo.config.fail_on_recompile_limit_hit = True

        # ...and raise the limit itself, which is a DIFFERENT knob and does not weaken
        # the one above. `recompile_limit` is per code object and defaults to 8; the
        # inference path compiles a dozen small helpers (the epilogues, the feature
        # kernels, the gather) whose bodies each specialise on a couple of flags, and a
        # process that exercises more than one configuration -- a test sweep, an
        # ablation driver -- exhausts 8 on a helper that has nothing to do with
        # attention. What protects the window kernel is the HARD FAILURE, not the
        # number: at 8 or at 32, an unanticipated sequence length stops the run instead
        # of silently falling back to eager flex_attention. 32 only buys tolerance for
        # more distinct caption lengths before that happens.
        torch._dynamo.config.recompile_limit = max(
            32, torch._dynamo.config.recompile_limit)
        if inference:
            _FLEX_CACHE[key] = torch.compile(
                functools.partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
                dynamic=False)
        else:
            _FLEX_CACHE[key] = torch.compile(flex_attention, dynamic=True)
    return _FLEX_CACHE[key]


def build_window_block_mask(layout: SequenceLayout, bounds, device, block_size=None,
                            anchor_frames="none"):
    """BlockMask for: video<->video pairs restricted to the per-frame window, every pair
    involving a global (text/audio) token dense. Cached per (layout, bounds, anchors).

    anchor_frames: how frames 0 and F-1 are dense -- "columns" (every video query sees
    all of both frames), "rows" (those two frames' queries see the whole sequence),
    "both", or "none". Under "both" the linear branch drops the two frames from its
    input (BidirectionalLinearBranch.forward's skip_ends) and softmax and branch stay an
    exact partition. The two anchor frames add a few points of mask density."""
    if anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {ANCHOR_FRAME_MODES}")
    if block_size is None:
        # FA4's CuteDSL block-sparse template is tile-shaped per arch. SM90: both sides
        # 128 ("sparse_block_size[1] must be a multiple of tile_n=128"). SM100: the
        # tiles are ASYMMETRIC -- a 128 mask is refused on the Q side ("block size 128,
        # which must be a multiple of 256") and a 256 mask on the KV side
        # ("sparse_block_size[1]=128 to match tile_n") -- so the mask must be built at
        # (Q=256, KV=128). The coarser Q granularity rounds the window up a little, but
        # that tax is far below the Triton fallback the process is otherwise latched
        # into.
        block_size = 128
        resolved = torch.device(device) if not isinstance(device, torch.device) else device
        if resolved.type == "cuda" and torch.cuda.get_device_capability(resolved)[0] >= 10:
            block_size = (256, 128)
    key = (layout.seq_len, layout.video_start, layout.num_frames,
           layout.tokens_per_frame, tuple(bounds), block_size, str(device),
           anchor_frames)
    if key in _MASK_CACHE:
        _MASK_CACHE.move_to_end(key)
        return _MASK_CACHE[key]

    video_start, video_end = layout.video_start, layout.video_end
    tokens_per_frame, num_frames = layout.tokens_per_frame, layout.num_frames
    window_lo = torch.tensor([lo for lo, _ in bounds], device=device)
    window_hi = torch.tensor([hi for _, hi in bounds], device=device)

    def mask_mod(batch, head, q_idx, kv_idx):
        query_is_video = (q_idx >= video_start) & (q_idx < video_end)
        key_is_video = (kv_idx >= video_start) & (kv_idx < video_end)

        # frame index of each row; the query side is clamped because mask_mod is also
        # evaluated on global rows, where the division would run out of range
        query_frame = torch.clamp(
            torch.div(q_idx - video_start, tokens_per_frame, rounding_mode="floor"),
            0, num_frames - 1)
        key_frame = torch.div(kv_idx - video_start, tokens_per_frame,
                              rounding_mode="floor")
        inside_window = ((key_frame >= window_lo[query_frame])
                         & (key_frame <= window_hi[query_frame]))
        is_anchor = lambda f: (f == 0) | (f == num_frames - 1)
        if anchor_frames in ("columns", "both"):
            inside_window = inside_window | is_anchor(key_frame)        # everyone sees them
        if anchor_frames in ("rows", "both"):
            inside_window = inside_window | is_anchor(query_frame)      # they see everyone
        # keep the pair unless BOTH sides are video and the key falls outside the window
        return (~(query_is_video & key_is_video)) | inside_window

    mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=layout.seq_len,
                             KV_LEN=layout.seq_len, device=device,
                             BLOCK_SIZE=block_size, _compile=True)
    _MASK_CACHE[key] = mask
    while len(_MASK_CACHE) > MAX_CACHED_MASKS:
        _MASK_CACHE.popitem(last=False)
    return mask


_FLEX_INFER_WARNED = False


def _warn_inference_flex():
    """One warning per process when INFERENCE lands on the flex/BlockMask window on
    sm100 -- not the preferred kernel there. Names whichever piece is missing: the
    decomposition (faster than even CLC-scheduled flex), FA_CLC=1 (the static schedule
    is slower), and the flash-attn gate patch (without it FA_CLC=1 is silently stripped
    for block-sparse calls). sm90 stays quiet: flex IS the right kernel there. With
    everything configured correctly there is no warning at all."""
    global _FLEX_INFER_WARNED
    if _FLEX_INFER_WARNED:
        return
    _FLEX_INFER_WARNED = True
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10:
            return
    except Exception:
        return
    msgs = ["kernels.softmax_backend is flex here; auto picks the decomposition, which "
            "runs this window faster than even CLC-scheduled flex"]
    if os.environ.get("FA_CLC") != "1":
        msgs.append("FA_CLC=1 is not set -- flex is running the STATIC block-sparse "
                    "schedule, which is noticeably slower on sm100")
    else:
        patched = True
        try:
            import flash_attn.cute.interface as _fa_iface
            with open(_fa_iface.__file__) as fh:
                src_text = fh.read()
            i = src_text.find("is_dense_noncausal =")
            # the patched assignment spans two physical lines, so search a window
            # after the assignment rather than line-by-line
            patched = i < 0 or "use_block_sparsity" in src_text[i:i + 220]
        except Exception:
            pass  # cannot inspect the install: better silent than crying wolf
        if not patched:
            msgs.append("FA_CLC=1 is set but the installed flash-attn STRIPS CLC for "
                        "block-sparse calls -- patch flash_attn/cute/interface.py so "
                        "is_dense_noncausal also excludes use_block_sparsity")
    if msgs:
        warnings.warn(
            "sm100 INFERENCE is on the flex/BlockMask window softmax: "
            + "; ".join(msgs), RuntimeWarning, stacklevel=3)


def window_softmax_flex(query, key, value, block_mask, scale, head_chunk=None,
                         inference=False):
    """query/key/value: [total, H, d] -> [total, H, d] via block-sparse FlexAttention.

    NO REPACK. flex wants [B, H, T, d] and both its backends accept it as a strided
    VIEW of the [T, H, d] the projections produce (d contiguous, T stride H*d), at the
    same speed and with bitwise-identical output. That removes three ~1.4 GiB
    transposing copies per layer (q, k, v) plus one on the way out: the output inherits
    q's strides, so `.transpose` back to [T, H, d] is a view and the gate/to_out
    downstream read it contiguously.

    Heads normally go in ONE call. They are only split when a single call would index
    past int32: inductor's Triton templates address the q/k/v tensors with 32-bit
    offsets, so a call is safe while total * head_dim * heads_per_call <= 2^31 - 1.
    That is a computed bound, not a probe — probing group sizes both recompiles
    flex_attention per size (straight into the recompile limit) and is strictly slower.
    At H3 scale (~104k x 128 x 56 = 7.5e8) the limit is far away, so one call.
    `head_chunk` overrides it for the one case the bound cannot see — running out of
    memory on a smaller card.

    `inference` selects the FLASH/static variant (see _flex_attention_fn) and DOWNGRADES
    permanently if it does not take. Downgrading rather than raising, because everything
    that can go wrong here is a property of the machine or the workload rather than of
    this call — no flash_attn.cute installed, a 9th distinct sequence length, a card
    whose CuteDSL template does not build — and the fallback is the Triton kernel,
    differing by a few 1e-3 relative (bf16 reduction order). A render that is ~1.6x
    slower on that leg beats a render that died. It says so once, loudly, because the
    only bad outcome here is nobody noticing that the fast path was never taken.
    """
    if inference:
        _warn_inference_flex()
    num_heads = value.shape[1]
    if head_chunk is None:
        budget = (2 ** 31 - 1) // (value.shape[0] * value.shape[2])
        head_chunk = max(1, min(num_heads, budget))
    use_flash = inference and not _FLEX_CACHE.get("infer_disabled")
    flex = _flex_attention_fn(inference=use_flash)

    outputs = []
    for first_head in range(0, num_heads, head_chunk):
        heads = slice(first_head, min(first_head + head_chunk, num_heads))
        q_g, k_g, v_g = (t[:, heads].unsqueeze(0).transpose(1, 2)      # [1, Hc, T, d] view
                         for t in (query, key, value))
        try:
            out_g = flex(q_g, k_g, v_g, block_mask=block_mask, scale=scale)
        except Exception as exc:
            if not use_flash:
                raise
            _FLEX_CACHE["infer_disabled"] = True
            print("hybrid_attention: the FLASH/static window kernel did not take on this "
                  f"machine; falling back to Triton for the rest of the process (~1.6x "
                  f"slower on the window). Reason: {type(exc).__name__}: "
                  f"{str(exc).strip().splitlines()[0][:160]}", flush=True)
            use_flash = False
            flex = _flex_attention_fn(inference=False)
            out_g = flex(q_g, k_g, v_g, block_mask=block_mask, scale=scale)
        outputs.append(out_g.squeeze(0).transpose(0, 1))                 # [T, Hc, d]

    # the common case is one group, and then this is a view, not a copy
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

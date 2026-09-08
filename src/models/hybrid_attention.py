"""HybridAttention: the orchestrator and nothing else.

Drop-in replacement for the diffusers MiniMaxH3Attention inside a DiT block. It owns
the shared QKV, calls the softmax window and the linear branch, and fuses:

    softmax_output = orig.to_out( softmax_gate(x) * window_softmax(q, k, v) )
    linear_readout = output_gate(x) * RMSNorm( linear_attention(q_raw, k_raw, v_raw) )
    output         = softmax_output + to_out_linear(linear_readout)   # video rows only

to_out_linear takes nn.Linear's default init. Init-only: any checkpoint overwrites this
weight on load.
"""
import torch
from torch import nn

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

from src.models.attention_gates import OutputGate
from src.models.sequence_layout import SequenceLayout
from src.models.linear_attention import BidirectionalLinearBranch
from src.models.softmax_attention import (apply_softmax_gate, build_window_block_mask,
                                          window_bounds, window_softmax_flex,
                                          window_softmax_reference)
from src.models.softmax_attention.decomposed import window_softmax_decomposed
from src.models.softmax_attention.kernels import _qk_prep
from src.models.ops.fp8_linear import Fp8Linear, quantize_activation
from src.checkpoints.key_mapping import ANCHOR_FRAME_MODES


class HybridAttention(nn.Module):
    """Drop-in replacement for the diffusers `MiniMaxH3Attention` inside a DiT block:
    same forward signature `(hidden_states, rotary_emb, attention_mask)`, same output
    contract (pre-residual attention output, batch-first). Batch must be 1 — H3 packs
    one request per document. Set `.layout` (SequenceLayout) before each forward and
    `.radius` to choose the frame window; the operating range is r <= 5.
    `radius >= num_frames-1` (or layout=None) makes the softmax branch equal to full
    attention.

    `.teacher_mode = True` makes forward a pure pass-through to the original attention —
    Stage A runs the trunk that way (teacher trajectory, no_grad) and computes the
    student output in a forward hook, keeping alignment a training strategy rather than
    a property of the module (see train_stage_a1.py's AlignHook)."""

    def __init__(self, orig_attn, hidden_size, delta_rule="sana_scaled", radius=4,
                 chunk=0, enable_softmax_gate=True, linear_head_dim=None,
                 softmax_impl="flex", anchor_frames="none", enable_text_state=False,
                 bridge="alpha", a_fp32=True, short_conv=()):
        """Keyword names mirror the ModelSpec transform config one-to-one
        (`softmax_attention.{radius,chunk}`, `anchor_frames`,
        `linear_attention.{delta_rule,linear_head_dim,bridge,a_fp32,enable_text_state}`,
        `linear_attention.short_conv.targets` as `short_conv`, `enable_softmax_gate`);
        `softmax_impl` is the one runtime knob ("flex" | "decomposed" | "ref", set by
        hybrid_transform.set_softmax_backend) and never enters a spec.

        anchor_frames: how frames 0 and F-1 sit in the softmax mask -- "columns" (every
        video query sees all of both frames), "rows" (those two frames' queries see the
        whole sequence), "both", or "none". A cross-branch fact: under "both" the two
        frames are exact softmax in both directions, so the linear branch drops them from
        its input (skip_ends) and the softmax/linear partition stays exact. Under
        "columns" or "rows" alone the partition would not be exact, so the branch keeps
        covering them."""
        super().__init__()
        self.orig = orig_attn                                    # original weights, reused
        self.num_heads = orig_attn.heads
        self.head_dim = orig_attn.head_dim
        self.radius = radius
        self.chunk = chunk           # 0 = frame window ("r<n>"); K = K-frame chunks ("c<n>")
        self.softmax_impl = softmax_impl                         # "flex" | "decomposed" | "ref"

        if anchor_frames not in ANCHOR_FRAME_MODES:
            raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of "
                             f"{ANCHOR_FRAME_MODES}")
        self.anchor_frames = anchor_frames

        # Seed both linear-branch scans with the prompt (see BidirectionalLinearBranch.
        # forward). Needs a layout that carries the text rows — layout_from_indices
        # must have been given text_indices.
        self.enable_text_state = enable_text_state
        self.linear_attention_enabled = True    # False = window-only ablation (pure sparse attention)
        self.layout: SequenceLayout = None
        self.teacher_mode = False

        # Two inference-only levels, separated so a benchmark can attribute the speedup.
        # `hybrid_inference_mode` covers only work intrinsic to the hybrid algorithm:
        # the FLASH window kernel and the tuned linear far branch. `inference_mode`
        # additionally enables general fusions such as QK-norm + RoPE. The production
        # setter turns both on; defaults stay slow-but-correct for training and forgotten
        # switches.
        self.hybrid_inference_mode = False
        self.inference_mode = False
        d_linear = linear_head_dim or self.head_dim
        self.linear_attention = BidirectionalLinearBranch(
            hidden_size, self.num_heads, d_linear, delta_rule=delta_rule, bridge=bridge,
            a_fp32=a_fp32, short_conv=short_conv)
        # The linear branch's own output projection, torch-default init: the readout it
        # consumes is SiLU'd and RMS-normalised, so there is no reason to seed it from
        # orig.to_out. Stage A1 trains it from scratch.
        self.to_out_linear = nn.Linear(self.num_heads * d_linear, hidden_size,
                                       bias=False)
        self.enable_softmax_gate = enable_softmax_gate
        if enable_softmax_gate:
            # per-head, direct (not low rank). 0.99 keeps the softmax branch at the
            # teacher on step 0.
            self.softmax_gate = OutputGate(hidden_size, self.num_heads, init_value=0.99)

    def _qkv(self, x, rotary_emb):
        """Replicates MiniMaxH3AttnProcessor up to (and excluding) the attention call,
        via the original module's submodules — projections (LoRA-wrapped ones apply),
        QK-norm and RoPE included. Also returns the raw (pre-QK-norm, pre-RoPE) q/k/v
        for the shared-QKV linear branch. x: [total, hidden]; everything [total, H, d].
        Under fp8 the three projections share one quantisation of x.

        Inference runs QK-norm + rope as ONE kernel (`_qk_prep`, not bitwise); training
        keeps the eager ops the checkpoints were trained under."""
        orig = self.orig
        projections = (orig.to_q, orig.to_k, orig.to_v)

        if all(isinstance(p, Fp8Linear) for p in projections):
            x_fp8, x_scale = quantize_activation(x)
            qkv = [p.forward_quantized(x_fp8, x_scale, out_dtype=x.dtype) for p in projections]
        else:
            qkv = [p(x) for p in projections]

        query_raw, key_raw, value = (t.unflatten(-1, (orig.heads, -1)) for t in qkv)  # [total, H, d]

        if self.inference_mode and rotary_emb is not None:
            query = _qk_prep(query_raw, orig.norm_q.weight, orig.norm_q.eps, *rotary_emb)
            key = _qk_prep(key_raw, orig.norm_k.weight, orig.norm_k.eps, *rotary_emb)
        else:
            query, key = orig.norm_q(query_raw), orig.norm_k(key_raw)
            if rotary_emb is not None:
                query = _apply_rotary_emb(query.unsqueeze(0), *rotary_emb).squeeze(0)
                key = _apply_rotary_emb(key.unsqueeze(0), *rotary_emb).squeeze(0)

        return query, key, value, (query_raw, key_raw, value)

    def _bounds(self, layout):
        return window_bounds(layout.num_frames, self.radius, self.chunk)

    def forward(self, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("H3 packs one request per document; no attention mask expected.")
        if hidden_states.shape[0] != 1:
            raise ValueError("HybridAttention assumes batch 1 (one packed document).")

        if self.teacher_mode:
            out = self.orig(hidden_states, rotary_emb, attention_mask)
        else:
            out = self._hybrid_forward(hidden_states[0], rotary_emb).unsqueeze(0)

        return out

    def _window_kernel(self, inference, on_cuda):
        """Which window-softmax kernel this forward runs. `softmax_impl` is the process-wide
        choice (hybrid_transform.set_softmax_backend); the decomposition is inference-only,
        training keeps the differentiable flex path."""
        if not on_cuda or self.softmax_impl == "ref":
            return "ref"
        if self.softmax_impl == "decomposed" and inference:
            return "decomposed"
        return "flex"

    def _window_softmax(self, query, key, value, layout, bounds, scale, inference):
        kernel = self._window_kernel(inference, query.is_cuda)
        if kernel == "decomposed":
            return window_softmax_decomposed(query, key, value, layout, bounds, scale,
                                             anchor_frames=self.anchor_frames)
        if kernel == "flex":
            block_mask = build_window_block_mask(layout, bounds, value.device,
                                                 anchor_frames=self.anchor_frames)
            return window_softmax_flex(query, key, value, block_mask, scale,
                                       inference=inference)
        return window_softmax_reference(query, key, value, layout, bounds, scale,
                                        anchor_frames=self.anchor_frames)

    def _hybrid_forward(self, x, rotary_emb):
        layout = self.layout
        hybrid_inference = self.hybrid_inference_mode or self.inference_mode
        bounds = self._bounds(layout) if layout is not None else None
        full_cover = layout is None or all(
            lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
        scale = self.head_dim ** -0.5

        query, key, value, qkv_raw = self._qkv(x, rotary_emb)
        if full_cover:
            # A window wide enough to cover every frame IS the original attention, so go
            # through the stock processor's own dispatch rather than a bare SDPA call:
            # two different bf16 attention kernels over ~100k keys differ by far more
            # than the 0.99 gate contributes. Matching the kernel makes the full-cover
            # path exactly the teacher.
            softmax_out = dispatch_attention_fn(
                query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=getattr(type(self.orig.processor), "_attention_backend", None),
            ).squeeze(0)
            # nothing lies outside the window, so the linear branch would double count
            linear_active = False
        else:
            softmax_out = self._window_softmax(query, key, value, layout, bounds, scale,
                                               hybrid_inference)
            linear_active = True

        # The roped q/k (and the flex output) are dead once the local branch is done;
        # drop them before the linear branch runs — at H3 scale they are ~3 GiB that would
        # otherwise sit under the linear branch's own peak.
        del query, key, value
        if self.enable_softmax_gate:
            flat = apply_softmax_gate(softmax_out, self.softmax_gate(x),
                                     inference=self.inference_mode)
        else:
            flat = softmax_out.reshape(x.shape[0], -1)
        out = self.orig.to_out[0](flat.type_as(x))
        out = self.orig.to_out[1](out)
        del softmax_out

        if linear_active and self.linear_attention_enabled:
            video_start, video_end = layout.video_start, layout.video_end
            video_x = x[video_start:video_end]
            video_qkv_raw = tuple(t[video_start:video_end] for t in qkv_raw)
            text_x = text_qkv_raw = None
            if self.enable_text_state:
                text_start, text_end = layout.text_range
                text_x = x[text_start:text_end]
                text_qkv_raw = tuple(t[text_start:text_end] for t in qkv_raw)

            # The grid is only DEMANDED when the conv exists: layout.frame_size raises
            # on grid-less layouts, and every conv-less config must keep working with
            # them (only frame_size consumers are gated on short_conv).
            linear_readout = self.linear_attention(video_x, layout.num_frames, layout.tokens_per_frame,
                                   bounds, qkv_raw=video_qkv_raw,
                                   frame_size=(layout.frame_size
                                               if self.linear_attention.short_conv is not None
                                               else None),
                                   skip_ends=self.anchor_frames == "both",
                                   text_x=text_x, text_qkv_raw=text_qkv_raw,
                                   inference=hybrid_inference)

            # The clone is autograd's: `out` is the output of self.orig.to_out[1], and
            # writing into it in place would corrupt what backward needs. With no graph
            # to protect there is nothing to copy -- and at H3 scale this tensor is
            # [~105k, 5376] bf16 = 1.05 GiB, copied once per layer per denoising step.
            if torch.is_grad_enabled():
                out = out.clone()
            out[video_start:video_end] += self.to_out_linear(linear_readout.type_as(x))
        return out

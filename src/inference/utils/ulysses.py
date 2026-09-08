"""Inference-only Ulysses sequence parallelism for the H3 hybrid transformer.

The rank layout, the collectives and the Triton pack kernel are ulysses_runtime.py; this
module is the forwards that use them (standard 8-way and branch-parallel 6+2), the
head-sharded far branch, and ``install_ulysses``.

This module deliberately lives outside ``src.models``.  It installs instance-local
forward methods after the inference model has been built, its LoRAs have been merged,
and the regular inference kernels/FP8 replacements have been selected.  Importing it
does not change model or training behaviour.

The block residual stream is sharded by packed-sequence row.  Inside attention one
all-to-all changes ``sequence-sharded / all-heads`` QKV into ``all-sequence /
head-sharded`` QKV; a second all-to-all restores the residual-stream layout.  The
window attention and the recurrent far branch therefore still see the complete
sequence.  No KV cache, stale state, token dropping, or approximate communication is
used.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3TransformerOutput,
)

from src.models.hybrid_transform import iter_hybrids
from src.models.softmax_attention import (
    build_window_block_mask,
    window_softmax_flex,
    window_softmax_reference,
)
from src.models.softmax_attention.kernels import apply_softmax_gate
from src.inference.utils.ulysses_runtime import (  # noqa: F401 -- re-exported
    UlyssesRuntime,
    balanced_splits,
    init_ulysses,
    sequence_splits,
)


def _window_softmax_branch(self, query, key, value, layout, bounds, scale):
    """Rank-local window softmax: the same kernel choice as HybridAttention._window_kernel
    (kernels.softmax_backend, inference-only decomposition). The decomposition is
    head-count-agnostic, so Ulysses shards ([T, heads_per_rank, d] strided views of
    the packed buffer) go through unchanged."""
    if self._window_kernel(self.inference_mode, query.is_cuda) == "decomposed":
        from src.models.softmax_attention.decomposed import window_softmax_decomposed
        return window_softmax_decomposed(query, key, value, layout, bounds, scale,
                                         anchor_frames=self.anchor_frames)
    block_mask = build_window_block_mask(
        layout, bounds, value.device, anchor_frames=self.anchor_frames
    )
    # FA4-CuTe flex mis-computes on slice-strided operands on sm100 (independent of
    # FA_CLC). sm90 computes these views correctly, so the defensive copy is gated to
    # sm100 and sm90 keeps its zero-copy path.
    if query.is_cuda and torch.cuda.get_device_capability(query.device)[0] >= 10:
        query, key, value = (
            t if t.is_contiguous() else t.contiguous() for t in (query, key, value)
        )
    return window_softmax_flex(query, key, value, block_mask, scale,
                               inference=self.inference_mode)


def _linear_branch_shard(attn, raw_qkv, beta, gate, frame_mean, first_head, last_head):
    """This rank's heads of the far branch over the whole sequence: slice the video (and
    prompt) rows out of the packed sequence and run the branch's inference body on the
    head range, with the beta / gate / frame mean the sequence owners computed before
    dispatch. Returns [seq_len, heads, d] with zeros on the non-video rows."""
    layout = attn.layout
    branch = attn.linear_attention
    heads = slice(first_head, last_head)
    out = gate.new_zeros(layout.seq_len, last_head - first_head, branch.head_dim)

    video = slice(layout.video_start, layout.video_end)
    text_raw = text_beta = None
    if attn.enable_text_state:
        text_start, text_end = layout.text_range
        text_raw = tuple(t[text_start:text_end] for t in raw_qkv)
        text_beta = beta[text_start:text_end]

    readout = branch(
        None, layout.num_frames, layout.tokens_per_frame, attn._bounds(layout),
        tuple(t[video] for t in raw_qkv),
        frame_size=(layout.frame_size if branch.short_conv is not None else None),
        skip_ends=attn.anchor_frames == "both",
        text_qkv_raw=text_raw,
        inference=True, heads=heads, beta=beta[video], gate=gate[video],
        frame_mean=frame_mean, text_beta=text_beta,
    )
    out[video] = readout.view(-1, last_head - first_head, branch.head_dim)
    return out


def _ulysses_attention_forward(self, hidden_states, rotary_emb=None, attention_mask=None):
    if attention_mask is not None:
        raise ValueError("H3 Ulysses does not accept an attention mask")
    if hidden_states.shape[0] != 1:
        raise ValueError("H3 Ulysses inference assumes batch size 1")
    if self.teacher_mode:
        raise RuntimeError("teacher_mode is incompatible with inference-only Ulysses")

    runtime: UlyssesRuntime = self._ulysses_runtime
    x = hidden_states[0]
    layout = self.layout
    if layout is None:
        raise RuntimeError("set_layout must run before a Ulysses hybrid forward")

    # The A4 refactor made the frame mean async and updated only the branch-parallel
    # path; this standard-Ulysses path kept the retired sync name, and nothing ran it
    # (every shipped config is branch-parallel), so it rotted into an AttributeError.
    # Restored with SYNC semantics on the surviving API: launch the all-reduce and
    # wait immediately -- deliberately no overlap, so softmax_ranks=0 stays an honest
    # pre-A4 baseline (A4's win belongs to the branch-parallel path only).
    frame_sums, frame_work = runtime.video_frame_mean_async(x, layout)
    frame_work.wait()
    frame_mean = frame_sums / layout.tokens_per_frame
    query, key, value, raw = self._qkv(x, rotary_emb)

    # Compute the x-dependent gates while all heads are local.  Sending them beside
    # QKV is substantially cheaper than all-gathering the 5376-wide residual stream.
    if self.enable_softmax_gate:
        softmax_gate = self.softmax_gate(x)
    else:
        softmax_gate = x.new_ones(x.shape[0], self.num_heads, 1)
    beta = torch.sigmoid(self.linear_attention.beta_proj(x)).unsqueeze(-1)
    linear_gate = self.linear_attention.output_gate(x)

    packed = torch.cat(
        [query, key, value, raw[0], raw[1], softmax_gate, beta, linear_gate],
        dim=-1,
    )
    packed = runtime.sequence_to_heads(packed)
    d = self.head_dim
    linear_d = self.linear_attention.head_dim
    widths = (d, d, d, d, d, 1, 1, linear_d)
    query, key, value, raw_q, raw_k, softmax_gate, beta, linear_gate = packed.split(
        widths, dim=-1
    )
    raw_qkv = (raw_q, raw_k, value)
    beta = beta.squeeze(-1)

    bounds = self._bounds(layout)
    full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
    scale = self.head_dim**-0.5
    if full_cover:
        softmax_out = dispatch_attention_fn(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(type(self.orig.processor), "_attention_backend", None),
        ).squeeze(0)
        linear_active = False
    elif self.softmax_impl in ("flex", "decomposed") and x.is_cuda:
        softmax_out = _window_softmax_branch(
            self, query, key, value, layout, bounds, scale
        )
        linear_active = True
    else:
        softmax_out = window_softmax_reference(
            query,
            key,
            value,
            layout,
            bounds,
            scale,
            anchor_frames=self.anchor_frames,
        )
        linear_active = True

    softmax_out = apply_softmax_gate(
        softmax_out, softmax_gate, inference=self.inference_mode
    ).view(layout.seq_len, runtime.heads_per_rank, d)
    first_head = runtime.rank * runtime.heads_per_rank
    last_head = first_head + runtime.heads_per_rank
    if linear_active and self.linear_attention_enabled:
        linear_out = _linear_branch_shard(
            self,
            raw_qkv,
            beta,
            linear_gate,
            frame_mean,
            first_head,
            last_head,
        )
    else:
        linear_out = linear_gate.new_zeros(layout.seq_len, runtime.heads_per_rank, linear_d)

    restored = runtime.heads_to_sequence(torch.cat([softmax_out, linear_out], dim=-1))
    softmax_local, linear_local = restored.split((d, linear_d), dim=-1)
    rows = x.shape[0]
    out = self.orig.to_out[0](softmax_local.reshape(rows, -1).type_as(x))
    out = self.orig.to_out[1](out)
    if linear_active and self.linear_attention_enabled:
        positions = torch.arange(runtime.local_start, runtime.local_end, device=x.device)
        is_video = (positions >= layout.video_start) & (positions < layout.video_end)
        if is_video.any():
            out[is_video] += self.to_out_linear(
                linear_local[is_video].reshape(int(is_video.sum().item()), -1).type_as(x)
            )
    return out.unsqueeze(0)


def _branch_parallel_attention_forward(
    self,
    hidden_states,
    rotary_emb=None,
    attention_mask=None,
):
    """Shared-QKV Ulysses with softmax and linear attention on disjoint rank groups."""
    if attention_mask is not None:
        raise ValueError("H3 branch-parallel Ulysses does not accept an attention mask")
    if hidden_states.shape[0] != 1:
        raise ValueError("H3 branch-parallel Ulysses inference assumes batch size 1")
    if self.teacher_mode:
        raise RuntimeError("teacher_mode is incompatible with inference-only Ulysses")

    runtime: UlyssesRuntime = self._ulysses_runtime
    x = hidden_states[0]
    layout = self.layout
    if layout is None:
        raise RuntimeError("set_layout must run before a Ulysses hybrid forward")

    # QKV is evaluated exactly once: every residual-stream owner projects all heads for
    # its S/8 rows.  The uneven A2A below sends the processed QKV to six softmax ranks
    # and the raw QKV to two linear ranks.
    profile_event = runtime.profile_start()
    # The far branch's alpha needs every frame's mean over tokens that straddle rank
    # boundaries: launch the all-reduce now and let QKV, gates and the dispatch hide it.
    frame_sums, frame_work = runtime.video_frame_mean_async(x, layout)
    runtime.profile_end("frame_mean_launch", profile_event)

    profile_event = runtime.profile_start()
    query, key, value, raw = self._qkv(x, rotary_emb)
    runtime.profile_end("qkv", profile_event)
    profile_event = runtime.profile_start()
    if self.enable_softmax_gate:
        softmax_gate = self.softmax_gate(x)
    else:
        softmax_gate = x.new_ones(x.shape[0], self.num_heads, 1)
    beta = torch.sigmoid(self.linear_attention.beta_proj(x)).unsqueeze(-1)
    # Only the low-rank half of the linear output gate travels: the receiving linear
    # rank applies `up` for its own heads (below), so the payload carries the small
    # hidden instead of the full 56 x d gate.
    gate_module = self.linear_attention.output_gate
    if gate_module.down is None:
        raise RuntimeError("branch-parallel Ulysses needs the low-rank linear output gate")
    linear_gate = gate_module.down(x)
    runtime.profile_end("gates", profile_event)

    # One Triton pack straight into the send buffers, then the softmax and linear A2As
    # on their own communicators; this rank waits for its own branch's payload only and
    # holds the other handle until its compute is done.
    profile_event = runtime.profile_start()
    packed, received_gate_hidden, pending_dispatch = (
        runtime.dispatch_fields_to_branches_overlapped(
            query,
            key,
            value,
            softmax_gate,
            raw[0],
            raw[1],
            beta,
            linear_gate,
        )
    )
    runtime.profile_end("branch_dispatch", profile_event)
    del query, key, value, raw, softmax_gate, beta, linear_gate

    profile_event = runtime.profile_start()
    frame_work.wait()
    frame_mean = frame_sums / layout.tokens_per_frame
    runtime.profile_end("frame_mean_wait", profile_event)

    bounds = self._bounds(layout)
    full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
    linear_active = not full_cover
    scale = self.head_dim**-0.5
    d = self.head_dim
    linear_d = self.linear_attention.head_dim
    first_head = runtime.branch_first_head
    last_head = first_head + runtime.branch_heads

    profile_event = runtime.profile_start()
    if runtime.branch_kind == "softmax":
        query, key, value, softmax_gate = packed.split((d, d, d, 1), dim=-1)
        if full_cover:
            branch_out = dispatch_attention_fn(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=getattr(type(self.orig.processor), "_attention_backend", None),
            ).squeeze(0)
        elif self.softmax_impl in ("flex", "decomposed") and x.is_cuda:
            branch_out = _window_softmax_branch(
                self, query, key, value, layout, bounds, scale
            )
        else:
            branch_out = window_softmax_reference(
                query,
                key,
                value,
                layout,
                bounds,
                scale,
                anchor_frames=self.anchor_frames,
            )
        branch_out = apply_softmax_gate(
            branch_out, softmax_gate, inference=self.inference_mode
        ).view(layout.seq_len, runtime.branch_heads, d)
    else:
        # The gate's `up` half runs here, on this rank's own heads only.
        raw_q, raw_k, value, beta = packed.split((d, d, d, 1), dim=-1)
        first = first_head * linear_d
        last = last_head * linear_d
        bias = (
            None
            if gate_module.up.bias is None
            else gate_module.up.bias[first:last]
        )
        linear_gate = torch.sigmoid(
            F.linear(
                received_gate_hidden,
                gate_module.up.weight[first:last],
                bias,
            )
        ).view(layout.seq_len, runtime.branch_heads, linear_d)

        if linear_active and self.linear_attention_enabled:
            branch_out = _linear_branch_shard(
                self,
                (raw_q, raw_k, value),
                beta.squeeze(-1),
                linear_gate,
                frame_mean,
                first_head,
                last_head,
            )
        else:
            branch_out = packed.new_zeros(
                layout.seq_len, runtime.branch_heads, linear_d
            )
    runtime.profile_end(f"{runtime.branch_kind}_compute", profile_event)
    if pending_dispatch is not None:
        profile_event = runtime.profile_start()
        pending_dispatch[0].wait()
        runtime.profile_end("branch_other_wait", profile_event)
        del pending_dispatch

    # One uneven reverse A2A is also the branch merge: every sequence owner receives
    # all 56 softmax heads and all 56 linear heads, then applies the original two
    # independent output projections locally.  Keeping the projections independent is
    # important for FP8 because each branch has its own rowwise activation scale.
    profile_event = runtime.profile_start()
    softmax_local, linear_local = runtime.branches_to_sequence(branch_out)
    runtime.profile_end("output_dispatch", profile_event)
    profile_event = runtime.profile_start()
    rows = x.shape[0]
    out = self.orig.to_out[0](softmax_local.reshape(rows, -1).type_as(x))
    out = self.orig.to_out[1](out)
    if linear_active and self.linear_attention_enabled:
        positions = torch.arange(runtime.local_start, runtime.local_end, device=x.device)
        is_video = (positions >= layout.video_start) & (positions < layout.video_end)
        if is_video.any():
            out[is_video] += self.to_out_linear(
                linear_local[is_video].reshape(int(is_video.sum().item()), -1).type_as(x)
            )
    runtime.profile_end("output_projection", profile_event)
    return out.unsqueeze(0)


def _ulysses_transformer_forward(
    self,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
    attention_kwargs=None,
    return_dict=True,
):
    """Inference-only copy of the H3 forward with the block residual stream sharded."""
    if torch.is_grad_enabled():
        raise RuntimeError("Ulysses forward is inference-only; use torch.no_grad()")
    if attention_kwargs and attention_kwargs.get("scale", 1.0) != 1.0:
        raise ValueError("merge LoRA weights before installing Ulysses; runtime LoRA scale is unsupported")
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"position_ids must be [S,3], got {tuple(position_ids.shape)}")

    runtime: UlyssesRuntime = self._ulysses_runtime
    sequence_length = position_ids.shape[0]
    first_attn = next(iter_hybrids(self))
    runtime.configure(sequence_length, first_attn.num_heads)
    rotary_emb = self.rope(position_ids)

    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(
        audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in))
    )
    text_embeds = self.context_embedder(
        encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
    )
    text_embeds = self.token_refiner(text_embeds)
    packed = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    packed = packed.index_copy(1, text_indices, text_embeds)
    packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

    start, end = runtime.local_start, runtime.local_end
    packed = packed[:, start:end].contiguous()
    local_adaln = adaln_indices[start:end]
    local_rotary = tuple(t[start:end] for t in rotary_emb)
    for block in self.transformer_blocks:
        packed = block(packed, temb, local_adaln, local_rotary)

    packed = runtime.gather_sequence(packed[0]).unsqueeze(0)
    packed = self.norm_out(packed, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
    video_output = self.proj_out(packed).index_select(1, video_indices)
    audio_output = self.audio_proj_out(packed).index_select(1, audio_indices)
    if not return_dict:
        return video_output, audio_output
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def install_ulysses(
    model,
    runtime: UlyssesRuntime,
    *,
    softmax_ranks: int | None = None,
) -> None:
    """Install Ulysses only on this already-built inference model instance.

    ``softmax_ranks`` (config ``parallel.softmax_ranks``): None/0 = standard Ulysses,
    every rank owns heads of both branches; n = branch-parallel, n softmax ranks plus
    (world - n) linear ranks -- 6+2 on eight GPUs is the shipped layout."""
    hybrids = list(iter_hybrids(model))
    if not hybrids:
        raise ValueError("Ulysses path currently requires a HybridAttention checkpoint")
    heads = hybrids[0].num_heads
    if softmax_ranks:
        runtime.enable_branch_parallel(softmax_ranks)
    if heads % runtime.world_size:
        raise ValueError(f"{heads} attention heads are not divisible by {runtime.world_size} ranks")
    for attn in hybrids:
        if attn.num_heads != heads:
            raise ValueError("all H3 blocks must have the same attention head count")
        if attn.linear_attention.head_dim != attn.head_dim:
            raise ValueError(
                "the inference Ulysses pack currently requires linear_head_dim == attention head_dim"
            )
        attn._ulysses_runtime = runtime
        if runtime.branch_parallel:
            attention_forward = _branch_parallel_attention_forward
        else:
            attention_forward = _ulysses_attention_forward
        attn.forward = types.MethodType(attention_forward, attn)
    model._ulysses_runtime = runtime
    model.forward = types.MethodType(_ulysses_transformer_forward, model)

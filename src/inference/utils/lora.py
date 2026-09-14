"""LoRA merging at inference: the Stage-B artifact's pairs and external (community)
adapters fold into the base projections the same way -- W += scale * (B @ A), delta
computed in fp32, added in the weight's dtype. One home for both so the merge
semantics cannot drift apart.

External adapters (e.g. community MiniMax-H3 few-step distills) come as safetensors
with peft key names against the DENSE model (`transformer_blocks.N.attn.to_q.lora_A.
default.weight`) and may stamp their alpha in the file metadata; when they do, the
merge is scaled by alpha / rank, since an unscaled merge of a low-alpha adapter would
be far too strong. A missing alpha means alpha == rank (scale 1.0).

On a hybrid model the dense attention lives one level down (`attn.orig.*`); target
resolution tries the dense name first, then re-roots through `.attn.orig.` -- the
transform wraps but never renames, so both spellings reach the same tensor.
"""
import re

import torch
from safetensors import safe_open

# Adapters trained against the ORIGINAL (non-diffusers) checkpoint layout, module by
# module onto the diffusers names. adaln_proj.linear and norm_out.linear match the
# diffusers Linears exactly; mlp.fc1 is the same fused swiglu projection as
# ff.net.0.proj, no split needed.
_ORIGINAL_RENAMES = (
    (".mlp.fc1.", ".ff.net.0.proj."),
    (".mlp.fc2.", ".ff.net.2."),
    (".attn.out_proj.", ".attn.to_out.0."),
)


def _translate_original_layout(state):
    """Rewrite original-layout keys (blocks.N..., fused qkv_proj) onto diffusers
    names. Handles both adapters trained directly against the original layout and
    ComfyUI re-exports (`diffusion_model.` prefix).

    Two structural conversions, both exact:
    - the fused qkv pair becomes three pairs sharing lora_A, with lora_B row-chunked
      in the checkpoint's (q, k, v) order -- delta rows factor per projection whether
      B is dense or block-diagonal, so this also covers rank-resized re-exports;
    - the swiglu halves of mlp.fc1's lora_B are SWAPPED: the original layout packs
      [gate; value], diffusers' ff.net.0.proj packs [value; gate] (diffusers' SwiGLU
      chunks (value, gate)). Without the swap the gate deltas land on the value half
      and vice versa.
    Diffusers-named adapters pass through untouched."""
    if not any(".qkv_proj." in k for k in state):
        return state
    out = {}
    for name, t in state.items():
        if name.startswith("diffusion_model."):
            name = name[len("diffusion_model."):]
        name = re.sub(r"^blocks\.", "transformer_blocks.", name)
        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        name = name.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
        if ".mlp.fc1." in name and ".lora_B." in name:
            gate, value = t.chunk(2, dim=0)
            t = torch.cat([value, gate], dim=0)
        for old, new in _ORIGINAL_RENAMES:
            name = name.replace(old, new)
        if ".attn.qkv_proj." in name:
            chunks = ((t, t, t) if ".lora_A." in name else t.chunk(3, dim=0))
            for proj, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                out[name.replace(".attn.qkv_proj.", f".attn.{proj}.")] = chunk
            continue
        out[name] = t
    return out


def _resolve(params, target):
    if target in params:
        return params[target]
    if ".attn." in target:
        rerooted = target.replace(".attn.", ".attn.orig.", 1)
        if rerooted in params:
            return params[rerooted]
    raise KeyError(f"LoRA target {target!r} has no parameter in the model "
                   "(also tried the .attn.orig. rerooting)")


def merge_lora_state(model, state, scale=1.0):
    """Fold peft-named `.lora_A. / .lora_B.` pairs into the model, in place.
    Returns the number of pairs merged."""
    params = dict(model.named_parameters())
    merged = 0
    for name, a in state.items():
        if ".lora_A." not in name:
            continue
        b = state[name.replace(".lora_A.", ".lora_B.")]
        target = name.split(".lora_A.")[0] + ".weight"
        w = _resolve(params, target)
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
        w.data.add_(delta.to(w.device, w.dtype))
        merged += 1
    return merged


def load_external_lora(path, alpha=None):
    """Read a community LoRA safetensors -> (state, scale, rank). The peft `.default.`
    adapter infix is stripped; `alpha` overrides the file metadata when given."""
    state = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            state[key.replace(".default.", ".")] = f.get_tensor(key)
    state = _translate_original_layout(state)
    ranks = sorted({state[k].shape[0] for k in state if ".lora_A." in k})

    if alpha is None and "alpha" not in meta:
        scale = 1.0                         # alpha == rank per pair: plain W += B @ A
    else:
        if len(ranks) != 1:
            raise ValueError(f"{path}: one alpha ({alpha or meta['alpha']}) cannot scale "
                             f"mixed ranks {ranks}")
        alpha = float(meta["alpha"]) if alpha is None else float(alpha)
        scale = alpha / ranks[0]

    return state, scale, ranks

"""The VDN-H3 transformer as a plain diffusers component.

This is the SOURCE of the remote code published under each checkpoint's `diffusers/`
directory on the Hub; `src/diffusers_export/export.py` copies it there together with the
`src.models` closure it imports, rewriting every `src.` import to a flat relative one.
Nothing here is imported by this repository's own inference stack -- `assemble.py`
remains the one assembly path for `infer.py` and `infer_ulysses.py`.

    AutoModel.from_pretrained("OpenVDN/vdn-minimax-h3",
                              subfolder="stage-dmd-step-250/diffusers",
                              trust_remote_code=True)

`from_pretrained` is overridden because the three things it assembles do not live in
one directory: the base transformer is 66 GB of MiniMax-H3 under `h3-base/`, and the
VDN weights are the linear branch and the LoRA adapters one level up from `diffusers/`.
Pointing at them instead of copying them is why the published layout holds no duplicate
weights. The paths are `config.json`'s `vdn` block, written relative to the checkpoint
directory that contains `diffusers/`.
"""
import json
import os
import posixpath

import torch
from diffusers import MiniMaxH3Transformer3DModel

from src.inference.utils.lora import merge_lora_state
from src.models.hybrid_transform import (apply_hybrid_attention_transform, iter_hybrids,
                                         set_inference_mode, set_softmax_backend)

CONFIG_KEY = "vdn"


def _open(source, repo_path, **hub_kwargs):
    """`repo_path` inside a local directory or a Hub repo id, as a local file path."""
    if os.path.isdir(source):
        return os.path.join(source, *repo_path.split("/"))

    from huggingface_hub import hf_hub_download

    return hf_hub_download(source, repo_path,
                           **{k: v for k, v in hub_kwargs.items() if v is not None})


def _sibling(subfolder, relative):
    """A `vdn` block path -- relative to the checkpoint directory, one level above
    `diffusers/` -- as a path from the repository root."""
    return posixpath.normpath(posixpath.join(subfolder or ".", "..", relative))


def _load_branch(model, weights):
    """The transform's own tensors. Refuses a key with no parameter rather than
    dropping it: a branch that half-loads renders, badly."""
    params = dict(model.named_parameters())
    unknown = [k for k in weights if k not in params]
    if unknown:
        raise RuntimeError(f"{len(unknown)} branch keys have no parameter, e.g. "
                           f"{sorted(unknown)[:4]}")
    for name, value in weights.items():
        params[name].data.copy_(value.to(params[name].dtype))
    return len(weights)


class VDNMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    """MiniMax-H3 with the VDN hybrid attention transform applied and the checkpoint's
    LoRA adapters folded in. Same forward, same config and the same outputs as the base
    class, so every MiniMax-H3 pipeline block accepts it unchanged."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, subfolder=None,
                        cache_dir=None, revision=None, token=None,
                        local_files_only=None, **kwargs):
        from safetensors.torch import load_file

        source = pretrained_model_name_or_path
        hub = {"cache_dir": cache_dir, "revision": revision, "token": token,
               "local_files_only": local_files_only}

        with open(_open(source, posixpath.join(subfolder or ".", "config.json"), **hub)) as f:
            spec = json.load(f)[CONFIG_KEY]

        # The base checkpoint is mixed precision and carries no `torch_dtype`, so a load
        # that names no dtype keeps every stored dtype -- a different model from the one
        # this checkpoint was distilled against, at twice the memory. bf16 is what
        # `render.load_models` asks for; `_keep_in_fp32_modules` still holds back the
        # time embedder. A caller that names a dtype gets that one.
        base = spec["base"]
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = torch.bfloat16
        model = super().from_pretrained(
            base["source"], subfolder=base["subfolder"], revision=base.get("revision"),
            cache_dir=cache_dir, token=token, local_files_only=local_files_only, **kwargs)

        apply_hybrid_attention_transform(model, spec["transform"])
        _load_branch(model, load_file(_open(source, _sibling(subfolder, spec["branch"]), **hub)))
        for adapter in spec["adapters"]:
            merge_lora_state(model, load_file(_open(source, _sibling(subfolder, adapter), **hub)))

        model.eval().requires_grad_(False)
        for attn in iter_hybrids(model):
            attn.teacher_mode = False
            for parameter in attn.parameters():
                if parameter.dtype == torch.float32:
                    parameter.data = parameter.data.to(torch.bfloat16)

        # The overlay `assemble.build_inference_model` applies for a render: forward-only
        # kernel bodies and the window-softmax backend. Off makes this slower, never
        # wrong, so a caller that means to build a graph can set inference_kernels false.
        if spec.get("inference_kernels", True):
            set_inference_mode(model, True)
            set_softmax_backend(model, spec.get("softmax_backend", "auto"))
        return model

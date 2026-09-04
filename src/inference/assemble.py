"""One model assembly for every inference entrypoint.

    model = build_inference_model(cfg, device)          # infer.py
    model = build_inference_model(cfg, device, load_decoders=runtime.is_main,
                                  log=runtime.is_main)  # infer_ulysses.py

Order of operations, the same for both: read the artifact's spec -> load the base ->
apply the transform -> load the branch weights -> fold the artifact's LoRAs -> fold the
external LoRAs -> eval/no-grad on device -> behaviour -> the inference kernel set
(set_inference_mode) -> window-softmax backend -> ablation -> fp8. Then `render_record`
says what ACTUALLY ran (fp8 layer count, the flex latch, the resolved backend), so a
silent downgrade is never silent in the JSON next to the mp4.

Every knob is a config field (src/config/inference.py); nothing here reads an
environment variable. `checkpoint: null` is the released DENSE model: no transform, no
branch, and the hybrid-only overlays are skipped.
"""
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.checkpoints import checkpoint_head_sha256, load_checkpoint
from src.config import resolved_dict
from src.inference.lora import load_external_lora, merge_lora_state
from src.inference.render import DEFAULT_MODEL_ROOT, load_models
from src.paths import resolve_weights
from src.models.factory import load_model_weights
from src.models.hybrid_transform import (apply_hybrid_attention_transform, iter_hybrids,
                                         set_inference_mode, set_softmax_backend)
from src.models.ops.fp8_linear import convert_linear_to_fp8
from src.models.softmax_attention.decomposed import decomposition_state
from src.models.softmax_attention.flex_attention import _FLEX_CACHE

# Semantic knobs an ablation may touch, and where they live. Anything else is refused.
ABLATABLE = ("radius", "chunk", "anchor_frames", "bridge", "enable_text_state",
             "linear_attention_enabled", "enable_softmax_gate")


@dataclass
class InferenceModel:
    transformer: Any
    vae: Any                                  # None on ranks that do not decode
    audio_vae: Any
    artifact: Any                             # CheckpointArtifact, or None for dense
    is_hybrid: bool
    merged_lora_pairs: int = 0
    external_loras: List[Dict[str, Any]] = field(default_factory=list)
    ablated: List[str] = field(default_factory=list)
    fp8_linears: int = 0
    softmax_backend: Optional[str] = None     # resolved (auto -> flex|decomposed)


def apply_ablation(model, overrides):
    for key in overrides:
        if key not in ABLATABLE:
            raise ValueError(f"{key!r} is not an ablatable knob ({ABLATABLE})")

    for attn in iter_hybrids(model):
        for key, value in overrides.items():
            setattr(attn, key, value)

    return sorted(overrides)


def flex_latch_state():
    return {"infer_disabled": bool(_FLEX_CACHE.get("infer_disabled")),
            "compiled_variants": sorted(k for k in _FLEX_CACHE if k != "infer_disabled")}


def build_inference_model(cfg, device, *, load_decoders: bool = True,
                          log: bool = True) -> InferenceModel:
    say = print if log else (lambda *a, **k: None)

    art = None
    if cfg.checkpoint:
        art = load_checkpoint(resolve_weights(cfg.checkpoint))
        if art.metadata.get("truncated_blocks"):
            raise RuntimeError(f"{cfg.checkpoint} is a truncated smoke-test artifact")

    base_source = cfg.base_source or (
        art.model_spec["base"]["source"] if art else DEFAULT_MODEL_ROOT)
    transformer, vae, audio_vae = load_models(
        base_source, device, vae_source=cfg.vae_source, load_decoders=load_decoders)

    merged = 0
    if art:
        apply_hybrid_attention_transform(
            transformer, art.model_spec["transforms"][0]["config"])

        branch_weights = {k: v for k, v in art.weights.items() if "lora_" not in k}
        lora_weights = {k: v for k, v in art.weights.items() if "lora_" in k}
        loaded = load_model_weights(transformer, branch_weights)
        merged = merge_lora_state(transformer, lora_weights) if lora_weights else 0

        say(f"built from spec: {loaded} branch tensors, {merged} LoRA pairs merged",
            flush=True)
    else:
        say("DENSE BASE render: no checkpoint -- no transform, no branch", flush=True)

    transformer.to(device).eval().requires_grad_(False)

    external_records = []
    for xl in cfg.external_loras:
        state, scale, ranks = load_external_lora(xl.path, xl.alpha)
        pairs = merge_lora_state(transformer, state, scale)

        say(f"external LoRA {xl.path}: {pairs} pairs merged (ranks {ranks}, "
            f"scale {scale:g})", flush=True)
        external_records.append({"path": os.path.abspath(xl.path), "pairs": pairs,
                                 "ranks": ranks, "scale": scale})

    # ---- overlays, in order ----
    is_hybrid = next(iter_hybrids(transformer), None) is not None

    for attn in iter_hybrids(transformer):
        attn.teacher_mode = cfg.behavior.teacher_mode
        for prm in attn.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)

    softmax_backend = None
    if is_hybrid:
        if cfg.kernels.inference_kernels:
            set_inference_mode(transformer, True)

        softmax_backend = set_softmax_backend(transformer, cfg.kernels.softmax_backend)
        say(f"window softmax: {softmax_backend}", flush=True)

    ablated = (apply_ablation(transformer, dict(cfg.ablation.overrides))
               if cfg.ablation.enabled and cfg.ablation.overrides else [])
    if ablated:
        say(f"ABLATION ACTIVE: {ablated} -- this output is a study, not a sample",
            flush=True)

    fp8_handle = []
    if cfg.precision.fp8.enabled:
        fp8_handle = convert_linear_to_fp8(
            transformer, skip_end_blocks=cfg.precision.fp8.skip_end_blocks,
            keep_original=cfg.precision.fp8.keep_original)
        say(f"fp8: {len(fp8_handle)} Linears quantised -- same seed will NOT "
            f"reproduce a bf16 render", flush=True)

    return InferenceModel(transformer=transformer, vae=vae, audio_vae=audio_vae,
                          artifact=art, is_hybrid=is_hybrid, merged_lora_pairs=merged,
                          external_loras=external_records, ablated=ablated,
                          fp8_linears=len(fp8_handle), softmax_backend=softmax_backend)


def render_record(cfg, model: InferenceModel) -> Dict[str, Any]:
    """The record next to the mp4: checkpoint identity, the overlay, the git commit and
    the ACTUAL kernel state. Entrypoints add their own sections (timings, parallel)."""
    art = model.artifact
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()

    return {
        "checkpoint": resolve_weights(cfg.checkpoint) if cfg.checkpoint else None,
        "checkpoint_head_sha256": (checkpoint_head_sha256(resolve_weights(cfg.checkpoint))
                                   if cfg.checkpoint else None),
        "model_spec_base_hash": art.model_spec["base"]["config_hash"] if art else None,
        "git_commit": commit,
        "overlay": resolved_dict(cfg),
        "ablation_active": model.ablated,
        "external_loras": model.external_loras,
        "fp8_linears": model.fp8_linears,
        "flex_backend": flex_latch_state(),
        "softmax_backend": {"resolved": model.softmax_backend, **decomposition_state()},
        "merged_lora_pairs": model.merged_lora_pairs,
    }


def latents_path(cfg) -> str:
    """Where `render.save_latents` puts the pre-VAE tensors: mirrors the mp4's path
    under results/ into `render.latents_root`."""
    out_abs = os.path.abspath(cfg.render.out)
    rel_out = os.path.relpath(out_abs, os.path.abspath("results"))

    if rel_out == os.pardir or rel_out.startswith(os.pardir + os.sep):
        rel_out = os.path.basename(out_abs)

    return os.path.join(cfg.render.latents_root, rel_out + ".latents.pt")


def write_json(record: Dict[str, Any], path: str) -> None:
    import json

    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")

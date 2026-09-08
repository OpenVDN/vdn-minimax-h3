"""build_model(model_spec): the one constructor every stage uses.

    artifact = load_checkpoint(path)
    model = build_model(artifact.model_spec)
    load_model_weights(model, artifact.weights)

Canonical model = architecture built, weights loadable, NO overlays yet (no FSDP,
no fp8, no compile, no inference bodies, no ablation). Overlays come after -- training:
src/utils/{lr_classes,activation_checkpointing}.py + FSDP in the trainers; inference:
hybrid_transform.set_inference_mode, ops/fp8_linear -- and never enter the spec.
"""
from typing import Any, Dict, Union

import diffusers
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, inject_adapter_in_model

from src.models.model_spec import (HYBRID_TRANSFORM_VERSION, BaseSpec, ModelSpec,
                                   TransformSpec, validate_spec)
from src.paths import resolve_weights
from src.models.hybrid_transform import (TRANSFORM_TYPE,
                                                    apply_hybrid_attention_transform)

_TRANSFORMS = {TRANSFORM_TYPE: apply_hybrid_attention_transform}
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32,
           "float16": torch.float16}


def build_model(spec: Union[ModelSpec, Dict[str, Any]], device: str = "cpu",
                base_source: str = None):
    """Load the HF base named by the spec, verify its config against the spec's
    resolved_config/hash, apply the transforms in order. `base_source` overrides the
    spec's source PATH only (a local copy of the base may live anywhere); class,
    revision and config identity still come from the spec."""
    if not isinstance(spec, ModelSpec):
        spec = ModelSpec.from_dict(spec)
    validate_spec(spec)
    base = spec.base
    if base.library != "diffusers":
        raise ValueError(f"unknown base library {base.library!r}")
    cls = getattr(diffusers, base.class_name)
    model = cls.from_pretrained(resolve_weights(base_source or base.source),
                                subfolder=base.subfolder,
                                torch_dtype=_DTYPES[base.resolved_config.get(
                                    "_load_dtype", "bfloat16")])
    _verify_base_config(model, base)
    model = model.to(device)
    for t in spec.transforms:
        if t.type not in _TRANSFORMS:
            raise ValueError(f"unknown transform {t.type!r} (version {t.version})")
        _TRANSFORMS[t.type](model, t.config)

    # Adapters are deliberately NOT injected here: the two consumers want different
    # things from the same spec. The B trainer wants live peft modules
    # (inject_adapters below); inference wants the adapter FOLDED into the base
    # weights (src/inference/utils/lora.merge_lora_state) so the render pays nothing for it.
    return model


def inject_adapters(model, spec: Union[ModelSpec, Dict[str, Any]]):
    """Inject every adapter the spec declares as LIVE peft modules (Stage B). Returns
    the model peft hands back. A spec with no adapters is a no-op."""
    if not isinstance(spec, ModelSpec):
        spec = ModelSpec.from_dict(spec)
    for adapter in spec.adapters:
        if adapter.type != "lora":
            raise ValueError(f"unknown adapter type {adapter.type!r}")
        cfg = adapter.config
        targets = (list(cfg["targets"]) if cfg.get("exact_targets")
                   else peft_targets(cfg["targets"]))
        model = inject_adapter_in_model(
            LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"],
                       target_modules=targets,
                       rank_pattern=dict(cfg.get("rank_pattern", {})),
                       alpha_pattern=dict(cfg.get("alpha_pattern", {}))),
            model, adapter_name=cfg.get("name", "default"))
    return model


def peft_targets(targets):
    """AdapterSpec targets -> peft's target_modules. The spec lists module suffixes
    (`attn.orig.to_q`, `token_refiner.refiner_blocks.*.attn.to_q`, ...); peft matches
    by suffix, so when the refiner is included the plain projection names reach both
    the DiT blocks (through `.attn.orig.`) and the refiner (`.attn.`). Without the
    refiner the DiT blocks are named explicitly so the refiner stays untouched."""
    if any(t.startswith("token_refiner") for t in targets):
        modules = ["to_q", "to_k", "to_v", "to_out.0"]
    else:
        modules = r"transformer_blocks\.\d+\.attn\.orig\.(to_q|to_k|to_v|to_out\.0)"

    return modules


def resolve_model_spec(model_section, loaded_config: dict) -> ModelSpec:
    """A1's resolution step: the YAML architecture section plus the
    LOADED base config in, a fully-resolved ModelSpec out. `linear_head_dim: null`
    resolves to the base's attention head dim HERE -- the spec never stores null."""
    section = (OmegaConf.to_container(model_section, resolve=True)
               if not isinstance(model_section, dict) else model_section)
    base_cfg = section["base"]
    ha = section["architecture"]["hybrid_attention"]
    lin = dict(ha["linear_attention"])
    if lin.get("linear_head_dim") is None:
        lin["linear_head_dim"] = int(loaded_config["attention_head_dim"])
    resolved = {k: loaded_config[k] for k in
                ("hidden_size", "num_layers", "num_attention_heads",
                 "attention_head_dim") if k in loaded_config}
    resolved["_load_dtype"] = base_cfg.get("dtype", "bfloat16")
    spec = ModelSpec(
        format_version=2,
        base=BaseSpec(library=base_cfg["library"], class_name=base_cfg["class_name"],
                      source=base_cfg["source"], subfolder=base_cfg["subfolder"],
                      revision=base_cfg.get("revision"), resolved_config=resolved),
        transforms=[TransformSpec("hybrid_attention", HYBRID_TRANSFORM_VERSION, dict(
            enable_softmax_gate=ha["enable_softmax_gate"],
            anchor_frames=ha["anchor_frames"],
            softmax_attention=dict(ha["softmax_attention"]),
            linear_attention=lin))],
    )
    return validate_spec(spec)


def _verify_base_config(model, base):
    """The spec's resolved_config is a subset-check against the loaded model's config:
    every stamped key must match. A missing stamp is fine (older spec, fewer keys);
    a MISMATCHED one means this is not the base the checkpoint was built on."""
    live = dict(model.config)
    for key, want in base.resolved_config.items():
        if key.startswith("_"):
            continue
        if key in live and live[key] != want:
            raise ValueError(f"base config mismatch on {key!r}: spec says {want!r}, "
                             f"loaded model says {live[key]!r}")


def load_model_weights(model, weights: Dict[str, torch.Tensor], strict: bool = True):
    """Load a v2 weights dict (transform parameters, and adapter weights when the
    matching adapters are already injected) into a canonical model. Refuses keys with
    no parameter by default."""
    params = dict(model.named_parameters())
    missing = [k for k in weights if k not in params]
    if missing and strict:
        raise RuntimeError(f"{len(missing)} weight keys have no parameter, e.g. "
                           f"{sorted(missing)[:4]}")
    loaded = 0
    for name, value in weights.items():
        if name not in params:
            continue
        p = params[name]
        p.data.copy_(value.to(dtype=p.dtype, device=p.device))
        loaded += 1
    return loaded

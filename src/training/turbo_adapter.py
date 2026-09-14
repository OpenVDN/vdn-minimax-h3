"""Live PEFT handling for Stage-DMD's frozen-teacher/trainable-student Turbo pair.

The external few-step adapter's weights are first translated by
``src.inference.utils.lora``. This module then reroots dense attention targets through
HybridAttention's ``attn.orig`` wrapper, injects two named adapters with the checkpoint's
mixed ranks, and keeps adapter switching independent of ``requires_grad`` after FSDP2 has
sharded the parameters.
"""
from collections import Counter

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer

from src.inference.utils.lora import load_external_lora


def reroot_external_lora_state(model, state):
    """Resolve dense LoRA module names against a hybrid model without changing bytes."""
    modules = dict(model.named_modules())
    rerooted = {}
    for name, tensor in state.items():
        marker = ".lora_A." if ".lora_A." in name else ".lora_B."
        if marker not in name:
            raise ValueError(f"external Turbo state contains a non-LoRA key: {name}")
        target, suffix = name.split(marker, 1)
        candidates = [target]
        if ".attn." in target:
            candidates.append(target.replace(".attn.", ".attn.orig.", 1))
        resolved = next((candidate for candidate in candidates if candidate in modules), None)
        if resolved is None:
            raise KeyError(f"Turbo target {target!r} has no module; tried {candidates}")
        rerooted[resolved + marker + suffix] = tensor
    return rerooted


def adapter_config_from_state(state, name="turbo"):
    """A fully resolved AdapterSpec config, including the adapter's per-module ranks."""
    ranks = {}
    for key, tensor in state.items():
        if ".lora_A." in key:
            ranks[key.split(".lora_A.", 1)[0]] = int(tensor.shape[0])
    if not ranks:
        raise ValueError("Turbo checkpoint contains no lora_A tensors")
    counts = Counter(ranks.values())
    default_rank = counts.most_common(1)[0][0]
    rank_pattern = {target: rank for target, rank in ranks.items()
                    if rank != default_rank}
    return {
        "name": name,
        "rank": default_rank,
        "alpha": default_rank,
        "targets": sorted(ranks),
        "rank_pattern": dict(sorted(rank_pattern.items())),
        # The adapter is defined as W_eff = W + B @ A, hence alpha == rank for every pair.
        "alpha_pattern": dict(sorted(rank_pattern.items())),
        "exact_targets": True,
    }


def extra_shard_module_paths(config):
    """LoRA target modules outside the block stacks already owned by FSDP2.

    An external adapter may target modules such as norm_out.linear in addition to the
    DiT/token-refiner blocks. Stage A/B never train those, so the shared sharder only
    wraps the two block stacks. Stage-DMD must explicitly shard every such outside target
    or its trainable LoRA parameters remain ordinary replicated tensors and cannot be
    safely checkpointed or resumed.
    """
    owned_prefixes = ("transformer_blocks.", "token_refiner.refiner_blocks.")
    return sorted({target for target in config["targets"]
                   if not target.startswith(owned_prefixes)})


def fake_adapter_config(turbo_config, rank, name="fake"):
    """The fake score's adapter: the Turbo checkpoint's exact target set at ONE uniform
    rank, alpha == rank (unit scale). PEFT zero-initialises lora_B, so the fake score
    equals the real score at step 0 -- DMD2's "initialise the fake from the teacher"
    without a second copy of anything."""
    if rank < 1:
        raise ValueError(f"fake adapter rank must be >= 1, got {rank}")
    return {
        "name": name,
        "rank": int(rank),
        "alpha": int(rank),
        "targets": list(turbo_config["targets"]),
        "rank_pattern": {},
        "alpha_pattern": {},
        "exact_targets": True,
    }


def load_external_adapter(model, checkpoint, name="turbo"):
    """Load and structurally resolve the external adapter, refusing scaled variants."""
    state, scale, _ = load_external_lora(checkpoint)
    if scale != 1.0:
        raise ValueError(f"Stage-DMD expects a unit LoRA scale (alpha == rank), got {scale}")
    state = reroot_external_lora_state(model, state)
    return state, adapter_config_from_state(state, name)


def inject_lora_adapter(model, config, adapter_name=None):
    """Inject one named, mixed-rank adapter described by ``adapter_config_from_state``."""
    adapter_name = adapter_name or config["name"]
    lora = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        target_modules=list(config["targets"]),
        rank_pattern=dict(config.get("rank_pattern", {})),
        alpha_pattern=dict(config.get("alpha_pattern", {})),
        lora_dropout=0.0,
        bias="none",
    )
    return inject_adapter_in_model(lora, model, adapter_name=adapter_name)


def copy_adapter_state(model, state, adapter_name):
    """Copy external ``.lora_[AB].weight`` tensors into one named PEFT adapter."""
    params = dict(model.named_parameters())
    copied = 0
    for name, tensor in state.items():
        if ".lora_A." in name:
            target = name.replace(".lora_A.", f".lora_A.{adapter_name}.")
        elif ".lora_B." in name:
            target = name.replace(".lora_B.", f".lora_B.{adapter_name}.")
        else:
            raise ValueError(f"not a LoRA tensor: {name}")
        if target not in params:
            raise KeyError(f"injected adapter is missing parameter {target}")
        if params[target].shape != tensor.shape:
            raise ValueError(f"shape mismatch for {target}: {params[target].shape} vs "
                             f"{tensor.shape}")
        params[target].data.copy_(tensor.to(params[target].dtype))
        copied += 1
    return copied


def set_active_adapters(model, adapter_names):
    """Switch forward participation only; never mutate FSDP-owned requires_grad flags."""
    active = list(adapter_names)
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module._disable_adapters = False
            module._active_adapter = active


def is_named_adapter_parameter(name, adapter_name):
    return (f".lora_A.{adapter_name}." in name
            or f".lora_B.{adapter_name}." in name)

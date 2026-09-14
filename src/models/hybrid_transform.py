"""The hybrid-attention transform: base H3 in, softmax+linear student out. This is
the ONLY place the architecture is constructed -- A1 reaches it through YAML,
A2/B/inference through a checkpoint's ModelSpec, all via build_model().

No diffusers source is modified: each DiT block's `.attn` is swapped for a
HybridAttention that wraps (and reuses) the original module, so the exact teacher is
always recoverable through `attn.orig`. The 2 token-refiner blocks are NOT converted:
they attend over text only, where there is no frame axis to window.
"""
import types

from src.models.hybrid_attention import HybridAttention
from src.models.ops.fused_block import fast_block_forward, fast_ff_forward

TRANSFORM_TYPE = "hybrid_attention"
TRANSFORM_VERSION = 2


def apply_hybrid_attention_transform(model, config: dict):
    """`config` is the ModelSpec transform config (nested v2 layout, RESOLVED values --
    see model_spec.hybrid_transform_spec). Returns the hybrid modules; their new
    parameters are the stage-A trainables."""
    soft = config["softmax_attention"]
    lin = dict(config["linear_attention"])
    short_conv = tuple(lin.pop("short_conv")["targets"])
    hidden_size = model.config.hidden_size
    hybrids = []
    for block in model.transformer_blocks:
        hybrid = HybridAttention(
            block.attn, hidden_size=hidden_size,
            delta_rule=lin["delta_rule"], linear_head_dim=lin["linear_head_dim"],
            bridge=lin["bridge"], a_fp32=lin["a_fp32"], short_conv=short_conv,
            enable_text_state=lin["enable_text_state"],
            radius=soft["radius"], chunk=soft["chunk"],
            anchor_frames=config["anchor_frames"],
            enable_softmax_gate=config["enable_softmax_gate"],
        )
        block.attn = hybrid
        hybrids.append(hybrid)
    return hybrids


def iter_hybrids(model):
    for block in model.transformer_blocks:
        if isinstance(block.attn, HybridAttention):
            yield block.attn


def set_layout(model, layout):
    for attn in iter_hybrids(model):
        attn.layout = layout


def set_teacher_mode(model, enabled: bool):
    for attn in iter_hybrids(model):
        attn.teacher_mode = enabled


def set_softmax_backend(model, backend: str) -> str:
    """Runtime window-softmax implementation (config ``kernels.softmax_backend``):
    ``auto`` (decomposed on every CUDA device; see decomposed.resolve_softmax_backend) |
    ``flex`` (BlockMask, what training uses) | ``decomposed`` (the mask as a union of
    dense calls) | ``ref`` (eager reference). Writes the resolved choice onto every
    hybrid layer and returns it. Never part of a checkpoint's spec."""
    from src.models.softmax_attention.decomposed import resolve_softmax_backend

    resolved = resolve_softmax_backend(backend)
    for attn in iter_hybrids(model):
        attn.softmax_impl = resolved
    return resolved


def set_inference_mode(model, enabled: bool):
    """Opt every hybrid layer in to the forward-only bodies, and install the fused
    block pointwise over every DiT block's forward. The CALLER states no graph will
    be built; the branch raises if that turns out false. Default is off, so a driver
    that forgets is slower, never wrong. Reentrant per weight-load on purpose -- the
    render entrypoint re-applies after every checkpoint swap (module swaps here are
    idempotent)."""
    for attn in iter_hybrids(model):
        attn.hybrid_inference_mode = enabled
        attn.inference_mode = enabled
    for block in model.transformer_blocks:
        ff = getattr(block, "ff", None)          # the tiny test blocks have no ff
        if enabled:
            block.forward = types.MethodType(fast_block_forward, block)
            if ff is not None:
                ff.forward = types.MethodType(fast_ff_forward, ff)
        else:
            block.__dict__.pop("forward", None)      # back to the class's own
            if ff is not None:
                ff.__dict__.pop("forward", None)


def set_hybrid_inference_mode(model, enabled: bool):
    """Enable only inference kernels intrinsic to HybridAttention.

    This is an attribution/ablation mode: the window softmax and linear far branch use
    their tuned inference bodies, while QK-norm + RoPE, block pointwise, and FF stay on
    their unfused implementations. Production inference should continue to call
    ``set_inference_mode(model, True)`` for the complete optimisation set.
    """
    set_inference_mode(model, False)
    for attn in iter_hybrids(model):
        attn.hybrid_inference_mode = enabled


def is_transform_parameter(name: str) -> bool:
    """Is this parameter NAME one the hybrid transform introduced? True for the
    branch, the gates and to_out_linear on a DiT block; False for the inherited
    softmax weights (`.attn.orig.*`), the token refiner's own attention (never
    converted) and LoRA adapters.

    Name-based on purpose, next to the structural `hybrid_new_parameters`: the FSDP2
    stages need this AFTER fully_shard has swapped every Parameter object and on
    checkpoint keys that have no object at all, and names survive both. It is the one
    spelling of that rule -- the trainers' trainable set, weight saves and train
    states all read it here."""
    return (".attn." in name and ".attn.orig." not in name
            and "token_refiner" not in name and "lora_" not in name)


def hybrid_new_parameters(model):
    """Parameters the transform introduced, per block -- the stage-A trainables and
    exactly what a v2 weights artifact stores. Inherited softmax weights (orig.*)
    are deliberately absent."""
    out = []
    for index, block in enumerate(model.transformer_blocks):
        attn = block.attn
        if not isinstance(attn, HybridAttention):
            continue
        named = [(f"transformer_blocks.{index}.attn.{name}", p)
                 for name, p in attn.named_parameters()
                 if not name.startswith("orig.")]
        out.append((index, named))
    return out


def transform_config_of_model(model) -> dict:
    """Read the RESOLVED transform config back off a converted model -- what the
    checkpoint writer stamps into the ModelSpec. The inverse of
    apply_hybrid_attention_transform."""
    attn = next(iter_hybrids(model), None)
    if attn is None:
        raise ValueError("model carries no HybridAttention -- nothing to stamp")
    lin = attn.linear_attention
    targets = list(lin.short_conv.projs) if lin.short_conv is not None else []
    return dict(
        enable_softmax_gate=attn.enable_softmax_gate,
        anchor_frames=attn.anchor_frames,
        softmax_attention=dict(radius=attn.radius, chunk=attn.chunk),
        linear_attention=dict(delta_rule=lin.delta_rule, bridge=lin.bridge,
                              linear_head_dim=lin.head_dim, a_fp32=lin.a_fp32,
                              short_conv=dict(targets=targets),
                              enable_text_state=attn.enable_text_state),
    )

"""Inference overlay config. Four families, four sections: runtime optimization
(kernels), precision modification (fp8/dtype), multi-GPU layout (parallel, honoured by
infer_ulysses.py only; infer.py refuses non-default values), semantic ablation -- and
the last is gated: any semantic override with ablation.enabled false is a load-time
error, not a warning.

Every knob is a field here (YAML + dotlist); nothing on the inference path reads an
environment variable (FA_CLC is flash-attn's own). Arch-dependent choices resolve at
the setter, not the config: `softmax_backend: auto` is the decomposed window kernel
on sm100 and flex on sm90, fp8 scale granularity follows the capability -- one YAML
for both clusters."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from src.config.common import KernelsConfig

# Mirror of decomposed.SOFTMAX_BACKENDS so a typo dies before a 33B model loads.
SOFTMAX_BACKENDS = ("auto", "flex", "decomposed", "ref")


@dataclass
class InferenceKernels(KernelsConfig):
    # softmax_backend (inherited): auto | flex | decomposed | ref -- WHICH window-softmax
    # kernel. auto = decomposed on sm100, flex on sm90 (hybrid_transform.set_softmax_backend).
    #
    # inference_kernels: set_inference_mode(model, True) -- the forward-only kernel set,
    # one switch, same arithmetic:
    #   attention   fused QK-norm + RoPE in one pass, q/k/v consumed as views of the
    #               packed [T,H,d] buffer; the window softmax on the static
    #               FLASH/CuteDSL flex variant instead of the dynamic Triton one; the
    #               softmax gate fused with the to_out repack
    #   far branch  one-kernel frame-statistics prologue; Triton 5-tap temporal conv;
    #               compiled RMSNorm + output gate epilogue; one triangular solve for
    #               the vdn inverse; readout in the bmm-native layout; cached gather
    #               indices; no anchor zero-fill; baddbmm scan updates; TF32 on the one
    #               fp32 GEMM
    #   block       RMSNorm + AdaLN affine and gate + residual as two compiled kernels;
    #               SwiGLU as one compiled kernel
    # False = the released arithmetic spelled eagerly (rung 0 of the yaml ladder).
    inference_kernels: bool = True


@dataclass
class Fp8Config:
    enabled: bool = False                # NEVER default-on: fp8 changes the sample
    skip_end_blocks: int = 4             # scale granularity is per arch, not a knob
    keep_original: bool = True           # false drops replaced BF16 weights (not reversible)


@dataclass
class PrecisionConfig:
    dtype: str = "bfloat16"
    fp8: Fp8Config = field(default_factory=Fp8Config)


@dataclass
class ParallelConfig:
    """torchrun layout for infer_ulysses.py."""
    softmax_ranks: int = 6               # 0 = standard Ulysses; n = n softmax + (world-n) linear
    profile: bool = False                # per-section CUDA events: profiling only


@dataclass
class WorkerConfig:
    """Optional multi-request mode for ``infer_ulysses.py``.

    The request file is JSONL. Each row overrides fields under ``render`` and must
    provide a distinct ``prompt_file`` and ``out``. With CPU offload enabled, rank 0
    swaps the already-assembled transformer out while decoding, then restores it;
    the other ranks remain resident throughout.
    """
    requests_file: Optional[str] = None
    decoder_cpu_offload: bool = False
    # 0 swaps the full transformer. A positive value swaps only that many trailing
    # transformer blocks, reducing PCIe traffic when the remaining weights and VAE fit.
    transformer_cpu_offload_blocks: int = 0


@dataclass
class BehaviorConfig:
    teacher_mode: bool = False


@dataclass
class AblationConfig:
    enabled: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderConfig:
    prompt_file: str = MISSING
    out: str = MISSING
    num_frames: int = 345
    num_steps: int = 50
    warmup_steps: int = 0                # NFE run and discarded first: 2 for any timing
                                         # (compile + the 2nd-timestep re-specialisation)
    seed: int = 42
    device: str = "cuda:0"
    video_shift: float = 12.0            # scheduler shifts; a distilled adapter is only
    audio_shift: float = 3.0             # valid at the shifts it was trained for
    save_latents: bool = False           # optional parity/debug tensors live outside
    latents_root: str = "artifacts/latents"  # results/: JSON/JSONL + MP4 only
    record: bool = False                 # also write <out>.inference.json (checkpoint
                                         # identity, resolved config, actual kernel
                                         # state, per-NFE timings) -- a debug aid


@dataclass
class ExternalLora:
    """A community adapter (safetensors, peft names against the DENSE model), merged
    after the checkpoint's own LoRA. alpha=None reads the file's metadata (missing
    there too -> alpha=rank, scale 1.0)."""
    path: str = MISSING
    alpha: Optional[float] = None


@dataclass
class InferenceConfig:
    checkpoint: Optional[str] = None     # None = render the DENSE base model
    base_source: Optional[str] = None    # base weights path override; identity stays with the spec
    vae_source: Optional[str] = None     # the decoders' root; None = the release copy (paths.H3_BASE)
    external_loras: List[ExternalLora] = field(default_factory=list)
    render: RenderConfig = field(default_factory=RenderConfig)
    kernels: InferenceKernels = field(default_factory=InferenceKernels)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


def validate_ablation(cfg) -> None:
    if cfg.ablation.overrides and not cfg.ablation.enabled:
        raise ValueError(
            f"semantic overrides {sorted(cfg.ablation.overrides)} require "
            f"ablation.enabled=true -- they change what the model computes"
        )


def validate_kernels(cfg) -> None:
    if cfg.kernels.softmax_backend not in SOFTMAX_BACKENDS:
        raise ValueError(f"kernels.softmax_backend {cfg.kernels.softmax_backend!r} "
                         f"not in {SOFTMAX_BACKENDS}")
    if cfg.render.warmup_steps < 0:
        raise ValueError("render.warmup_steps must be >= 0")
    if cfg.precision.fp8.skip_end_blocks < 0:
        raise ValueError("precision.fp8.skip_end_blocks must be >= 0")
    if cfg.worker.transformer_cpu_offload_blocks < 0:
        raise ValueError("worker.transformer_cpu_offload_blocks must be >= 0")


def validate_parallel(cfg) -> None:
    """infer_ulysses.py: the layout is checked before the process group exists; the
    world-size bound (softmax_ranks < WORLD_SIZE) is checked at install."""
    if cfg.parallel.softmax_ranks < 0:
        raise ValueError("parallel.softmax_ranks must be >= 0 (0 = standard Ulysses)")


def validate_single_process(cfg) -> None:
    """infer.py: a parallel.* override would silently do nothing on the single-GPU
    entrypoint, so it is refused."""
    if OmegaConf.to_container(cfg.parallel) != asdict(ParallelConfig()):
        raise ValueError("parallel.* only applies to src/inference/infer_ulysses.py "
                         "(torchrun); infer.py is single-process")
    if cfg.worker.requests_file:
        raise ValueError("worker.* only applies to src/inference/infer_ulysses.py "
                         "(torchrun)")

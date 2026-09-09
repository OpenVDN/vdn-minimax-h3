"""Dedicated multi-GPU, inference-only Ulysses entrypoint.

Launch with ``torchrun --nproc_per_node=8 src/inference/infer_ulysses.py
--config configs/inference/{8,50}nfe_tuned_fp8_ulysses_{h200,b200}.yaml ...``. The ordinary ``src/inference/infer.py``
path is intentionally untouched. The rank layout, optimisation ladder, profiling and
warm-up are the ``parallel.*`` / ``render.warmup_steps`` config fields -- no
environment variable is read beyond torchrun's own LOCAL_RANK/WORLD_SIZE.

t2va, i2va and fl2va all run here: the conditioning rows a keyframe cache adds sit
outside the layout's video span, so they shard, gather and attend as ordinary global
rows and no collective changes shape.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist

from src.config import load_config
from src.config.inference import (
    InferenceConfig,
    validate_ablation,
    validate_kernels,
    validate_parallel,
)
from src.inference.utils.assemble import (
    build_inference_model,
    latents_path,
    render_record,
    write_json,
)
from src.inference.render import decode_and_save, generate_latents, load_prompt
from src.inference.utils.ulysses import init_ulysses, install_ulysses


def main():
    process_started = time.perf_counter()
    cfg = load_config(
        InferenceConfig,
        extra_validators=[validate_ablation, validate_kernels, validate_parallel],
    )
    if not cfg.checkpoint:
        raise ValueError("the dedicated Ulysses path requires a hybrid checkpoint")
    if cfg.behavior.teacher_mode:
        raise ValueError("teacher_mode is not supported by inference-only Ulysses")

    runtime = init_ulysses(profile_enabled=cfg.parallel.profile)
    device = runtime.device
    torch.set_grad_enabled(False)

    model = build_inference_model(
        cfg, device, load_decoders=runtime.is_main, log=runtime.is_main
    )
    if not model.is_hybrid:
        raise ValueError("Ulysses path currently requires a HybridAttention checkpoint")

    install_ulysses(model.transformer, runtime, softmax_ranks=cfg.parallel.softmax_ranks)
    if runtime.is_main and runtime.branch_parallel:
        print(
            f"branch-parallel Ulysses: {runtime.softmax_ranks} softmax ranks + "
            f"{runtime.world_size - runtime.softmax_ranks} linear ranks; "
            f"QKV projected once on sequence owners",
            flush=True,
        )
    elif runtime.is_main:
        print(
            f"standard Ulysses: {runtime.world_size} ranks, every rank owns heads of "
            "both branches",
            flush=True,
        )

    prompt_embeds, text_token_tags, conditions = load_prompt(cfg.render.prompt_file, str(device))
    if conditions and runtime.is_main:
        print(f"keyframes anchored {conditions[0]}", flush=True)
    runtime.barrier()
    torch.cuda.synchronize(device)
    model_setup_seconds = time.perf_counter() - process_started

    def sample(num_steps, step_seconds=None):
        return generate_latents(
            model.transformer,
            prompt_embeds,
            text_token_tags,
            cfg.render.num_frames,
            num_steps,
            cfg.render.seed,
            device,
            video_shift=cfg.render.video_shift,
            audio_shift=cfg.render.audio_shift,
            runtime=runtime,
            step_seconds=step_seconds,
            conditions=conditions,
        )

    warmup_steps = cfg.render.warmup_steps
    if warmup_steps:
        if runtime.is_main:
            print(f"warming up {warmup_steps} NFE in the measured process", flush=True)
        sample(warmup_steps)

    runtime.reset_profile()

    step_seconds: list[float] = []
    denoise_started = time.perf_counter()
    latents, audio_latents = sample(cfg.render.num_steps, step_seconds)
    denoise_seconds = time.perf_counter() - denoise_started

    timings = {
        "denoise_seconds": denoise_seconds,
        "seconds_per_step": denoise_seconds / cfg.render.num_steps,
        "step_seconds": step_seconds,
        "model_setup_seconds": model_setup_seconds,
    }
    local_profile = {
        name: milliseconds / cfg.render.num_steps
        for name, milliseconds in runtime.profile_milliseconds().items()
    }
    profile_by_rank = [None] * runtime.world_size
    dist.all_gather_object(profile_by_rank, local_profile)
    timings["parallel_profile_ms_per_nfe_by_rank"] = profile_by_rank

    if runtime.is_main and cfg.render.save_latents:
        lpath = latents_path(cfg)
        os.makedirs(os.path.dirname(lpath) or ".", exist_ok=True)
        torch.save({"video": latents.cpu(), "audio": audio_latents.cpu()}, lpath)

    decode_started = time.perf_counter()
    if runtime.is_main:
        decode_and_save(
            latents, audio_latents, model.vae, model.audio_vae, cfg.render.out, str(device)
        )
        torch.cuda.synchronize(device)

    runtime.barrier()
    timings["decode_and_encode_seconds"] = time.perf_counter() - decode_started
    timings["end_to_end_seconds"] = time.perf_counter() - process_started

    if runtime.is_main:
        record = render_record(cfg, model)
        record["parallel"] = {
            "kind": "ulysses_branch_parallel" if runtime.branch_parallel else "ulysses",
            "world_size": runtime.world_size,
            "backend": runtime.backend,
            "sequence_splits": list(runtime.splits),
            "heads_per_rank": runtime.heads_per_rank,
            "softmax_ranks": runtime.softmax_ranks,
            "softmax_head_splits": list(runtime.softmax_head_splits),
            "linear_head_splits": list(runtime.linear_head_splits),
            "warmup_steps": warmup_steps,
        }
        record["timings"] = timings

        if cfg.render.record:
            json_path = cfg.render.out + ".inference.json"
            write_json(record, json_path)
            print(f"wrote {json_path}", flush=True)

        print(
            "Ulysses timing: "
            f"setup {timings['model_setup_seconds']:.2f}s, "
            f"denoise {timings['denoise_seconds']:.2f}s "
            f"({timings['seconds_per_step']:.2f}s/step; per NFE "
            + " ".join(f"{x:.2f}" for x in timings["step_seconds"]) + "), "
            f"decode+encode {timings['decode_and_encode_seconds']:.2f}s, "
            f"E2E {timings['end_to_end_seconds']:.2f}s",
            flush=True,
        )

        if runtime.profile_enabled:
            for rank, profile in enumerate(profile_by_rank):
                fields = ", ".join(
                    f"{name}={milliseconds:.1f}ms"
                    for name, milliseconds in sorted(profile.items())
                )
                print(f"profile rank {rank}: {fields}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

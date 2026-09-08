"""Dedicated multi-GPU, inference-only Ulysses entrypoint.

Launch with ``torchrun --nproc_per_node=8 src/inference/infer_ulysses.py
--config configs/inference/{8,50}nfe_tuned_fp8_ulysses_{h200,b200}.yaml ...``. The ordinary ``src/inference/infer.py``
path is intentionally untouched. The rank layout, optimisation ladder, profiling and
warm-up are the ``parallel.*`` / ``render.warmup_steps`` config fields -- no
environment variable is read beyond torchrun's own LOCAL_RANK/WORLD_SIZE.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.config import load_config
from src.config.inference import (
    InferenceConfig,
    validate_ablation,
    validate_kernels,
    validate_parallel,
)
from src.inference.assemble import (
    build_inference_model,
    latents_path,
    render_record,
    write_json,
)
from src.inference.render import decode_and_save, generate_latents, load_decoders, load_text
from src.inference.ulysses import init_ulysses, install_ulysses


_REQUEST_FIELDS = {
    "prompt_file", "out", "num_frames", "num_steps", "seed", "video_shift",
    "audio_shift", "save_latents", "latents_root", "record",
}


def load_requests(cfg):
    """Return one resolved config per render request.

    Keeping request parsing independent of distributed state makes malformed queues
    fail before a rank enters a collective. Every rank reads the same immutable file.
    """
    path = cfg.worker.requests_file
    if not path:
        return [cfg]

    requests = []
    outputs = set()
    with open(path) as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: request must be a JSON object")
            unknown = set(item) - _REQUEST_FIELDS
            if unknown:
                raise ValueError(
                    f"{path}:{line_number}: unknown render fields {sorted(unknown)}")
            missing = {"prompt_file", "out"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            if item["out"] in outputs:
                raise ValueError(f"{path}:{line_number}: duplicate out {item['out']!r}")
            outputs.add(item["out"])
            requests.append(OmegaConf.merge(cfg, {"render": item}))

    if not requests:
        raise ValueError(f"{path}: request file is empty")
    return requests


def move_module(module, device):
    module.to(device)
    return module


def transformer_offload_modules(transformer, count):
    """Select the whole transformer or a trailing block subset for decoder offload."""
    if count == 0:
        return [transformer]
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if count > len(blocks):
        raise ValueError(
            f"requested {count} offload blocks, transformer has {len(blocks)}")
    return blocks[-count:]


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
    requests = load_requests(cfg)
    persistent = bool(cfg.worker.requests_file)

    model = build_inference_model(
        cfg,
        device,
        load_decoders=(runtime.is_main
                       and (not persistent or not cfg.worker.decoder_cpu_offload)),
        log=runtime.is_main,
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

    if runtime.is_main and persistent and cfg.worker.decoder_cpu_offload:
        model.vae, model.audio_vae = load_decoders(cfg.vae_source, "cpu")
        offload_modules = transformer_offload_modules(
            model.transformer, cfg.worker.transformer_cpu_offload_blocks)
    else:
        offload_modules = []

    first_cfg = requests[0]
    prompt_embeds, text_token_tags = load_text(first_cfg.render.prompt_file, str(device))
    runtime.barrier()
    torch.cuda.synchronize(device)
    model_setup_seconds = time.perf_counter() - process_started

    def sample(request_cfg, num_steps, step_seconds=None):
        return generate_latents(
            model.transformer,
            prompt_embeds,
            text_token_tags,
            request_cfg.render.num_frames,
            num_steps,
            request_cfg.render.seed,
            device,
            video_shift=request_cfg.render.video_shift,
            audio_shift=request_cfg.render.audio_shift,
            runtime=runtime,
            step_seconds=step_seconds,
        )

    warmup_steps = cfg.render.warmup_steps
    if warmup_steps:
        if runtime.is_main:
            print(f"warming up {warmup_steps} NFE in the measured process", flush=True)
        sample(first_cfg, warmup_steps)

    del prompt_embeds, text_token_tags

    for request_index, request_cfg in enumerate(requests):
        request_started = time.perf_counter()
        prompt_embeds, text_token_tags = load_text(
            request_cfg.render.prompt_file, str(device))
        runtime.reset_profile()

        step_seconds: list[float] = []
        denoise_started = time.perf_counter()
        latents, audio_latents = sample(
            request_cfg, request_cfg.render.num_steps, step_seconds)
        denoise_seconds = time.perf_counter() - denoise_started

        timings = {
            "request_index": request_index,
            "denoise_seconds": denoise_seconds,
            "seconds_per_step": denoise_seconds / request_cfg.render.num_steps,
            "step_seconds": step_seconds,
            "model_setup_seconds": model_setup_seconds if request_index == 0 else 0.0,
        }
        local_profile = {
            name: milliseconds / request_cfg.render.num_steps
            for name, milliseconds in runtime.profile_milliseconds().items()
        }
        profile_by_rank = [None] * runtime.world_size
        dist.all_gather_object(profile_by_rank, local_profile)
        timings["parallel_profile_ms_per_nfe_by_rank"] = profile_by_rank

        if runtime.is_main and request_cfg.render.save_latents:
            lpath = latents_path(request_cfg)
            os.makedirs(os.path.dirname(lpath) or ".", exist_ok=True)
            torch.save({"video": latents.cpu(), "audio": audio_latents.cpu()}, lpath)

        decode_started = time.perf_counter()
        if runtime.is_main:
            swap_out_seconds = 0.0
            swap_back_seconds = 0.0
            if persistent and cfg.worker.decoder_cpu_offload:
                swap_started = time.perf_counter()
                for module in offload_modules:
                    move_module(module, "cpu")
                torch.cuda.empty_cache()
                move_module(model.vae, device)
                move_module(model.audio_vae, device)
                torch.cuda.synchronize(device)
                swap_out_seconds = time.perf_counter() - swap_started
            decode_and_save(
                latents, audio_latents, model.vae, model.audio_vae,
                request_cfg.render.out, str(device)
            )
            torch.cuda.synchronize(device)

            if persistent and cfg.worker.decoder_cpu_offload:
                swap_started = time.perf_counter()
                move_module(model.vae, "cpu")
                move_module(model.audio_vae, "cpu")
                torch.cuda.empty_cache()
                for module in offload_modules:
                    move_module(module, device)
                torch.cuda.synchronize(device)
                swap_back_seconds = time.perf_counter() - swap_started

            timings["decoder_swap_out_seconds"] = swap_out_seconds
            timings["decoder_swap_back_seconds"] = swap_back_seconds

        runtime.barrier()
        timings["decode_and_encode_seconds"] = time.perf_counter() - decode_started
        timings["request_seconds"] = time.perf_counter() - request_started

        if runtime.is_main:
            record = render_record(request_cfg, model)
            record["parallel"] = {
                "kind": "ulysses_branch_parallel" if runtime.branch_parallel else "ulysses",
                "world_size": runtime.world_size,
                "backend": runtime.backend,
                "sequence_splits": list(runtime.splits),
                "heads_per_rank": runtime.heads_per_rank,
                "softmax_ranks": runtime.softmax_ranks,
                "softmax_head_splits": list(runtime.softmax_head_splits),
                "linear_head_splits": list(runtime.linear_head_splits),
                "warmup_steps": warmup_steps if request_index == 0 else 0,
                "persistent_worker": persistent,
                "decoder_cpu_offload": bool(cfg.worker.decoder_cpu_offload),
                "transformer_cpu_offload_blocks": int(
                    cfg.worker.transformer_cpu_offload_blocks),
            }
            record["timings"] = timings

            if request_cfg.render.record:
                json_path = request_cfg.render.out + ".inference.json"
                write_json(record, json_path)
                print(f"wrote {json_path}", flush=True)

            print(
                f"Ulysses request {request_index}: "
                f"denoise {timings['denoise_seconds']:.2f}s "
                f"({timings['seconds_per_step']:.2f}s/step; per NFE "
                + " ".join(f"{x:.2f}" for x in timings["step_seconds"]) + "), "
                f"decode path {timings['decode_and_encode_seconds']:.2f}s, "
                f"request {timings['request_seconds']:.2f}s",
                flush=True,
            )

            if runtime.profile_enabled:
                for rank, profile in enumerate(profile_by_rank):
                    fields = ", ".join(
                        f"{name}={milliseconds:.1f}ms"
                        for name, milliseconds in sorted(profile.items())
                    )
                    print(f"profile rank {rank}: {fields}", flush=True)

        del prompt_embeds, text_token_tags, latents, audio_latents

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

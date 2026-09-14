"""The v2 single-GPU inference entrypoint.

    python src/inference/infer.py --config configs/inference/50nfe.yaml \
        checkpoint=... render.prompt_file=... render.out=...

Assembly (spec -> base -> transform -> weights -> LoRAs -> overlays -> fp8) is
src/inference/utils/assemble.py, shared with infer_ulysses.py; this file only renders. Every
knob is a config field (YAML + dotlist, src/config/inference.py); nothing reads an
environment variable. `checkpoint: null` renders the released DENSE model. Every render
can also write <out>.inference.json (`render.record=true`) with the checkpoint identity,
the resolved config and the ACTUAL kernel state -- a debug aid, off by default.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from src.config import load_config
from src.config.inference import (InferenceConfig, validate_ablation, validate_kernels,
                                  validate_single_process)
from src.inference.utils.assemble import (build_inference_model, latents_path, render_record,
                                    write_json)
from src.inference.render import decode_and_save, generate_latents, load_prompt


def main():
    cfg = load_config(InferenceConfig, extra_validators=[
        validate_ablation, validate_kernels, validate_single_process])
    device = cfg.render.device
    torch.set_grad_enabled(False)

    model = build_inference_model(cfg, device)
    prompt_embeds, text_token_tags, conditions = load_prompt(cfg.render.prompt_file, device)
    if conditions:
        print(f"keyframes anchored {conditions[0]}", flush=True)

    if cfg.render.warmup_steps:
        print(f"warming up {cfg.render.warmup_steps} NFE (discarded)", flush=True)

        generate_latents(
            model.transformer, prompt_embeds, text_token_tags,
            cfg.render.num_frames, cfg.render.warmup_steps, cfg.render.seed, device,
            video_shift=cfg.render.video_shift, audio_shift=cfg.render.audio_shift,
            conditions=conditions)

    step_seconds: list[float] = []
    latents, audio_latents = generate_latents(
        model.transformer, prompt_embeds, text_token_tags,
        cfg.render.num_frames, cfg.render.num_steps, cfg.render.seed, device,
        video_shift=cfg.render.video_shift, audio_shift=cfg.render.audio_shift,
        step_seconds=step_seconds, conditions=conditions)

    denoise = sum(step_seconds)
    print(f"timing: denoise {denoise:.2f}s ({denoise / len(step_seconds):.2f}s/NFE; per NFE "
          + " ".join(f"{x:.2f}" for x in step_seconds) + ")", flush=True)

    if cfg.render.save_latents:
        lpath = latents_path(cfg)
        os.makedirs(os.path.dirname(lpath) or ".", exist_ok=True)
        torch.save({"video": latents.cpu(), "audio": audio_latents.cpu()}, lpath)
        print(f"wrote {lpath}", flush=True)

    decode_and_save(latents, audio_latents, model.vae, model.audio_vae, cfg.render.out,
                    device)

    if cfg.render.record:
        jpath = cfg.render.out + ".inference.json"
        record = render_record(cfg, model)
        record["timings"] = {"step_seconds": step_seconds}
        write_json(record, jpath)
        print(f"wrote {jpath}", flush=True)


if __name__ == "__main__":
    main()

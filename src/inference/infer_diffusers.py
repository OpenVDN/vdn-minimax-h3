"""Render through the PUBLISHED diffusers components, with nothing from this repository.

    python src/inference/infer_diffusers.py "a prompt" --out results/diffusers.mp4
    python src/inference/infer_diffusers.py "a prompt" --steps 50 \
        --transformer stage-b-step-2000/diffusers
    python src/inference/infer_diffusers.py "a prompt" \
        --first prompts/image/first.png --last prompts/image/last.png

This is the README's `Load it with Diffusers` snippet as a runnable file: it imports
`diffusers` and nothing else, so it works from a bare `pip install diffusers` checkout
of nowhere in particular. It deliberately shares NO code with infer.py and
infer_ulysses.py -- those are the fast stack (fp8, the decomposed window kernel,
Ulysses); this is the portable one, single-GPU bf16.

Two things are not optional on one GPU. `workflow=` keeps the unused 61.7 GB
transformer partition from being fetched, and the offload is what lets the 66 GB
transformer and the 66 GB Qwen3-VL text encoder share a card: whichever is not running
sits on the CPU.
"""
import argparse

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video

REPO = "OpenVDN/vdn-minimax-h3"
FPS = 24


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompt")
    p.add_argument("--out", default="results/diffusers.mp4")
    p.add_argument("--steps", type=int, default=8,
                   help="model evaluations (NFE). The scheduler counts sigma grid "
                        "points, one more than that, and this passes the +1 for you")
    p.add_argument("--frames", type=int, default=124,
                   help="snapped up to the next 17n+5; 5 to 15 seconds at 24 fps")
    p.add_argument("--transformer", default=None,
                   help="a checkpoint's diffusers/ subfolder. Default: whatever the "
                        "repository's index names, the 8-step model")
    p.add_argument("--first", help="keyframe the video starts from")
    p.add_argument("--last", help="keyframe the video ends on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    keyframes = {}
    if args.first or args.last:
        from diffusers.utils import load_image

        if args.first:
            keyframes["image"] = load_image(args.first)
        if args.last:
            keyframes["last_image"] = load_image(args.last)

    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        REPO, workflow="fl2va" if keyframes else "t2va",
        components_manager=manager, collection="vdn")

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if args.transformer:
        load_kwargs["subfolder"] = {"transformer": args.transformer}
    pipe.load_components(**load_kwargs)
    manager.enable_auto_cpu_offload(device=args.device)

    videos, audio, rate = pipe(
        prompt=args.prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps + 1,
        generator=torch.Generator(args.device).manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
        **keyframes,
    ).values()

    import numpy as np

    frames = torch.from_numpy(np.stack([np.asarray(frame) for frame in videos[0]]))
    encode_video(frames, fps=FPS, output_path=args.out,
                 audio=audio[0].float().cpu(), audio_sample_rate=rate)
    print(f"wrote {args.out}: {len(videos[0])} frames, "
          f"{audio.shape[-1] / rate:.2f}s of audio")


if __name__ == "__main__":
    main()

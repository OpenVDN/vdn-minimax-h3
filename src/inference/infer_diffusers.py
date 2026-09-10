"""Render through the PUBLISHED diffusers components, not this repository's stack.

    python src/inference/infer_diffusers.py                       # prompts/example_0.pt
    python src/inference/infer_diffusers.py "a prompt" --out results/diffusers.mp4
    python src/inference/infer_diffusers.py "a prompt" --steps 50 \
        --transformer stage-b-step-2000/diffusers
    python src/inference/infer_diffusers.py "a prompt" \
        --first prompts/image/first.png --last prompts/image/last.png

This is the README's `Load it with Diffusers` snippet as a runnable file. Everything it
renders comes from the Hub, so `pip install diffusers` is the whole setup; the only
thing it reads from here is the default prompt, and a prompt of your own replaces that.
It deliberately shares NO code with infer.py and
infer_ulysses.py -- those are the fast stack (fp8, the decomposed window kernel,
Ulysses); this is the portable one, single-GPU bf16.

Two things are not optional on one GPU. `workflow=` keeps the unused 61.7 GB
transformer partition from being fetched, and the offload is what lets the 62 GB
Qwen3-VL text encoder and the 66 GB transformer share a card: whichever is not running
sits on the CPU.

`memory_reserve_margin` is part of that offload rather than a tuning knob. The strategy
asks only whether the incoming model's *weights* fit, and at the 3 GB default both do
fit on a 140 GB card at once -- so it offloads nothing, ever, and denoising starts with
10 GB of room and dies in the first block. 40 GB is what sends the text encoder back to
the CPU before the transformer runs; 345 frames then peak at 85 GB.
"""
import argparse
import os

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video

REPO = "OpenVDN/vdn-minimax-h3"
FPS = 24
# The repository's own showcase prompt. Every prompt cache carries the text it was
# encoded from, so the default is that text rather than a second copy of it here.
DEFAULT_PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "prompts", "example_0.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompt", nargs="?",
                   help=f"defaults to the text {os.path.basename(DEFAULT_PROMPT)} was "
                        "encoded from")
    p.add_argument("--out", default="results/diffusers.mp4")
    p.add_argument("--steps", type=int, default=8,
                   help="model evaluations (NFE). The scheduler counts sigma grid "
                        "points, one more than that, and this passes the +1 for you")
    p.add_argument("--frames", type=int, default=345,
                   help="snapped up to the next 17n+5; 5 to 15 seconds at 24 fps. The "
                        "default is the 14.4 seconds the reported numbers use, which "
                        "one 140 GB GPU holds in bf16 with room to spare")
    p.add_argument("--transformer", default=None,
                   help="a checkpoint's diffusers/ subfolder. Default: whatever the "
                        "repository's index names, the 8-step model")
    p.add_argument("--first", help="keyframe the video starts from")
    p.add_argument("--last", help="keyframe the video ends on")
    p.add_argument("--fp8", action="store_true",
                   help="every wide Linear in fp8 e4m3: the weights drop from 62 GB to "
                        "43 and the GEMMs roughly double. Changes the sample -- this "
                        "seed will not reproduce the bf16 render")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    prompt = args.prompt or torch.load(DEFAULT_PROMPT, map_location="cpu",
                                       weights_only=True)["prompt"]

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
    if args.fp8:
        # A dict keys a kwarg to one component; the text encoder would not know it.
        load_kwargs["fp8"] = {"transformer": True}
    pipe.load_components(**load_kwargs)
    manager.enable_auto_cpu_offload(device=args.device, memory_reserve_margin="40GB")

    videos, audio, rate = pipe(
        prompt=prompt,
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

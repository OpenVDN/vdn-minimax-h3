"""Encode a keyframe request -- prompt + first and/or last keyframe -- and cache it as
one prompt .pt that infer.py renders like any t2va cache, plus the keyframe
conditioning. Which keyframes you pass is the mode: `--first` alone is i2va, `--last`
alone is l2va, both is fl2va.

    python src/inference/encode_keyframes.py --prompt "..." \\
        --first first.png --last last.png --out prompts/mine.pt

Replicates the diffusers `fl2va` presentation exactly (MiniMaxH3ResizeStep,
MiniMaxH3FL2VATextEncoderStep, MiniMaxH3KeyframeVaeEncoderStep): the first keyframe is
stretched onto the 768-short-edge canvas of its own aspect ratio, the last is
cover-cropped onto it; the presentation is `"<Picture i>: " + vision block` per keyframe
(vision rows tagged as VIDEO) then the prompt verbatim; the Qwen3-VL hidden state after
layer 50 conditions the transformer; each keyframe is VAE-encoded under the fixed seed 42
posterior sample, rounded to fp16, and normalised. The .pt carries:

    prompt, prompt_embeds (L, 5120) bf16, text_token_tags (L,), keyframe_anchors
    ["first", "last"], condition_latents [(1, 24, 1, H/16, W/16) fp32 ...], height, width
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

from src.inference.render import PIXEL_MEAN, PIXEL_STD
from src.paths import H3_BASE, resolve_weights, upstream_snapshot

TEXT_ENCODER_LAYER = 50
VIDEO_TAG, TEXT_TAG = 0, 1
CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS = 32, 768, 768 * 1344
KEYFRAME_ENCODE_SEED = 42

# (anchors) -> mode, a substring of its instruction, the instruction. MiniMax-H3 expects
# the prompt to open with the line for its mode (VIDEO_PROMPT_WRITING_GUIDE_base_en.md);
# i2va's is fixed, the other two name the final shot and the duration.
KEYFRAME_MODES = {
    ("first",): ("i2va", "is fully referenced",
                 "For the target video, at 0.00 seconds into the target video, "
                 "<Picture 1> (from [Shot 1]) is fully referenced."),
    ("last",): ("l2va", "How the reference pictures align",
                "How the reference pictures align with the target video \u2014 <Picture 1> "
                "(from [Shot N]) aligns with the S.SS-second mark of the target video."),
    ("first", "last"): ("fl2va", "How the reference pictures align",
                        "How the reference pictures align with the target video \u2014 Picture 1 "
                        "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                        "Picture 2 (from Shot N) aligns with the S.SS-second mark of the "
                        "target video."),
}


def put_on_canvas(keyframes):
    """MiniMaxH3ResizeStep: the canvas follows the first keyframe's aspect ratio; the
    first keyframe is stretched onto it, every follower cover-cropped (the released
    model's rounding, not VaeImageProcessor's). Returns (keyframes, height, width)."""
    height, width = resolve_canvas_size(*keyframes[0].size, CANVAS_MULTIPLE,
                                        CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS)
    prepared = []
    for index, keyframe in enumerate(keyframes):
        keyframe = keyframe.convert("RGB")
        if keyframe.size == (width, height):
            prepared.append(keyframe)
        elif index == 0:
            prepared.append(keyframe.resize((width, height), Image.Resampling.LANCZOS))
        else:
            scale = max(width / keyframe.size[0], height / keyframe.size[1])
            resized_size = (max(width, round(keyframe.size[0] * scale)),
                            max(height, round(keyframe.size[1] * scale)))
            left = max(0, (resized_size[0] - width) // 2)
            top = max(0, (resized_size[1] - height) // 2)
            resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
            prepared.append(resized.crop((left, top, left + width, top + height)))
    return prepared, height, width


def build_presentation(processor, prompt, keyframes):
    """MiniMaxH3FL2VATextEncoderStep's tokenisation: label + vision block per keyframe,
    then the prompt verbatim; returns token ids, per-row tags, vision inputs."""
    tokenizer = processor.tokenizer
    vision = processor.image_processor(images=keyframes, return_tensors="pt")
    image_grid_thw = vision["image_grid_thw"]
    vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": image_grid_thw}
    token_ids, token_tags = [], []
    merge_size = processor.image_processor.merge_size ** 2
    for index in range(len(keyframes)):
        num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
        label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        vision_ids = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                      + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                      + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        token_ids += label_ids + vision_ids
        token_tags += [TEXT_TAG] * len(label_ids) + [VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags, vision_inputs


@torch.no_grad()
def qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, device):
    """diffusers' get_qwen3vl_prompt_embeds, inlined: hidden state after decoder layer 50
    of `text_encoder.model` with the vision tensors and Qwen-internal mm_token_type_ids."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]),
                                     dtype=torch.long, device=device)
    vision_kwargs = {name: (value.to(device, text_encoder.dtype) if name.startswith("pixel_")
                            else value.to(device)) for name, value in vision_inputs.items()}
    outputs = text_encoder.model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids, use_cache=False, output_hidden_states=True,
        **vision_kwargs)
    return outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()


def check_instruction(prompt, anchors):
    """The mode these keyframes select. Warns on stderr when the prompt does not open
    with that mode's instruction line -- never refuses: the check is a substring, and a
    prompt is free text."""
    mode, marker, template = KEYFRAME_MODES[tuple(anchors)]
    opening = next((line for line in prompt.splitlines() if line.strip()), "")
    if marker.lower() not in opening.lower():
        print(f"warning: {mode} keyframes, but the prompt does not open with {mode}'s "
              f"instruction line. MiniMax-H3 expects it as the first line:\n"
              f"    {template}\nEncoding the prompt as given.", file=sys.stderr, flush=True)
    return mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--first", type=str, help="keyframe the video starts from (image path)")
    p.add_argument("--last", type=str, help="keyframe the video ends on (image path)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--model_root", type=str, default=None,
                   help="a MiniMax-H3 snapshot with processor/ and text_encoder/; "
                        "default: downloaded from the Hub on first use")
    p.add_argument("--vae_root", type=str, default=H3_BASE, help="root holding vae/")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    if not (args.first or args.last):
        p.error("pass --first and/or --last (a request without keyframes is t2va: "
                "use encode_prompt.py)")
    if args.model_root is None:
        args.model_root = upstream_snapshot("processor", "text_encoder")

    pairs = [(anchor, path) for anchor, path in (("first", args.first), ("last", args.last)) if path]
    anchors = [anchor for anchor, _ in pairs]
    mode = check_instruction(args.prompt, anchors)
    keyframes, height, width = put_on_canvas([Image.open(path) for _, path in pairs])
    print(f"{mode}: canvas {width}x{height}; keyframes {anchors}", flush=True)
    device = args.device

    processor = Qwen3VLProcessor.from_pretrained(args.model_root, subfolder="processor")
    token_ids, token_tags, vision_inputs = build_presentation(processor, args.prompt, keyframes)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_root, subfolder="text_encoder", dtype=torch.bfloat16).to(device)
    text_encoder.eval().requires_grad_(False)
    prompt_embeds = qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, device)
    del text_encoder
    torch.cuda.empty_cache()

    vae = AutoencoderKLMiniMaxH3.from_pretrained(resolve_weights(args.vae_root), subfolder="vae").to(device)
    vae.eval()
    with torch.no_grad():
        condition_latents = [
            encode_vae_condition(vae, torch.from_numpy(np.array(k)).to(device).permute(2, 0, 1)[None, :, None],
                                 PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED)
            for k in keyframes]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "prompt": args.prompt,
        "prompt_embeds": prompt_embeds,                                   # (L, 5120) bf16
        "text_token_tags": torch.tensor(token_tags, dtype=torch.long),   # vision rows tagged VIDEO
        "keyframe_anchors": anchors,
        "keyframe_files": [path for _, path in pairs],
        "condition_latents": condition_latents,
        "height": height, "width": width,
    }, args.out)
    print(f"wrote {args.out}: {len(token_ids)} tokens ({token_tags.count(VIDEO_TAG)} vision rows), "
          f"{len(keyframes)} keyframes {[tuple(c.shape) for c in condition_latents]}", flush=True)


if __name__ == "__main__":
    main()

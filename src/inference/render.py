"""Render glue shared by every inference entrypoint: load_models, load_text,
generate_latents (the denoising loop + packed layout), and decode_and_save with its
atomic .partial.mp4 write.
"""
import os
import time

import torch

from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
                       MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, patchify_video_latents)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames, audio_latent_num_frames, video_latent_num_frames)
from diffusers.utils.export_utils import encode_video

from src.models.sequence_layout import layout_from_indices
from src.paths import H3_BASE, resolve_weights
from src.models.hybrid_transform import iter_hybrids, set_layout

# The release copy under the repository (transformer/ vae/ audio_vae/); the decoders and
# the dense `checkpoint=null` render load from here unless the config names another root
# (`vae_source` / `base_source`). Relative paths resolve against the repo root, not cwd.
DEFAULT_MODEL_ROOT = H3_BASE
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LATENT_H, LATENT_W = 48, 84
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def load_models(model_root: str, device: str, vae_source: str = None,
                load_decoders: bool = True):
    """`model_root` holds the transformer; both decoders come from `vae_source`,
    default the release copy. Ranks that never decode pass `load_decoders=False` and
    get (transformer, None, None)."""
    vae_source = resolve_weights(vae_source or DEFAULT_MODEL_ROOT)
    model_root = resolve_weights(model_root)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        model_root, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval().requires_grad_(False)

    if not load_decoders:
        return transformer, None, None

    vae = AutoencoderKLMiniMaxH3.from_pretrained(vae_source, subfolder="vae").to(device)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(vae_source, subfolder="audio_vae").to(device)
    vae.eval()
    audio_vae.eval()
    return transformer, vae, audio_vae


def load_decoders(vae_source: str, device: str):
    """Load only the final decoders, optionally onto CPU for a persistent worker."""
    vae_source = resolve_weights(vae_source or DEFAULT_MODEL_ROOT)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(vae_source, subfolder="vae").to(device)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        vae_source, subfolder="audio_vae").to(device)
    vae.eval().requires_grad_(False)
    audio_vae.eval().requires_grad_(False)
    return vae, audio_vae



def load_text(prompt_file: str, device: str):
    """A cached prompt from encode_prompt.py (prompts/*.pt)."""
    text = torch.load(prompt_file, map_location="cpu", weights_only=True)
    return text["prompt_embeds"].to(device, torch.bfloat16), text["text_token_tags"]


@torch.no_grad()
def generate_latents(transformer, prompt_embeds, text_token_tags, num_frames, num_steps, seed, device,
                     video_shift=12.0, audio_shift=3.0, runtime=None, step_seconds=None):
    """The sampler. `runtime` (Ulysses) adds a barrier before and after the loop so every
    rank enters and leaves together; `step_seconds`, if a list, receives the
    device-synchronised wall time of every NFE (for the timing log and the record)."""
    num_frames = align_num_frames(num_frames, 17, 5)
    num_latent_frames = video_latent_num_frames(num_frames, 17, 5)
    num_audio_latents = audio_latent_num_frames(num_frames)

    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags, num_latent_frames, LATENT_H, LATENT_W, num_audio_latents,
            PATCH_SIZE, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, keyframe_anchors=(),
        )
    )
    position_ids, token_tags = position_ids.to(device), token_tags.to(device)
    video_indices, audio_indices, text_indices = (
        video_indices.to(device), audio_indices.to(device), text_indices.to(device),
    )

    # A hybrid-converted transformer needs the packed layout before every forward; the
    # layout is the same at every denoising step, so set it once per generation.
    if next(iter_hybrids(transformer), None) is not None:
        # frame_size and text_indices unconditionally: carrying them is free, and only
        # their consumers are gated (short_conv / text_state).
        set_layout(transformer, layout_from_indices(
            video_indices, num_latent_frames,
            (LATENT_H // PATCH_SIZE[1]) * (LATENT_W // PATCH_SIZE[2]),
            seq_len=position_ids.shape[0],
            frame_size=(LATENT_H // PATCH_SIZE[1], LATENT_W // PATCH_SIZE[2]),
            text_indices=text_indices,
        ))

    scheduler = MiniMaxH3Scheduler(shift=video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=audio_shift)
    scheduler.set_timesteps(num_steps, device=device)
    audio_scheduler.set_timesteps(num_steps, device=device)

    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, 24, num_latent_frames, LATENT_H, LATENT_W),
                          generator=generator, device=device, dtype=torch.float32)
    video_rows = patchify_video_latents(latents, PATCH_SIZE)
    audio_rows = torch.randn((num_audio_latents * AUDIO_CHANNELS, 32),
                             generator=generator, device=device, dtype=torch.float32)

    if runtime is not None:
        runtime.barrier()
        torch.cuda.synchronize(device)

    seq_len = position_ids.shape[0]
    for t, audio_t in zip(scheduler.timesteps, audio_scheduler.timesteps):
        step_started = time.perf_counter()
        row_timesteps = torch.full((seq_len,), float(t), dtype=torch.float32, device=device)
        row_timesteps[audio_indices] = float(audio_t)
        timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        noise_pred, audio_noise_pred = transformer(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds[None],
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=False,
        )
        video_rows = scheduler.step(noise_pred[0].float(), t, video_rows, return_dict=False)[0]
        audio_rows = audio_scheduler.step(audio_noise_pred[0].float(), audio_t, audio_rows, return_dict=False)[0]

        if step_seconds is not None:
            torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - step_started)

    if runtime is not None:
        torch.cuda.synchronize(device)
        runtime.barrier()

    # Unpatchify (the AfterDenoise step's reshape) and unpack the channel-major audio rows.
    rows = video_rows.reshape(-1, num_latent_frames, LATENT_H // 2, LATENT_W // 2, 24, 1, 2, 2)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = rows.reshape(-1, 24, num_latent_frames, LATENT_H, LATENT_W).contiguous()
    audio_latents = audio_rows.reshape(AUDIO_CHANNELS, num_audio_latents, 32).permute(0, 2, 1).contiguous()
    return latents, audio_latents


@torch.no_grad()
def decode_and_save(latents, audio_latents, vae, audio_vae, out_path: str, device: str):
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        video = vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio = audio_vae.decode(audio_latents * audio_latents_std + audio_latents_mean, return_dict=False)[0]
    audio = audio.float().permute(1, 0, 2)[0]  # (2, num_samples)

    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()  # (F, H, W, 3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Encode to a sibling temp name and rename into place, so a render that is killed
    # mid-encode never leaves a truncated clip under the final name (anything that
    # treats "the mp4 exists" as "done" would otherwise pick it up). os.replace is
    # atomic within a directory, so the final name only ever appears complete.
    # The suffix goes BEFORE the extension: PyAV infers the container format from the
    # output filename, and "clip.mp4.partial" makes av.open raise "Could not determine
    # output format". Same directory either way, which is what os.replace needs.
    stem, ext = os.path.splitext(out_path)
    partial_path = f"{stem}.partial{ext}"
    encode_video(frames, fps=24, output_path=partial_path,
                 audio=audio.cpu(), audio_sample_rate=audio_vae.config.sampling_rate)
    os.replace(partial_path, out_path)
    print(f"wrote {out_path}", flush=True)

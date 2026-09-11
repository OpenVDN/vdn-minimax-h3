"""Shared packing for the hybrid trainers: latent sample -> noisy packed model inputs.

One implementation so every stage consumes byte-identical batches: same
[text | target audio | target video] layout via the pipeline's own
`build_packed_sequence`, same inference-locked (12u, 3u) sigma pairing, same data-ward
velocity target. Additionally derives the SequenceLayout the windowed attention needs.
"""
import torch

from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    patchify_video_latents,
)

from src.models.sequence_layout import SequenceLayout, layout_from_indices

# Layout constants of the released checkpoint (modular_pipeline.py properties).
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0


def sample_sigmas(generator: torch.Generator) -> tuple[float, float]:
    """One u per step, pushed through both exponential shifts: sigma' = s*u / (1 + (s-1)*u).

    This is the density `MiniMaxH3Scheduler.set_timesteps` walks at inference (uniform in
    u), and pairing both modalities on the same u reproduces the (12u, 3u) sigma pairs
    every inference step actually visits.
    """
    u = torch.rand((), generator=generator).clamp(1e-4, 1.0)
    sigma_v = VIDEO_SHIFT * u / (1 + (VIDEO_SHIFT - 1) * u)
    sigma_a = AUDIO_SHIFT * u / (1 + (AUDIO_SHIFT - 1) * u)
    return float(sigma_v), float(sigma_a)


def few_step_timesteps(num_steps: int, video_shift: float,
                       audio_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact paired model-evaluation times used by MiniMaxH3Scheduler.

    ``num_steps`` is the model-evaluation count (NFE). The scheduler counts sigma grid
    points instead, the terminal clean one for the final Euler update included, so an
    8-step schedule is nine grid points and eight paired forward times. Keeping the
    formula here beside ``sample_sigmas`` makes Stage-DMD's training support a discrete
    subset of the same shifted flow path Stage-B samples continuously.
    """
    if num_steps < 1:
        raise ValueError(f"few-step schedule needs num_steps >= 1, got {num_steps}")
    if video_shift <= 0 or audio_shift <= 0:
        raise ValueError("few-step scheduler shifts must be positive")

    base_sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)[:-1]

    def shifted_time(shift):
        sigma = shift * base_sigma / (1 + (shift - 1) * base_sigma)
        return 1.0 - sigma

    return shifted_time(video_shift), shifted_time(audio_shift)


def sample_few_step_timesteps(generator: torch.Generator, num_steps: int,
                              video_shift: float,
                              audio_shift: float) -> tuple[float, float, int]:
    """Uniformly select one paired forward time from a discrete inference grid."""
    video, audio = few_step_timesteps(num_steps, video_shift, audio_shift)
    index = int(torch.randint(len(video), (), generator=generator))
    return float(video[index]), float(audio[index]), index


def build_layout(text_token_tags: torch.Tensor, video_shape: tuple,
                 num_audio_latents: int, device):
    """The t2va packed layout `[text | target audio | target video]`, no conditioning rows."""
    _, num_latent_frames, latent_height, latent_width = video_shape
    position_ids, token_tags, video_indices, audio_indices, text_indices, ncv, nca = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=PATCH_SIZE,
            audio_channels=AUDIO_CHANNELS,
            audio_tag=AUDIO_TAG,
            video_tag=VIDEO_TAG,
            keyframe_anchors=(),
        )
    )
    assert ncv == 0 and nca == 0
    return (position_ids.to(device), token_tags.to(device), video_indices.to(device),
            audio_indices.to(device), text_indices.to(device))


def pack_noisy_batch(sample, device, noise_generator, sigma_generator,
                     fixed_timesteps: tuple[float, float] | None = None):
    """One dataset sample -> (model kwargs, velocity targets, sigma info, SequenceLayout).

    Returns a dict with:
      inputs   kwargs for MiniMaxH3Transformer3DModel.forward (batch 1)
      target_video_rows / target_audio_rows   fp32 data-ward velocity targets
      t_v      the video timestep (for logging)
      layout   SequenceLayout of this packed sequence (for set_layout)
    """
    video = sample["video_latents"].to(device, torch.float32)[None]         # (1, 24, T, h, w)
    audio = sample["audio_latents"].to(device, torch.float32)               # (2, 32, A)
    prompt_embeds = sample["prompt_embeds"].to(device, torch.bfloat16)[None]
    text_token_tags = sample["text_token_tags"]

    num_audio_latents = audio.shape[-1]
    audio_rows = audio.permute(0, 2, 1).reshape(-1, audio.shape[1])          # (2A, 32) channel-major

    position_ids, token_tags, video_indices, audio_indices, text_indices = build_layout(
        text_token_tags, tuple(video.shape[1:]), num_audio_latents, device
    )

    if fixed_timesteps is None:
        sigma_v, sigma_a = sample_sigmas(sigma_generator)
        t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
    else:
        t_v, t_a = map(float, fixed_timesteps)
        if not (0.0 <= t_v < 1.0 and 0.0 <= t_a < 1.0):
            raise ValueError(f"model-evaluation timesteps must be in [0, 1), got "
                             f"video={t_v}, audio={t_a}")

    noise_v = torch.randn(video.shape, generator=noise_generator, device=device, dtype=torch.float32)
    noise_a = torch.randn(audio_rows.shape, generator=noise_generator, device=device, dtype=torch.float32)
    noisy_video_rows = patchify_video_latents(t_v * video + (1.0 - t_v) * noise_v, PATCH_SIZE)
    noisy_audio_rows = t_a * audio_rows + (1.0 - t_a) * noise_a
    target_video_rows = patchify_video_latents(video - noise_v, PATCH_SIZE)  # data-ward velocity
    target_audio_rows = audio_rows - noise_a

    # Distinct timesteps + per-row index, the transformer's (timestep, timestep_indices)
    # contract. Text rows never reach an output head and inherit the video timestep.
    row_timesteps = torch.full((position_ids.shape[0],), t_v, dtype=torch.float32, device=device)
    row_timesteps[audio_indices] = t_a
    timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)

    _, num_latent_frames, latent_height, latent_width = tuple(video.shape[1:])
    tokens_per_frame = (latent_height // PATCH_SIZE[1]) * (latent_width // PATCH_SIZE[2])

    # frame_size and text_indices unconditionally: carrying them is free, and only
    # their consumers are gated (ShortConv on short_conv, the text seed on text_state).
    layout = layout_from_indices(video_indices, num_latent_frames, tokens_per_frame,
                                 seq_len=position_ids.shape[0],
                                 frame_size=(latent_height // PATCH_SIZE[1],
                                             latent_width // PATCH_SIZE[2]),
                                 text_indices=text_indices)

    inputs = dict(
        hidden_states=noisy_video_rows[None],
        audio_hidden_states=noisy_audio_rows[None],
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )
    return dict(inputs=inputs, target_video_rows=target_video_rows,
                target_audio_rows=target_audio_rows, t_v=t_v, t_a=t_a,
                layout=layout)


# ------------------------------------------------------------- Stage-DMD (sampler side)
# Stage-DMD's generator produces its own latents, so it needs the packing above for an
# ARBITRARY state x_t (the sampler's), not for data + noise. Everything that does not
# depend on the state or the timestep -- the packed layout of one prompt -- is built once
# per prompt (PackedPrompt) and reused across every forward of that iteration: the
# rollout, the generator step, the real/fake score forwards and the fake updates.

def few_step_sigmas(num_steps: int, video_shift: float,
                    audio_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The full (num_steps + 1)-point sigma grids, terminal 0 included: exactly
    `MiniMaxH3Scheduler.sigmas`, of which `few_step_timesteps` is `1 - sigmas[:-1]`."""
    if num_steps < 1:
        raise ValueError(f"few-step schedule needs num_steps >= 1, got {num_steps}")
    if video_shift <= 0 or audio_shift <= 0:
        raise ValueError("few-step scheduler shifts must be positive")
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    return (video_shift * base / (1 + (video_shift - 1) * base),
            audio_shift * base / (1 + (audio_shift - 1) * base))


def x0_from_velocity(sample: torch.Tensor, velocity: torch.Tensor,
                     timestep: torch.Tensor) -> torch.Tensor:
    """The scheduler's `denoised`: x0 = x_t + (1 - t) * v, sigma recovered from the
    TIMESTEP the transformer was conditioned on (fp32 arithmetic on fp32 rows)."""
    timestep = torch.as_tensor(timestep, dtype=sample.dtype, device=sample.device)
    return sample + (1 - timestep) * velocity


def euler_blend(sample: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor,
                sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    """`MiniMaxH3Scheduler.step` for fp32 rows, arithmetic verbatim: the Euler (eta=0)
    update as the blend `r*x_t + (1-r)*x0` with `r = sigma_next / sigma` taken from the
    sigma GRID (the scheduler keeps the two sigma sources apart on purpose)."""
    denoised = x0_from_velocity(sample, velocity, timestep)
    sigma = torch.as_tensor(sigma, dtype=sample.dtype, device=sample.device)
    sigma_next = torch.as_tensor(sigma_next, dtype=sample.dtype, device=sample.device)
    ratio = sigma_next / sigma
    return ratio * sample + (1.0 - ratio) * denoised


def sample_paired_timesteps(generator: torch.Generator, u_min: float, u_max: float,
                            video_shift: float,
                            audio_shift: float) -> tuple[float, float, float]:
    """One u ~ U[u_min, u_max] pushed through both shifts: the (12u, 3u) pairing the
    model is trained and sampled under, restricted to DMD's interior range. Returns
    (t_video, t_audio, u) with t = 1 - sigma."""
    if not (0.0 <= u_min < u_max <= 1.0):
        raise ValueError(f"need 0 <= u_min < u_max <= 1, got [{u_min}, {u_max}]")
    u = u_min + (u_max - u_min) * float(torch.rand((), generator=generator))
    sigma_v = video_shift * u / (1 + (video_shift - 1) * u)
    sigma_a = audio_shift * u / (1 + (audio_shift - 1) * u)
    return 1.0 - sigma_v, 1.0 - sigma_a, u


class PackedPrompt:
    """One prompt's packed [text | audio | video] geometry, built once and reused.

    `inputs(video_rows, audio_rows, t_v, t_a)` returns the transformer kwargs for an
    arbitrary state, byte-identical to what `pack_noisy_batch` builds for the same rows
    and timesteps; `layout` is what `set_layout` wants."""

    def __init__(self, prompt_embeds: torch.Tensor, text_token_tags: torch.Tensor,
                 video_shape: tuple, num_audio_latents: int, device):
        self.prompt_embeds = prompt_embeds.to(device, torch.bfloat16)[None]
        self.video_shape = tuple(video_shape)
        self.num_audio_latents = int(num_audio_latents)
        self.device = device
        (self.position_ids, self.token_tags, self.video_indices, self.audio_indices,
         self.text_indices) = build_layout(text_token_tags, self.video_shape,
                                           self.num_audio_latents, device)
        _, num_latent_frames, latent_height, latent_width = self.video_shape
        tokens_per_frame = (latent_height // PATCH_SIZE[1]) * (latent_width // PATCH_SIZE[2])
        self.seq_len = int(self.position_ids.shape[0])
        self.layout = layout_from_indices(
            self.video_indices, num_latent_frames, tokens_per_frame, seq_len=self.seq_len,
            frame_size=(latent_height // PATCH_SIZE[1], latent_width // PATCH_SIZE[2]),
            text_indices=self.text_indices)
        self.num_video_rows = int(self.video_indices.numel())

    def inputs(self, video_rows: torch.Tensor, audio_rows: torch.Tensor,
               t_v: float, t_a: float) -> dict:
        t_v, t_a = float(t_v), float(t_a)
        if not (0.0 <= t_v < 1.0 and 0.0 <= t_a < 1.0):
            raise ValueError(f"model-evaluation timesteps must be in [0, 1), got "
                             f"video={t_v}, audio={t_a}")
        if video_rows.shape[0] != self.num_video_rows:
            raise ValueError(f"video rows {tuple(video_rows.shape)} do not match the "
                             f"packed geometry ({self.num_video_rows} rows)")
        if audio_rows.shape[0] != self.num_audio_latents * AUDIO_CHANNELS:
            raise ValueError(f"audio rows {tuple(audio_rows.shape)} do not match "
                             f"{self.num_audio_latents} latents x {AUDIO_CHANNELS} channels")
        row_timesteps = torch.full((self.seq_len,), t_v, dtype=torch.float32,
                                   device=self.device)
        row_timesteps[self.audio_indices] = t_a
        timestep, timestep_indices = torch.unique(row_timesteps, sorted=True,
                                                  return_inverse=True)
        return dict(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=self.prompt_embeds,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=self.token_tags,
            position_ids=self.position_ids,
            video_indices=self.video_indices,
            audio_indices=self.audio_indices,
            text_indices=self.text_indices,
            return_dict=False,
        )


def initial_noise(video_shape: tuple, num_audio_latents: int, generator: torch.Generator,
                  device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-noise packed rows, drawn exactly as `render.generate_latents` draws them:
    the video in latent space (then patchified), the audio directly as channel-major rows."""
    video = torch.randn((1, *video_shape), generator=generator, device=device,
                        dtype=torch.float32)
    video_rows = patchify_video_latents(video, PATCH_SIZE)
    audio_rows = torch.randn((num_audio_latents * AUDIO_CHANNELS, 32), generator=generator,
                             device=device, dtype=torch.float32)
    return video_rows, audio_rows

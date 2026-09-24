# Inference with this repository

How the repository's own stack renders: the entry points, the configs, the kernels behind them, fp8, the eight-GPU layout, memory, and what a prompt cache carries. The commands to run are in the [README](../README.md#inference-with-our-repository); the portable diffusers path is in [diffusers.md](diffusers.md).

## Entry points

| Script | What it does |
|---|---|
| [src/inference/infer.py](../src/inference/infer.py) | one GPU |
| [src/inference/infer_ulysses.py](../src/inference/infer_ulysses.py) | eight GPUs under `torchrun`, sequence parallel |
| [src/inference/encode_prompt.py](../src/inference/encode_prompt.py) | a text prompt to a prompt cache (`.pt`) |
| [src/inference/encode_keyframes.py](../src/inference/encode_keyframes.py) | a prompt plus keyframes (`--first`, `--last`) or reference images (`--refs`) to a prompt cache |
| [src/inference/infer_diffusers.py](../src/inference/infer_diffusers.py) | the published diffusers component; shares no code with the two above |

Encoding is separate from rendering because the Qwen3-VL text encoder is 63 GB: a prompt is encoded once, and every render reuses the cache without loading the encoder.

## Configs

A config file says how to compute, and the command line says what: the checkpoint, the prompt cache, the output. Every knob is a field of [src/config/inference.py](../src/config/inference.py), set in the YAML or as a `key=value` override. Nothing on the inference path reads an environment variable.

The files in [configs/inference/](../configs/inference/) form a ladder, once for 8 and once for 50 model evaluations:

| File | Adds |
|---|---|
| `8nfe.yaml` | the released arithmetic in bf16; with `checkpoint=null`, the dense MiniMax-H3 |
| `8nfe_tuned.yaml` | the inference kernels; the same sample up to kernel rounding |
| `8nfe_tuned_fp8.yaml` | fp8 on every wide Linear, 2 warm-up evaluations; the top single-GPU tier |
| `8nfe_tuned_fp8_ulysses_h200.yaml`, `_b200.yaml` | the eight-GPU layout tuned for that card |

The fields:

| Field | Default | Meaning |
|---|---|---|
| `checkpoint` | `null` | a checkpoint directory; `null` renders the dense base model |
| `base_source`, `vae_source` | the release copy | where the base transformer and the two decoders load from |
| `external_loras` | `[]` | community adapters, see [Weights](#weights) |
| `kernels.inference_kernels` | `true` | the forward-only kernel set |
| `kernels.softmax_backend` | `auto` | `auto`, `flex`, `decomposed` or `ref` |
| `precision.fp8.enabled` | `false` | fp8 on the wide Linears |
| `precision.fp8.skip_end_blocks` | `4` | first and last N blocks left in bf16; the shipped fp8 configs use 0 |
| `parallel.softmax_ranks` | `6` | `infer_ulysses.py` only: ranks given to the softmax heads; 0 is standard Ulysses |
| `render.prompt_file`, `render.out` | required | the prompt cache and the mp4 |
| `render.num_frames` | `345` | aligned to 17n+5; 345 frames are 14.4 seconds at 24 fps |
| `render.num_steps` | `50` | model evaluations |
| `render.warmup_steps` | `0` | evaluations run and discarded first |
| `render.seed` | `42` | |
| `render.video_shift`, `render.audio_shift` | `12.0`, `3.0` | scheduler shifts; a distilled adapter is valid only at the shifts it was trained at |
| `render.record` | `false` | also write `<out>.inference.json` |
| `render.save_latents` | `false` | also save the video and audio latents under `render.latents_root` |

`render.num_steps` is the number of model evaluations. The scheduler counts sigma grid points, one more than that, and the render passes the extra one itself.

## Assembly

Both entry points build the model through [src/inference/utils/assemble.py](../src/inference/utils/assemble.py), in this order: read the checkpoint's spec, load the base, apply the hybrid transform, load the linear branch, merge the checkpoint's LoRAs, merge the external LoRAs, select the inference kernels and the window-softmax backend, then convert to fp8. The record written by `render.record=true` states what actually ran: the checkpoint's identity, the resolved config, the resolved backend, the number of fp8 Linears, and the time of every evaluation.

## Inference kernels

`kernels.inference_kernels: true` swaps in forward-only kernels with the same arithmetic:

- **Attention:** QK-norm and RoPE fused in one pass over views of the packed q/k/v buffer, and the softmax gate fused with the output repack.
- **Linear branch:** one kernel for the frame statistics, a Triton 5-tap temporal convolution, a compiled RMSNorm and output gate, one triangular solve for the inverse, and `baddbmm` scan updates.
- **Block:** RMSNorm with the AdaLN affine, gate and residual as two compiled kernels, and SwiGLU as one.

The first run compiles them, which takes several minutes; later runs reuse the cache.

## Window softmax

Three implementations of the same windowed attention:

- `flex`: FlexAttention with a block mask. Training uses it.
- `decomposed`: the mask as a union of dense attentions, with no mask machinery. Global rows attend to everything in one dense call, and the frames that share a window go through one variable-length call.
- `ref`: the eager reference, for parity checks.

`auto` is `decomposed` on every CUDA device. It is the faster kernel on B200 and matches `flex` on H200. The two differ by bf16 reduction order, and by their transient memory, which [Memory](#memory) measures. Each picks its kernel by the card:

| Card | `flex` | `decomposed` |
|---|---|---|
| Hopper, data-center Blackwell (sm90, sm100, sm110) | FlexAttention's Flash backend (FA4) | FA4's varlen kernel, cuDNN SDPA |
| Ampere, Ada, consumer Blackwell (sm8x, sm120) | FlexAttention's Triton kernel | PyTorch's `varlen_attn`, SDPA's own kernels |

Nothing degrades to dense attention. A missing package raises its own import error: `flash-attn-4` on the first row, PyTorch 2.13 on the second.

## fp8

`precision.fp8.enabled: true` replaces every Linear at least 4096 wide on both sides with [Fp8Linear](../src/models/ops/fp8_linear.py): the qkv and output projections, the feed-forward pair and the linear branch's readout, 363 Linears with `skip_end_blocks: 0`. Narrow Linears stay in bf16.

- The quantizer is a Triton kernel, one program per row. The attention quantizes its input once for the three projections, and the feed-forward fuses SwiGLU with the quantization of its output.
- Scales are per row on sm90 and per tensor on sm100 and above, which is where each card's fast GEMM is. The granularity follows the card and is not a knob.
- The swap is one way: the bf16 weight is released as each Linear is replaced, so the model is roughly half its weight afterwards.
- fp8 needs compute capability 9.0 or above.

fp8 changes the sample: one evaluation stays close to bf16 (cosine 0.998), and the trajectory settles into a different video of the same quality. A seed reproduces a render only within one precision.

## Eight GPUs

`infer_ulysses.py` shards the packed sequence by row across the ranks. Inside attention one all-to-all turns sequence-sharded, all-heads q/k/v into all-sequence, head-sharded q/k/v, and a second one restores the layout, so the window softmax and the linear branch both see the whole sequence. There is no KV cache, token dropping or approximate communication.

The layout is branch-parallel: `parallel.softmax_ranks` ranks run the softmax heads and the rest run the linear branch. The shipped configs state the split and the kernel each card was tuned with:

| Config | Split | Window kernel |
|---|---|---|
| `_ulysses_h200.yaml` | 6 + 2 | `flex` |
| `_ulysses_b200.yaml` | 5 + 3 | `decomposed` |

Only rank 0 loads the decoders and writes the mp4. T2VA, keyframe and reference caches all run here: conditioning rows sit outside the video span, so they shard and attend as ordinary global rows.

Setup is dominated by eight ranks reading 66 GB each, so keep `ckpts/` on a local disk. The last log line splits the time into setup, denoise, and decode plus encode. Eight-GPU renders are not bit-reproducible from run to run; one-GPU renders are.

## Memory

The fp8 configs fit an 80 GB card, on one GPU and on eight. Peak GPU memory in GiB at 345 frames and 8 evaluations, on the T2VA example, measured on an H200 with the CUDA allocator capped at 78 GiB:

| Phase | One GPU | Eight GPUs, per rank |
|---|---:|---:|
| Load: the bf16 transformer | 61.7 | 61.7 |
| Assembly, through the fp8 conversion | 65.9 | 65.9 |
| Denoise | 71.4 | 55.9 to 60.4 |
| Decode | 67.8 | 68.5, rank 0 only |

The decoders wait on the CPU and move to the GPU when denoising is over, and the fp8 conversion releases each bf16 weight as it goes, so the model never sits on the card twice. Assembly then returns the freed memory to the driver, where NCCL allocates its buffers at the first collective. On one GPU the bf16 configs need more than 80 GB: 66 GiB of weights plus 26 GiB of transients.

The two encoders fit the same card: `encode_prompt.py` peaks at 63.2 GiB, and `encode_keyframes.py` with six references at 69.2.

Conditioning rows raise the denoise peak, and the window kernel decides by how much. `decomposed` gathers the global rows once per window group, and `flex` has no gather. One GPU, the same cap:

| Request | Window kernel | Denoise peak | Seconds per evaluation |
|---|---|---:|---:|
| T2VA | `decomposed` | 71.4 | 11.8 |
| T2VA | `flex` | 62.7 | 11.8 |
| FL2VA example | `decomposed` | 74.4 | 14.2 |
| Ref2VA-like example | `decomposed` | out of memory | 19.1 without the cap |
| Ref2VA-like example | `flex` | 63.8 | 17.8 |

`auto` stays `decomposed`. On an 80 GB card, render a reference cache with `kernels.softmax_backend=flex`:

```bash
python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 kernels.softmax_backend=flex \
  render.prompt_file=prompts/reference/example_ref2va.pt render.out=results/example_ref2va.mp4
```

## Prompt caches and conditioning

A prompt cache is one `.pt` holding the prompt text, the Qwen3-VL hidden state after layer 50 as `(L, 5120)` bf16, and a tag per row. A keyframe or reference cache adds the anchors and one VAE latent per image. Each token costs 10 KB, and an image costs about one token per 32×32 pixels, so a cache with images runs to tens of megabytes.

| Request | Encoder flags | Packed sequence |
|---|---|---|
| T2VA | `encode_prompt.py --prompt` | `[text, audio, video]` |
| I2VA, L2VA, FL2VA | `encode_keyframes.py --first` and/or `--last` | `[text, keyframes, audio, video]` |
| Ref2VA-like | `encode_keyframes.py --refs a.png b.png ...` | `[text, one block per reference, audio, video]` |

Keyframes go onto the video canvas: the first is stretched to it and the last is cover-cropped. References keep their own aspect ratio on a 768-pixel short edge (`--ref_size`), and each takes its own rotary slot ahead of the video. Every image appears twice: as a `<Picture i>` vision block in the text, and as conditioning rows that are noised to t = 0.999 and held there while only the generated rows are stepped. On the hybrid model the conditioning rows are global tokens, attended densely in both directions by the window softmax and absent from the linear scan.

Conditioning rows lengthen the sequence. A T2VA render is about 104k audio and video rows plus the text. The Samoyed reference example adds 6,948 text and 5,664 reference rows, and runs at 19.1 seconds per evaluation on one H200 in fp8 and 3.4 on eight.

## Frames, canvas and seeds

The canvas is 1344×768. `render.num_frames` is aligned to 17n+5, and 345 frames become 102 latent frames.

One-GPU renders are bit-reproducible: the same config, seed and cache give the same mp4 bytes on the same GPU model. The initial noise differs between GPU models.

## Output

The mp4 is 24 fps with the audio muxed in. It is encoded to a sibling `.partial.mp4` and renamed into place, so a file under the final name is always complete.

## Weights

A path written as `ckpts/<name>` that does not exist locally is downloaded from the Hub into the Hugging Face cache and used from there, so every command in the README also runs without a local `ckpts/`.

`external_loras` merges community adapters for the dense MiniMax-H3, safetensors with peft names, after the checkpoint's own LoRA:

```bash
python src/inference/infer.py --config configs/inference/8nfe.yaml \
  checkpoint=null "external_loras=[{path: turbo.safetensors}]" \
  render.prompt_file=prompts/example_0.pt render.out=results/turbo.mp4
```

`alpha` defaults to the value in the file's metadata, and to the rank when the file has none.

## Timing a render

Set `render.warmup_steps: 2`, as the fp8 configs do. The first evaluation compiles, and the second re-specializes for the next timestep. The timing line reports every evaluation after that, and `render.record=true` keeps them in the JSON. The numbers in the README's [Results](../README.md#results) exclude model loading, warm-up, VAE decoding and mp4 encoding.

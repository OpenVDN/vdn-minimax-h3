# Video DeltaNet: Hybrid Attention to Speed Up Video Models with Near-Lossless Quality

[[`Blog`](https://openvdn.github.io/)] [[`🤗 HuggingFace`](https://huggingface.co/OpenVDN/vdn-minimax-h3)] [[`License`](#license)]

We release **VDN-Minimax-H3** (**VDN-H3**), a hybrid-attention model that
generates video faster than it plays, powered by
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). It offers these key
features:

- **Fast inference:** On 8 B200 GPUs, VDN-H3 generates a 14.4-second clip in
  **11.23 seconds** using 8 denoising steps.
- **Hybrid Architecture:** We propose a hybrid-attention architecture: one frame-wise
  linear attention branch that is highly efficient, and a softmax branch that maintains
  the backbone's visual quality and consistency.
- **Plug-and-Play:** The checkpoint adds a separate linear attention branch and two
  small LoRA adapters that can be merged into the backbone during inference without
  touching the backbone weights.
- **Fully open-source:** We don't just open-source the weights. The optimized inference
  stack and its corresponding training code are released together.

We present some samples of generated videos here:

<table>
<tr>
<td width="33%"><video src="https://github.com/user-attachments/assets/04aa614a-3ff3-43ab-a3fa-ed3f49e039ec" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/7ac8d7dc-f635-4e3f-85da-89d681efdd52" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/e3e9f30e-434e-4a0f-bb87-0ba5749abea4" controls muted></video></td>
</tr>
<tr>
<td width="33%"><video src="https://github.com/user-attachments/assets/811bb4d3-f036-45a7-8098-2752fdc5619d" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/5f5ccfa6-5d88-4ac4-842a-774358432e28" controls muted></video></td>
<td width="33%"><video src="https://github.com/user-attachments/assets/06dd0699-7047-4b25-bcb7-8b541a4e2718" controls muted></video></td>
</tr>
</table>

## Set up environment

**Before you begin**, please read the [license](#license) before downloading or
running VDN-H3.

1. Clone the VDN-H3 repository from GitHub.

```bash
git clone https://github.com/OpenVDN/vdn-minimax-h3.git
cd vdn-minimax-h3
```

2. Create the environment. We recommend PyTorch 2.13 (`torch.__version__` =
   `2.13.0+cu129`) and installing FlashAttention 4, since our code requires
   [FlexAttention's Flash backend](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).

```bash
conda create -n vdn python=3.12 -y
conda activate vdn
pip install uv

uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
```

3. Install the other packages shown in `pyproject.toml`, `flash-attn-4` included
   (`--prerelease=allow` is needed for its pre-release `nvidia-cutlass-dsl`
   dependency).

```bash
uv pip install --prerelease=allow -e .
```

4. Install the patched Diffusers. The setup script handles everything:

```bash
bash scripts/setup_diffusers.sh
```

## Quick Start — Generate your own video

### Download the weights

Download everything (about 82 GB) into `ckpts/` using

```bash
hf download OpenVDN/vdn-minimax-h3 --local-dir ckpts
```

The layout will look like

```text
ckpts/
  h3-base/             the released MiniMax H3: transformer, video and audio VAEs, schedulers · 72 GB
  stage-b-step-2000/   VDN-H3-50-step: linear_branch/ + adapters/default/ LoRA · 4.3 GB
  stage-dmd-step-250/  VDN-H3-8-step: the above + adapters/turbo/ · 5.1 GB
```

### Your first render

The simplest way to start is by running the model on a single GPU:

```bash
bash scripts/inference/8nfe_tuned_fp8.sh
```

Note that the first run needs to compile all of the kernels, which might take several
minutes. Later runs can reuse the cache.

### Use your own prompt

We provide [three examples](prompts/README.md) and encode them using the
Qwen3-VL-32B VLM. For your own prompt, you should first encode it using the VLM, then
render it through the main diffusion model:

```bash
python src/inference/encode_prompt.py --prompt "..." --out prompts/mine.pt

python src/inference/infer.py \
  --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 \
  render.prompt_file=prompts/mine.pt \
  render.out=results/mine.mp4
```

We strongly recommend rewriting it first using
[H3-Context-IR](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)
or the official
[prompt-writing skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)
before encoding it. This can greatly improve the generated video quality.

### Choosing an inference configuration

We support both single-GPU and multi-GPU inference for the released model. Single-GPU
scripts auto-detect the best kernels for your GPU. Multi-GPU scripts vary for
different hardware (H200, B200) to achieve the best performance.

```bash
bash scripts/inference/8nfe_tuned_fp8.sh                # one GPU
bash scripts/inference/8nfe_tuned_fp8_ulysses_h200.sh   # eight H200s, one node
bash scripts/inference/8nfe_tuned_fp8_ulysses_b200.sh   # eight B200s, one node
```

## Results

We report steady-state denoising speed on the 768p, 14.4-second video generation
workload for the released model using our inference pipeline on H200s and B200s:

**H200:**

| Configuration | GPUs | Seconds/NFE | 50 NFE (VDN-H3-50-step) | 8 NFE (VDN-H3-8-step) |
|---|---:|---:|---:|---:|
| dense MiniMax H3 | 1 | 32.7 | 27.3 min | 4.4 min |
| VDN-H3 FP8 | 1 | 11.2 | 9.4 min | 90.5 s |
| VDN-H3 FP8 Distributed | 8 | 2.29 | 1.9 min | 18.3 s |

**B200:**

| Configuration | GPUs | Seconds/NFE | 50 NFE (VDN-H3-50-step) | 8 NFE (VDN-H3-8-step) |
|---|---:|---:|---:|---:|
| dense MiniMax H3 (cuDNN) | 1 | 16.74 | 13.95 min | 2.23 min |
| VDN-H3 FP8 | 1 | 6.41 | 5.3 min | 51 s |
| VDN-H3 FP8 Distributed | 8 | 1.40 | 1.2 min | 11.23 s |

We exclude model loading, warm-up, VAE decoding, and MP4 encoding. For a live setup,
we recommend running the text prompt rewriter, VAE decoding, and MP4 conversion on
separate machines, so the eight GPUs only denoise.

### Reuse an assembled model for multiple renders

`infer_ulysses.py` can consume a finite JSONL request queue without rebuilding,
re-merging, or re-quantizing the transformer between renders. Each non-empty row must
set a distinct `prompt_file` and `out`; it may override the other `render.*` fields
except `warmup_steps`:

```json
{"prompt_file":"prompts/first.pt","out":"results/first.mp4","seed":42}
{"prompt_file":"prompts/second.pt","out":"results/second.mp4","seed":43}
```

A ready-to-edit queue is available at `examples/requests.jsonl`.

Launch the normal Ulysses command with
`worker.requests_file=requests.jsonl`. On GPUs that cannot hold the transformer and
both decoders together, also set `worker.decoder_cpu_offload=true`. Rank 0 then moves
the already-assembled transformer to CPU for decoding and restores it afterward; the
other ranks keep their transformer replicas resident. The per-output inference record
reports request latency and both swap times separately.

By default the complete transformer is swapped. If memory measurements show that only
part of it must move, `worker.transformer_cpu_offload_blocks=N` limits swapping to the
last `N` transformer blocks and reduces PCIe traffic. Start conservatively: too small a
value can run out of memory during VAE decoding.

Compact FP8 workers can additionally set `precision.fp8.keep_original=false` to drop
the replaced BF16 Linear weights. This makes FP8 conversion non-reversible within the
process, but substantially reduces resident memory and PCIe traffic. The default stays
`true`, preserving the existing `revert_fp8` behavior.

This mode expects pre-encoded prompt files. For an online deployment, keep prompt
encoding and, where possible, VAE decoding outside the denoiser worker.

Reference measurement by **wuyaole** on 8× RTX PRO 5000 72GB, using the released
8-NFE checkpoint at 1344×768 and 345 frames:

| rank-0 decode strategy | warm request | denoise | transformer swap |
|---|---:|---:|---:|
| full transformer CPU offload | 194.60 s | 91.11 s | 44.06 s |
| trailing 25 blocks, second consecutive request | 179.70 s | 91.29 s | 28.85 s |
| trailing 20 blocks | 170.49 s | 90.82 s | 19.61 s |

The numbers include VAE decoding and MP4 encoding but exclude the one-time model
assembly and warm-up. Treat the block count as hardware- and workload-specific; the
25-block result was verified on two consecutive requests, while the 20-block row is a
single completed boundary run.

## Training Recipe

VDN-H3 is trained in three stages based on the frozen dense model, each starting from
the previous stage's final checkpoint. We additionally include a DMD training stage to
align with the community
[few-step distillation LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
The training scripts are located in `src/training/` and `scripts/training/`, and all
training configurations can be found under `configs/training/`.

| Stage | What it trains | Steps |
|---|---|---:|
| A1 | the new linear-attention branch, aligned per layer to the dense model | 200 |
| A2 | the same parameters, end-to-end | 500 |
| B | a LoRA on the QKV and O projections plus the linear branch | 2000 |
| DMD | the 8-step `turbo` LoRA, by DMD2 (no GAN) | 250 |

To reproduce the training process, run:

```bash
bash scripts/training/stage_a1.sh       data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_a2.sh       data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_b.sh        data.index_file=/path/to/video_index.jsonl
bash scripts/training/stage_dmd_vdn.sh  data.index_file=/path/to/video_index.jsonl
```

### Data Preprocess

The trainers read pre-encoded video latents, audio latents, and text latents, following
the H3 standard pipeline. Captions should be written in the same format inference
expects.

The preprocessed data should be placed as follows, with a `video_index.jsonl` carrying
the metadata:

```
<root>/
├── video_index.jsonl       one JSON row per clip, carrying its "latent_path"
├── video/
│   ├── 00000.pt            (24, 102, 48, 84) bf16 — video VAE, normalized space
│   └── ...
├── audio/
│   ├── 00000.pt            (2, 32, 575) bf16 — audio VAE, stereo, 40 latents/s
│   └── ...
└── text/
    ├── 00000.pt            {"prompt_embeds": (L, 5120) bf16,
    │                        "text_token_tags": (L,) int64}
    └── ...
```

`data.index_file` names the jsonl. Only `latent_path` is read from a row; the audio and
text sidecars are found by path arithmetic — same file name, sibling directory — so
`latent_path` must end in `video/<name>.pt`. The reader is
`src/training/dataset_h3_latents.py`.

### Stage-DMD

Stage-DMD is data-free, only requiring the text rows. Its `turbo` LoRA adapter is
initialized from
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora),
which the config expects in `ckpts/external/`:

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir ckpts/external
```

## Acknowledgement

VDN-H3 is built on [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and starts
from its released transformer weights. We also thank
[Diffusers](https://github.com/huggingface/diffusers),
[FlashAttention](https://github.com/Dao-AILab/flash-attention), and
[Triton](https://github.com/triton-lang/triton), on which the optimized inference path
is built.

## BibTeX

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

---

*VDN-Minimax-H3 · Independent architecture study · 2026*

## License

This repository contains the VDN-H3 training and inference code, which is licensed
under the [Apache License, Version 2.0](LICENSE). Copyright 2026 the VDN authors.

**The model weights are not in this repository and are not covered by that license.**
VDN-H3 is a derivative of MiniMax H3, and its weights are distributed separately at
[huggingface.co/OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
under the
[MiniMax H3 Community License Agreement](licenses/MiniMax-H3-Community-License-Agreement.txt).

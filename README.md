# ComfyUI-Qlip

GPU-accelerated inference for diffusion models in ComfyUI, powered by [Qlip](https://thestage.ai) from TheStage AI.

Qlip compiles transformer blocks into optimized engines, delivering significant speedups with full runtime LoRA support. Engines are compiled once and reused across all inference runs.

<p align="center">
  <img src="assets/example_image.png" alt="ComfyUI-Qlip workflow example" width="700">
  <br>
  <em>FLUX.2 Klein 9B with LoRA in ComfyUI. Just add the Qlip Engines Loader and Qlip LoRA Stack nodes to any existing workflow — works with any supported model and any sampler, as long as the compiled engines support the required input shapes.</em>
</p>

## Updates

**2026-04-15** — Wan 2.2 I2V + shared memory + runtime patches + API client
- **Wan 2.2 I2V (14B)** — image-to-video, FP8 + LoRA, two-stage pipeline (high-noise + low-noise), dynamic shapes up to 640x640
- **Shared memory** — new `shared_memory` parameter in QlipEnginesLoader. Multiple loaders with the same group name share one GPU memory pool (size = max, not sum). Enables Wan 2.2 two-transformer setup without doubling VRAM
- **Custom model patches** (`qlip_patch.py`) — place `qlip_patch.py` next to engine files, auto-loaded at runtime. Supports `patch_signatures(dm)` and `patch_caller(dm)` hooks
- **LTX-Video 2.3 (22B)** — text-to-video, FP8 + LoRA, dynamic shapes (512px–1408px, 9–121 frames)
- **Qwen Image Edit** — image editing, BF16 + LoRA, dynamic shapes (512px–1536px), reference image support

## Table of Contents

- [How It Works](#how-it-works)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Precompiled Engines](#precompiled-engines)
- [Nodes](#nodes)
- [Workflows](#workflows)
- [LoRA Details](#lora-details)
- [Compiling New Models](#compiling-new-models)
- [Troubleshooting](#troubleshooting)

## How It Works

- **Compilation**: Compiles PyTorch transformer blocks into fused engines with optimized kernels
- **Dynamic FP8 Quantization**: On-the-fly FP8 quantization reduces memory ~2x while maintaining quality
- **Dynamic LoRA as Input Tensors**: LoRA weights are runtime inputs — hot-swap without recompilation
- **Weight Streaming**: Large models stream weights from CPU/disk, reducing GPU memory requirements
- **Dynamic Shapes**: Single compiled engine supports a range of input resolutions

## Supported Models

| Model | Architecture | Parameters | Type | Precompiled Engines |
|-------|-------------|------------|------|---------------------|
| [**FLUX.2 Klein**](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Dual-stream DiT | 9B | Image | [TheStageAI/Elastic-FLUX-2-Klein](https://huggingface.co/TheStageAI/Elastic-FLUX-2-Klein) |
| [**LTX-Video 2**](https://huggingface.co/Lightricks/LTX-2) (LTXAV) | Audio-Video DiT | 19B | Video | [TheStageAI/Elastic-LTX-2](https://huggingface.co/TheStageAI/Elastic-LTX-2) |
| [**Z-Image-Turbo**](https://huggingface.co/Comfy-Org/z_image_turbo) | NextDiT (Lumina2) | 6B | Image | [TheStageAI/Elastic-Z-Image-Turbo](https://huggingface.co/TheStageAI/Elastic-Z-Image-Turbo) |
| [**LTX-Video 2.3**](https://huggingface.co/Lightricks/LTX-2.3) (LTXAV) | Audio-Video DiT | 22B | Video (t2v) | [TheStageAI/Elastic-LTX-2.3](https://huggingface.co/TheStageAI/Elastic-LTX-2.3) |
| [**Qwen Image Edit**](https://huggingface.co/Comfy-Org/Qwen_Image_Edit) | Joint DiT | 14B | Image Edit | [TheStageAI/Elastic-Qwen-Image-Edit](https://huggingface.co/TheStageAI/Elastic-Qwen-Image-Edit) |
| [**Wan 2.2 I2V**](https://github.com/Wan-Video/Wan2.2) | DiT | 14B | Video (i2v) | [TheStageAI/Elastic-Wan2.2-I2V](https://huggingface.co/TheStageAI/Elastic-Wan2.2-I2V) |

More models coming soon — stay tuned for updates.

> **Note:** LTX-Video 2.3, Qwen Image Edit, and Wan 2.2 I2V require a specific ComfyUI version and custom node setup. See each model's HuggingFace repo for installation instructions and compatible ComfyUI commit.

### Model Details

#### FLUX.2 Klein (9B)

Image generation model. Compiled with separate engines for **distilled** and **base** variants.

| Variant | CFG | Batch | Steps | Static Sizes | Dynamic Range |
|---------|-----|-------|-------|-------------|---------------|
| 9B **distilled** | 1.0 | 1 | 4 | 1024x1024, 1024x864 | 512px — 768px — 1024px |
| 9B **base** | 3.5 | 2 | 50 | 1024x1024, 1024x864 | 512px — 768px — 1024px |

Distilled and base models require separate engines (different batch sizes). Dynamic range supports any resolution from 512x512 to 1024x1024 (including non-square). Static profiles provide optimal performance at the listed sizes.

> **H100 vs Blackwell — different mode.** The H100 engines above are plain
> **text-to-image** generation. The **Blackwell** FLUX.2 Klein engines (RTX 5090 +
> B200) are compiled in **image-edit mode** (`--edit --max-ref-images 1`): they
> accept one reference image/latent in addition to the prompt, so `img_seq_len`
> carries the extra ref tokens. Use the edit workflow with these engines (a
> text-only graph won't supply the reference-latent input the engine expects).

#### LTX-Video 2 (19B)

Audio-video generation model. Currently only the **distilled** variant is compiled (cfg=1.0, batch=1). Only **image-to-video (i2v)** workflow is supported.

| Variant | CFG | Batch | Workflow | Static Sizes (WxHxFrames) | Dynamic Range |
|---------|-----|-------|----------|--------------------------|---------------|
| 19B **distilled** | 1.0 | 1 | i2v | 768x512x41, 1280x720x121, 1408x896x121 | 512px—768px—1408px, 41—121 frames |

Video resolution and frame count are both dynamic. Audio tokens are computed automatically from the frame count.

#### Z-Image-Turbo

Image generation model. Compiled with cfg=1.0, batch=1, static sizes only.

| CFG | Batch | Static Sizes |
|-----|-------|-------------|
| 1.0 (turbo) | 1 | 1024x1024, 1024x768, 768x1024 |

#### LTX-Video 2.3 (22B)

Text-to-video model. 22B full-scale (non-distilled) LTXAV architecture with audio+video dual-stream attention. Compiled as t2v (text-to-video) with fixed batch=2.

| Variant | CFG      | Batch | Engine Type | Static Sizes (WxHxFrames) | Dynamic Range |
|---------|----------|-------|-------------|--------------------------|---------------|
| 22B distilled FP8 + LoRA | 1.0, 4.0 | 2 (fixed) | t2v | 768x512x41, 1280x720x121, 1408x896x121 | 512px—768px—1408px, 9—121 frames |

Video resolution and frame count are both dynamic. LoRA rank 1–256.

#### Qwen Image Edit

Image editing model. Takes a reference image + text prompt → generates edited image. Uses joint concat (txt+img → single sequence) with fixed text length padding.

| Variant | CFG | Batch | Static Sizes | Dynamic Sizes | Ref Images |
|---------|-----|-------|-------------|---------------|------------|
| BF16 + LoRA | 1.0 | 1 | 768, 1024, 1328, 1536 | All WxH from {512, 768, 1024, 1328, 1536} | 1 |

Text padded to 1536 tokens. Reference method: `index_timestep_zero`. LoRA rank 1–256. Requires `qlip_patch.py` in engines directory for inference.

> **ComfyUI version:** LTX-Video 2.3, Qwen Image Edit, and Wan 2.2 I2V require ComfyUI commit **`b615af1c`** (newer than the default `048dd2f3` used for other models). Use `git checkout b615af1c` in your ComfyUI directory before running these models.

#### Wan 2.2 I2V

Image-to-video generation model. Two-stage pipeline: high-noise transformer generates initial video, low-noise transformer refines it. Both transformers are compiled separately. Uses `shared_memory` option in QlipEnginesLoader to share one GPU memory pool between the two transformers.

| Variant | Transformer | CFG | Batch | Dynamic Sizes | Num frames |
|---------|------------|-----|-------|---------------|------------|
| FP8 + LoRA | High-noise | 1.0 | 1 | Up to 640x640 | 81         |
| FP8 + LoRA | Low-noise | 1.0 | 1 | Up to 640x640 | 81         |

Each transformer has its own engines directory and QlipEnginesLoader node. Set `shared_memory="wan"` on both loaders to share one GPU memory pool (reduces VRAM usage since the two transformers never run simultaneously).

## Benchmarks

All measurements: single image/video generation, batch size 1, H100, torch 2.8.0, **warm run** (second run, engines already loaded). Current precompiled engines include LoRA support, which adds minor overhead (~5-15%) compared to non-LoRA engines. Non-LoRA engines with faster inference will be available in a future release.

### Image Generation (1024x1024, 20 steps for flux, 8 steps for z-image, cfg=1)

| Model | Method | Time (s) | Speedup |
|-------|--------|----------|---------|
| **FLUX.2 Klein 9B** | Eager (PyTorch) | 3.743 | 1.0x |
| | torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 3.064 | 1.22x |
| | **Qlip BF16 + LoRA** | 2.944 | 1.27x |
| | **Qlip FP8 + LoRA** | 2.215 | **1.69x** |
| **Z-Image-Turbo** | Eager (PyTorch) | 1.436 | 1.0x |
| | torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 0.999 | 1.44x |
| | [SGLDiffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion) | 0.920 | 1.56x |
| | **Qlip BF16 + LoRA** | 0.957 | 1.50x |
| | **Qlip FP8 + LoRA** | 0.773 | **1.86x** |

### Video Generation (LTX-Video 2, 1280x720, 121 frames, 8 basic sampler steps + 3 upsampling steps)

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 7.829 | 1.0x |
| torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 5.330 | 1.47x |
| **Qlip BF16 + LoRA** | 6.020 | 1.30x |
| **Qlip FP8 + LoRA** | 5.051 | **1.55x** |

### Video Generation (LTX-Video 2.3, 1280x720, 8 basic sampler steps + 3 upsampling steps, cfg 1.0, 121 frames, t2v, H100)

Transformer pass time only (2nd run, warm). 22B model, FP8 + LoRA.

| Method | Transformer Time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 15.188 | 1.0x |
| **Qlip FP8 + LoRA** | 8.537 | **1.78x** |

### Image Editing (Qwen Image Edit, cfg 1.0, 4 steps 1328x1328, H100)

Transformer pass time only (2nd run, warm). 1 reference image.

| Method | Transformer Time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 3.135 | 1.0x |
| **Qlip BF16 + LoRA** | 2.389 | **1.31x** |

### Video Generation (Wan 2.2 I2V, 640x640, 81 frames, 4 steps, cfg 1.0, H100)

Transformer pass time only (2nd run, warm). Two-stage pipeline (high-noise + low-noise).

| Method | Transformer Time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 18.573 | 1.0x |
| **Qlip FP8 + LoRA** | 12.035 | **1.54x** |

> All benchmarked engines include runtime LoRA support. LoRA adds ~5-15% overhead due to additional MatMul operations per layer. Faster non-LoRA engines will be available in a future update.
>
> More GPUs (B200, L40S, RTX 5090) coming soon.

### Blackwell engines — planned (in progress)

We are recompiling the lineup for **Blackwell**: RTX 5090 (sm_120a) for models that
fit in ~31 GB, B200 (sm_100) for the rest, and all of them on B200 for a full
sm_100 set. New on Blackwell: **NVFP4 (FP4)** weights (~4× smaller than BF16, native
fast path) in addition to FP8, plus FP4 attention on the 5090. Single image/video
generation, batch size 1, warm run (engines already loaded), LoRA-enabled engines.
Measurements filled in as they come; `_TBD_` = not measured yet. FLUX.2 Klein is
the **distilled** variant — **4 sampling steps** (distillation lets it converge in
4 vs ~50 for the base model), cfg 1.0, batch 1; times are the full warm generation
at 1024x2048.

#### FLUX.2 Klein 9B (edit) — RTX 5090, 1024x2048 (distilled, 4 steps)

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 10.144 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 7.919 | 1.28x |
| **Qlip FP8-static + LoRA** | 7.520 | 1.35x |
| **Qlip NVFP4 + LoRA** | 6.058 | 1.67x |
| **Qlip NVFP4 + FP4-attention + LoRA** | 3.299 | 3.07x |
| **Qlip NVFP4 + FP4-attention (no LoRA)** | **2.810** | **3.61x** |

> **LoRA overhead (measured).** `+ LoRA` engines carry runtime LoRA support (extra
> per-layer MatMuls), and that cost is paid **even when no LoRA is applied** — the
> engine still gets a zero-filled `lora_packed` tensor (LoRA is a compile-time graph
> input). Direct measurement on the NVFP4 + FP4-attention engine:
> **3.299 s (LoRA) → 2.810 s (no-LoRA) = −0.489 s (~15%)**. So a **no-LoRA engine is
> always the fastest** for a given precision. Applying the same ~0.49 s delta as an
> estimate, the no-LoRA floors for the other Klein engines would be roughly:
> FP8-dynamic ≈ 7.43 s, FP8-static ≈ 7.03 s, NVFP4 ≈ 5.57 s (projected — only the
> NVFP4+FP4-attn no-LoRA row is actually built/measured).
>
> **We beat eager SageAttention3.** Eager NVFP4 `.safetensors` + SageAttention3 is
> **3.226 s** (the fastest non-Qlip path). Our Qlip **NVFP4 + FP4-attention** engine
> is **3.299 s with LoRA** (essentially on par despite carrying LoRA) and
> **2.810 s without LoRA — ~13% faster than eager+SageAttention3**, at a **3.61×**
> speedup over BF16 eager. (FP4 attention here is a dedicated `QlipFP4Attention`
> plugin node that replaces the SDPA pattern; the block's Linears stay NVFP4.)

#### Z-Image-Turbo 6B — RTX 5090

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 2.333 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 1.644 | 1.42x |
| **Qlip NVFP4 + LoRA** | 1.143 | 2.04x |
| **Qlip NVFP4 + FP4-attention + LoRA** | 1.017 | 2.29x |
| **Qlip NVFP4 + FP4-attention (no LoRA)** | 0.884 | 2.64x |

#### Reference: external NVFP4 `.safetensors` (eager, RTX 5090)

Pre-quantized NVFP4 weight checkpoints run in **eager PyTorch** (not our Qlip
engines) — a baseline to compare our engines against, measured once plain and once
with **SageAttention3** (`sageattn3_blackwell`, the eager FP4 attention kernel for
consumer Blackwell). Order: NVFP4 alone, then + SageAttention3.

Checkpoints:
- **FLUX.2 Klein 9B NVFP4** — [`black-forest-labs/FLUX.2-klein-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) (`flux-2-klein-9b-nvfp4.safetensors`, gated)
- **Z-Image-Turbo SVDQuant FP4** — [`nunchaku-ai/nunchaku-z-image-turbo`](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo) (`svdq-fp4_r128-z-image-turbo.safetensors`, `svdq-fp4_r32-z-image-turbo.safetensors`). These load **only via the nunchaku custom node + `nunchaku` runtime package** (SVDQuant diffusers-layout weights). A plain `UNETLoader` fails with `KeyError: noise_refiner.0.attention.to_k.weight` — the checkpoint stores unfused `to_q/to_k/to_v` while ComfyUI's Z-Image expects fused `attention.qkv`.

**FLUX.2 Klein 9B NVFP4 (edit) — eager, 1024x2048 (distilled, 4 steps)**

Speedup is vs the BF16 eager baseline (10.144s, top table).

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager NVFP4 `.safetensors` | 5.934 | 1.71x |
| Eager NVFP4 `.safetensors` + SageAttention3 | 3.226 | 3.14x |
| Eager NVFP4 `.safetensors` + SageAttention3 (compiled) | 3.262 | 3.11x |

> Compiling the SageAttention3 op gives **no gain on Klein** (3.262 s vs 3.226 s —
> within noise / slightly worse), unlike Z-Image where it helps (see below). "compiled"
> = `torch.compile` on the SageAttention3 op only, not the whole model.

> The SageAttention3 rows use the eager `sageattn3` kernel (and base
> `sageattention` for the KJNodes `Patch Sage Attention KJ` path) on consumer
> Blackwell — built from source per the SageAttention repo. These are an external
> eager baseline, separate from the Qlip engines.

**Z-Image-Turbo SVDQuant FP4 (r128) — eager**

Speedup is vs the BF16 eager baseline (2.333s, Z-Image table above).

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager SVDQuant FP4 `.safetensors` | 0.988 | 2.36x |
| Eager SVDQuant FP4 `.safetensors` + SageAttention3 | 0.951 | 2.45x |
| Eager SVDQuant FP4 `.safetensors` + SageAttention3 (compiled) | 0.842 | 2.77x |

> "compiled" = `torch.compile` applied to the **SageAttention3 attention op only**
> (its `allow_compile` path), not the whole model.

#### FLUX.2 Klein 9B (edit) — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | _TBD_ | 1.0x |
| **Qlip FP8-dynamic + LoRA** | _TBD_ | _TBD_ |
| **Qlip NVFP4 + LoRA** | _TBD_ | _TBD_ |

#### Z-Image-Turbo 6B — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | _TBD_ | 1.0x |
| **Qlip FP8-dynamic + LoRA** | _TBD_ | _TBD_ |
| **Qlip NVFP4 + LoRA** | _TBD_ | _TBD_ |

#### LTX-Video 2 19B (i2v) — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | _TBD_ | 1.0x |
| **Qlip FP8-dynamic + LoRA** | _TBD_ | _TBD_ |
| **Qlip NVFP4 + LoRA** | _TBD_ | _TBD_ |

#### Qwen Image Edit 20B — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | _TBD_ | 1.0x |
| **Qlip FP8-dynamic + LoRA** | _TBD_ | _TBD_ |
| **Qlip NVFP4 + LoRA** | _TBD_ | _TBD_ |

#### LTX-Video 2.3 22B (t2v) — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | _TBD_ | 1.0x |
| **Qlip FP8-dynamic + LoRA** | _TBD_ | _TBD_ |
| **Qlip NVFP4 + LoRA** | _TBD_ | _TBD_ |

> Routing: **≤14B → 5090, ≥19B → B200**. **FLUX.2 Klein on Blackwell is built in
> image-edit mode** (`--edit --max-ref-images 1`) — unlike the H100 engines, which
> are plain text-to-image (use the edit workflow with the Blackwell Klein engines).
> **LTX-2 (i2v) and LTX-2.3 (t2v) are both kept** — different tasks, not redundant.
> **Wan 2.2 is not part of this Blackwell round.** FP4 attention is **5090-only**
> (the SageAttention3 kernel targets sm_120a; B200 keeps FP8/BF16 attention). New
> precompiled-engine `hf_repo` paths (e.g. `.../models/RTX5090/...`,
> `.../models/B200/...`) will be published alongside the existing `H100` ones.

> **⚠ ComfyUI commit for ALL Blackwell engines — `b615af1c`.** Unlike the H100
> engines (which use a mix of `048dd2f3` for FLUX.2 Klein / LTX-2 / Z-Image and
> `b615af1c` for LTX-2.3 / Qwen / Wan), **every Blackwell engine — on both the
> RTX 5090 and the B200 — is compiled against ComfyUI `b615af1c`**, including
> FLUX.2 Klein and Z-Image-Turbo. Engines are tied to the ComfyUI block structure,
> so to run a Blackwell engine you must `git checkout b615af1c` in your ComfyUI
> (it is backwards-compatible with the older models). Compile and inference must use
> the same commit. _(Verified on the 5090 build pod: `v0.18.1-53-gb615af1c`.)_

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (Hopper, Blackwell, Ada Lovelace)
- ComfyUI installed and working (see below if starting from scratch)

### Step 1: Install ComfyUI-Qlip nodes

> **ComfyUI version matters.** Different models require different ComfyUI commits:
> - FLUX.2 Klein, LTX-Video 2, Z-Image-Turbo → commit **`048dd2f3`** (default below)
> - LTX-Video 2.3, Qwen Image Edit, Wan 2.2 I2V → commit **`b615af1c`** (newer)
>
> Check the model's HuggingFace repo for the exact commit. If you need multiple models from different commits, use the **newer** commit (`b615af1c`) — it is backwards-compatible with older models.

**From scratch** (no ComfyUI yet):
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git checkout 048dd2f3  # or b615af1c for LTX-2.3 / Qwen Image Edit / Wan 2.2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/TheStageAI/ComfyUI-Qlip
cd ..
```

**Existing ComfyUI** — activate your venv and clone:
```bash
source /path/to/ComfyUI/venv/bin/activate
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/TheStageAI/ComfyUI-Qlip
```

> If ComfyUI-Qlip is published to the Comfy Registry, you can also install via `comfy node install comfyui-qlip`.

### Step 2: Install Qlip dependencies

From the ComfyUI-Qlip directory (with the same venv activated):

```bash
pip install -r custom_nodes/ComfyUI-Qlip/requirements.txt
```

This installs `qlip.core[nvidia]` from the TheStage AI package registry.

> **Which `requirements.txt`?** Two are shipped, for two different GPU generations:
>
> | File | Target | torch | CUDA | `tensorrt-cu12` | qlip extra |
> |------|--------|-------|------|----------|-----------|
> | [`requirements.txt`](requirements.txt) | **H100 / Hopper / Ada (CUDA 12)** | 2.9.1 | 12.x | 10.13.3.9 | `qlip.core[nvidia]` |
> | [`requirements_new.txt`](requirements_new.txt) | **Blackwell (5090 sm_120a / B200 sm_100, CUDA 13)** | 2.12.0 (cu130) | 13.0 | 10.15.1.29 | `qlip.core[blackwell]` |
>
> Blackwell needs CUDA 13, the `tensorrt-cu12` ≥ 10.13 wheel (FP4 support), and the
> `blackwell` qlip extra. Install PyTorch from the cu130 index **first**, then the
> requirements file (see the header note inside `requirements_new.txt`).
>
> Key Blackwell pins (full list in `requirements_new.txt`): torch 2.12.0+cu130 /
> torchvision 0.27.0 (from the cu130 index), `tensorrt-cu12` 10.15.1.29, onnx 1.20.1
> / onnxscript 0.7.0, diffusers 0.38.0, transformers 5.10.2. The hard requirements
> for NVFP4 are **CUDA 13 + `tensorrt-cu12` ≥ 10.15 + an FP4-capable qlip** (one that
> exposes `NVIDIA_NVFP4_W4A4`).

### Step 3: Setup TheStage API token

Get your token at [app.thestage.ai](https://app.thestage.ai). Required for Qlip engine access.

```bash
pip install thestage
thestage config set --access-token <YOUR_API_TOKEN>
```

### (Optional) Download models

If you don't have the required models yet, use the download scripts from [`scripts/`](scripts/):

```bash
export COMFYUI_PATH=/path/to/ComfyUI

# Original models (ComfyUI commit 048dd2f3)
bash custom_nodes/ComfyUI-Qlip/scripts/download_z_image_turbo_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_ltx_2_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_flux_klein_models.sh

# New models (ComfyUI commit b615af1c)
bash custom_nodes/ComfyUI-Qlip/scripts/download_ltx_2_3_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_qwen_image_edit_models.sh
```

FLUX.2 Klein is a gated model — requires `huggingface-cli login` and license acceptance. Override the HuggingFace cache location with `HF_HUB_CACHE` if needed. Scripts skip already-downloaded files.

### Step 4: Launch ComfyUI

```bash
python main.py --listen 0.0.0.0 --port 8188
```

> Always activate the same venv before launching: `source venv/bin/activate`

### (Optional) Additional custom nodes for benchmarking

```bash
cd /path/to/ComfyUI/custom_nodes

# KJNodes — provides torch.compile node for comparison benchmarks
git clone https://github.com/kijai/ComfyUI-KJNodes

# SGLDiffusion — SGLang-based acceleration (Z-Image-Turbo, FLUX Klein)
git clone https://github.com/sgl-project/ComfyUI_SGLDiffusion
pip install "sglang[diffusion]"
```

### 6. (Optional) Flash Attention for Hopper GPUs

```bash
pip install flash-attn --no-build-isolation
```

### (Required for FP4-attention engines) Build the `fp4attn` plugin — RTX 5090 only

> **Extra step — read this if you use any `…-fp4attn…` engine.** FP4-attention
> engines contain a `QlipFP4Attention` plugin node. Unlike plain FP8/NVFP4 engines
> (which load with just `qlip.core[nvidia]`), an fp4-attention engine **will not
> load** until the `fp4attn` plugin is available — the loader raises an actionable
> error otherwise. The plugin is **RTX 5090 / consumer Blackwell (sm_120a) only**
> (its kernel targets `sm_120a`; it does **not** build/run on H100 sm_90 or B200
> sm_100). Plain FP8 / NVFP4 (no fp4-attn) engines need none of this.

Prerequisites: CUDA 13 toolkit (`nvcc`), `tensorrt-cu12` ≥ 10.15, PyTorch
2.11+cu13x. The plugin is **self-contained** — its FP4 attention CUDA kernel is
compiled into the plugin `.so` (vendored build), so it needs **no** separate
SageAttention package at build or run time.

```bash
# 1. Bootstrap build prerequisites (downloads TRT C++ headers + CUTLASS source,
#    checks nvcc/ninja). Ships with qlip.core.
qlip-setup-plugins          # or: python -m qlip.plugins setup

# 2. The plugin builds JIT on first import (~30-90s nvcc, cached afterwards). Verify:
python -c "import torch; from qlip.plugins.fp4attn import ensure_plugin_registered; \
           ensure_plugin_registered(); print('fp4attn plugin OK')"
```

If anything is missing, `ensure_plugin_registered()` prints exactly what to fix and
points back at `qlip-setup-plugins` (see `qlip/plugins/BUILD.md` /
`qlip/plugins/README.md` in the qlip package). At inference, ComfyUI's
`QlipEnginesLoader` calls `ensure_plugin_registered()` automatically — once the
plugin is built, fp4-attn engines just work.

> SageAttention3 is a **separate, optional** thing — only for the eager-PyTorch
> attention path (KJNodes `Patch Sage Attention KJ`), not for these compiled
> engines. The `fp4attn` plugin does not depend on it.

## Precompiled Engines

Precompiled engines are hosted on HuggingFace. The **Qlip Engines Loader** node can download them automatically — just set the `hf_repo` input:

| Model | `hf_repo` value |
|-------|----------------|
| FLUX.2 Klein 9B Distilled BF16 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-9b_lora` |
| FLUX.2 Klein 9B Distilled FP8 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-9b-fp8_lora` |
| FLUX.2 Klein 9B Base BF16 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-base-9b_lora` |
| FLUX.2 Klein 9B Base FP8 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-base-9b-fp8_lora` |
| LTX-2 19B Distilled BF16 + LoRA | `TheStageAI/Elastic-LTX-2:models/H100/ltx-2-19b-distilled_lora` |
| LTX-2 19B Distilled FP8 + LoRA | `TheStageAI/Elastic-LTX-2:models/H100/ltx-2-19b-distilled-fp8_lora` |
| Z-Image-Turbo BF16 + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/H100/z-image-turbo_lora` |
| Z-Image-Turbo FP8 + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/H100/z-image-turbo-fp8_lora` |
| LTX-Video 2.3 22B Distilled FP8 + LoRA (t2v) | `TheStageAI/Elastic-LTX-2.3:models/H100/ltx-2.3-22b-distilled-fp8_lora` |
| Qwen Image Edit BF16 + LoRA | `TheStageAI/Elastic-Qwen-Image-Edit:models/H100/qwen-image-edit-bf16_lora` |
| Wan 2.2 I2V High-Noise FP8 + LoRA | `TheStageAI/Elastic-Wan2.2-I2V:models/H100/wan-i2v-high-noise-fp8_lora` |
| Wan 2.2 I2V Low-Noise FP8 + LoRA | `TheStageAI/Elastic-Wan2.2-I2V:models/H100/wan-i2v-low-noise-fp8_lora` |

The format is `org/repo:path/to/engines`. Engines are downloaded once and cached.

Alternatively, download manually with `huggingface-cli`:

```bash
# Example: FLUX.2 Klein 9B FP8 + LoRA
huggingface-cli download TheStageAI/Elastic-FLUX-2-Klein \
    --local-dir ./engines/flux-klein \
    --include "models/H100/klein-9b-fp8_lora/*"
```

Then point `engines_path` to the downloaded directory.

## Nodes

### Qlip Engines Loader

Loads pre-compiled engines and replaces transformer blocks at runtime. Caches engines across runs — first load takes a few seconds, subsequent runs are instant.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model from any loader (UNETLoader, CheckpointLoaderSimple, etc.) |
| `engines_path` | STRING | No | `""` | Path to directory with `.qlip`/`.engine` files |
| `hf_repo` | STRING | No | `""` | HuggingFace repo with engines, e.g. `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-9b-fp8_lora` |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack from `Qlip LoRA Stack` node(s) |
| `cuda_graph` | BOOLEAN | No | `False` | Enable CUDA Graph capture — reduces kernel launch overhead for faster inference |

LoRA is auto-detected: if `lora_config.json` exists in the engines directory or a `lora_stack` is connected, LoRA support is enabled automatically. No manual toggle needed.

**Output:** `MODEL`

### Qlip LoRA Stack

Builds a chainable list of LoRA entries. Each node adds one LoRA file. Chain multiple nodes via `prev_stack` to stack LoRAs.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `lora_path` | STRING | Yes | `""` | Path to LoRA `.safetensors` file |
| `strength` | FLOAT | Yes | `1.0` | Strength multiplier (`-10.0` to `10.0`) |
| `prev_stack` | QLIP_LORA_STACK | No | | Previous stack to extend |

**Output:** `QLIP_LORA_STACK`

### Qlip LoRA Switch

Enables or disables LoRA at runtime without reloading engines. Use after `Qlip Engines Loader` with LoRA-enabled engines.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model with loaded engines |
| `enable` | BOOLEAN | Yes | `True` | Enable or disable LoRA |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack to load (when enabling) |

**Output:** `MODEL`

### Qlip Timer Start

Records a start timestamp. Place **before** the node(s) you want to measure.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `passthrough` | * | Yes | | Any data — passed through unchanged |
| `timer_name` | STRING | Yes | `"timer_1"` | Name for this timer (must match Timer Stop) |
| `cuda_sync` | BOOLEAN | No | `True` | Call `torch.cuda.synchronize()` for accurate GPU timing |

**Output:** same data as `passthrough`

### Qlip Timer Stop

Records elapsed time since the matching Timer Start and displays it. Place **after** the measured node(s).

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `passthrough` | * | Yes | | Any data — passed through unchanged |
| `timer_name` | STRING | Yes | `"timer_1"` | Name for this timer (must match Timer Start) |
| `cuda_sync` | BOOLEAN | No | `True` | Call `torch.cuda.synchronize()` for accurate GPU timing |

**Output:** same data as `passthrough`. Elapsed time is shown in the node UI and printed to console.

### Qlip Timer Report

Displays a summary table of all timer measurements. Connect the `trigger` input to any node output that executes after all Timer Stop nodes.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `trigger` | * | No | | Connect any output to ensure execution order |
| `track_cold_start` | BOOLEAN | No | `False` | Show cold start (first run) comparison — displays delta % vs first measurement |

**Output:** none (display only). Results are shown in the node UI and printed to console.

Results auto-reset between workflow runs. Cold start values persist across runs for comparison.

## Workflows

Ready-to-use ComfyUI workflow files are in [`workflows/`](workflows/):

| File | Description | ComfyUI commit |
|------|-------------|----------------|
| `Flux-Klein.json` | FLUX.2 Klein image generation | `048dd2f3` |
| `video_ltx2_i2v_distilled.json` | LTX-2 19B image-to-video | `048dd2f3` |
| `z-image-turbo.json` | Z-Image-Turbo image generation | `048dd2f3` |
| `video_ltx2_3_t2v.json` | LTX-Video 2.3 22B text-to-video | `b615af1c` |
| `qwen_image_edit.json` | Qwen Image Edit | `b615af1c` |

Model download scripts are in [`scripts/`](scripts/):

| Script | Description |
|--------|-------------|
| `download_flux_klein_models.sh` | FLUX.2 Klein (diffusion model, text encoder, VAE). Requires HF token |
| `download_ltx_2_models.sh` | LTX-2 19B (checkpoints, text encoder, LoRAs, upscaler) |
| `download_z_image_turbo_models.sh` | Z-Image-Turbo (diffusion model, text encoder, VAE, LoRA) |
| `download_ltx_2_3_models.sh` | LTX-Video 2.3 22B (checkpoint, Gemma text encoder, LoRAs, upscaler) |
| `download_qwen_image_edit_models.sh` | Qwen Image Edit (diffusion model, text encoder, LoRA) |

To download models for a specific workflow:

```bash
export COMFYUI_PATH=/path/to/ComfyUI

# Original models (ComfyUI 048dd2f3)
bash custom_nodes/ComfyUI-Qlip/scripts/download_flux_klein_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_ltx_2_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_z_image_turbo_models.sh

# New models (ComfyUI b615af1c)
bash custom_nodes/ComfyUI-Qlip/scripts/download_ltx_2_3_models.sh
bash custom_nodes/ComfyUI-Qlip/scripts/download_qwen_image_edit_models.sh
```

Scripts skip already-downloaded files, so they are safe to re-run. Override the HuggingFace cache location with `HF_HUB_CACHE` if needed.

### Basic (no LoRA)

```
UNETLoader / CheckpointLoaderSimple (model.safetensors)
  -> Qlip Engines Loader (engines_path=...)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

### With LoRA

```
Qlip LoRA Stack (lora.safetensors, strength=1.0)  ─────────────────────┐
                                                                         |
UNETLoader / CheckpointLoaderSimple (model.safetensors)                  |
  -> Qlip Engines Loader (engines_path=..., lora_stack=^)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

LoRA support is auto-detected from `lora_config.json` in the engines directory. No manual toggle needed.

### Multi-LoRA stacking

```
Qlip LoRA Stack (style_lora.safetensors, 0.8)
  -> Qlip LoRA Stack (detail_lora.safetensors, 0.5, prev_stack=^)
    -> Qlip Engines Loader (engines_path=..., lora_stack=^)
```

Multiple LoRAs are stacked — their ranks concatenate. Total rank must fit within `--max-lora-rank` used at compilation time.

## LoRA Details

LoRA weights are **runtime inputs** to compiled engines, not baked into weights. The `lora_packed` tensor shape `[num_layers, rank, max_features, 2]` holds A and B matrices. The rank dimension is dynamic, allowing different LoRA files without recompilation.

**Caching behavior:**
- Engines loaded once per path, reused across runs
- LoRA weights hot-swapped in-place when stack changes
- Same LoRA between runs — weights already loaded, swap skipped
- LoRA removed — packed tensors zeroed, no engine reload

**Constraints:**
- Engines compiled without `--lora` cannot accept LoRA at runtime
- Engines compiled with `--lora` work both with and without LoRA (auto-detected via `lora_config.json`)
- LyCORIS / LoKR format is not supported

## Compiling New Models

You can compile any ComfyUI-compatible model into Qlip engines. This is useful when:
- You want to accelerate a model not in the precompiled engines list
- You need engines for a specific GPU (engines are hardware-specific)
- You want custom resolution ranges or LoRA support

### Using Claude Code (recommended)

The easiest way to add a new model is with [Claude Code](https://claude.ai/claude-code). This repository includes an [agent skill](skills/qlip-model-compiler/SKILL.md) that knows how to write compilation scripts, handle model-specific patches, and run compilation.

**What to provide:**

1. **ComfyUI workflow JSON** — export from ComfyUI. **Important**: expand all subgraphs/groups before exporting so every node is visible in the JSON file
2. **Path to ComfyUI** installation (so the agent can read model source code)
3. **Model path** and **LoRA path** (if needed)
4. **Target GPU** and **target resolutions**
5. **Server SSH access** (if compiling remotely)

**Example prompt:**
```
I want to compile my model with Qlip.

Workflow: /path/to/my_workflow.json
ComfyUI: /path/to/ComfyUI
Model: my_model.safetensors (UNETLoader, in models/diffusion_models/)
LoRA: my_lora.safetensors (in models/loras/)
Text encoder: my_encoder.safetensors (CLIPLoader)
VAE: my_vae.safetensors
Target: H100, 512-1024px, FP8 quantization, LoRA support
```

Claude will analyze the workflow and model source code, write download/compile/benchmark scripts, run compilation, and add inference support to the nodes if needed.

### Manual Compilation

If you prefer to write scripts manually, see the [compilation skill reference](skills/qlip-model-compiler/SKILL.md) for the full API reference including imports, function signatures, patching patterns, and bash wrapper templates.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Shape mismatch errors | Ensure engines match the model (LoRA-enabled engines need `lora_config.json` in the engines directory) |
| Slow first run | Normal — engine loading takes a few seconds on first use |
| Engines don't work after GPU change | Engines are GPU-specific — recompile after changing hardware |
| FP8 quality looks bad | Try `--unfuse-qkv` and `--skip-first-blocks 1 --skip-last-blocks 1` |
| LoRA not taking effect | Check that `lora_config.json` exists in engines directory and LoRA rank ≤ max compiled rank |
| `flash_attn 3 package is not installed` | Install with `pip install flash-attn --no-build-isolation` (optional, Hopper only) |

## License

Proprietary. Powered by [TheStage AI](https://thestage.ai).

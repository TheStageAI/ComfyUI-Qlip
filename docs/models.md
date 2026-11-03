# Supported models and engines

Which models ComfyUI-Qlip accelerates, which precompiled engines exist for each GPU, and how to get them.

## Supported models

| Model | Architecture | Parameters | Type | Precompiled engines |
|-------|-------------|------------|------|---------------------|
| [**FLUX.2 Klein**](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Dual-stream DiT | 9B | Image | [TheStageAI/Elastic-FLUX-2-Klein](https://huggingface.co/TheStageAI/Elastic-FLUX-2-Klein) |
| [**LTX-Video 2**](https://huggingface.co/Lightricks/LTX-2) (LTXAV) | Audio-Video DiT | 19B | Video | [TheStageAI/Elastic-LTX-2](https://huggingface.co/TheStageAI/Elastic-LTX-2) |
| [**Z-Image-Turbo**](https://huggingface.co/Comfy-Org/z_image_turbo) | NextDiT (Lumina2) | 6B | Image | [TheStageAI/Elastic-Z-Image-Turbo](https://huggingface.co/TheStageAI/Elastic-Z-Image-Turbo) |
| [**LTX-Video 2.3**](https://huggingface.co/Lightricks/LTX-2.3) (LTXAV) | Audio-Video DiT | 22B | Video (t2v) | [TheStageAI/Elastic-LTX-2.3](https://huggingface.co/TheStageAI/Elastic-LTX-2.3) |
| [**Qwen Image Edit**](https://huggingface.co/Comfy-Org/Qwen_Image_Edit) | Joint DiT | 14B | Image edit | [TheStageAI/Elastic-Qwen-Image-Edit](https://huggingface.co/TheStageAI/Elastic-Qwen-Image-Edit) |
| [**Wan 2.2 I2V**](https://github.com/Wan-Video/Wan2.2) | DiT | 14B | Video (i2v) | [TheStageAI/Elastic-Wan2.2-I2V](https://huggingface.co/TheStageAI/Elastic-Wan2.2-I2V) |
| [**Krea 2 Turbo**](https://huggingface.co/krea/Krea-2-Turbo) | Single-stream DiT | — | Image | [TheStageAI/Elastic-Krea-2](https://huggingface.co/TheStageAI/Elastic-Krea-2) |

Each model's Hugging Face repo lists its installation steps and compatible ComfyUI commit.

### FLUX.2 Klein (9B)

Image generation. The **distilled** and **base** variants need separate engines because their
batch sizes differ.

| Variant | CFG | Batch | Steps | Static sizes | Dynamic range |
|---------|-----|-------|-------|-------------|---------------|
| 9B **distilled** | 1.0 | 1 | 4 | 1024x1024, 1024x864 | 512px — 768px — 1024px |
| 9B **base** | 3.5 | 2 | 50 | 1024x1024, 1024x864 | 512px — 768px — 1024px |

The dynamic range covers any resolution from 512x512 to 1024x1024, including non-square. The
static profiles are fastest at the listed sizes.

> **H100 and Blackwell engines run in different modes.** The H100 engines are text-to-image. The
> Blackwell engines (RTX 5090 and B200) are compiled in image-edit mode
> (`--edit --max-ref-images 1`): they take one reference image/latent in addition to the prompt,
> so `img_seq_len` includes the reference tokens. Use `flux_klein_edit.json` with them; a
> text-only graph does not supply the reference-latent input.

### LTX-Video 2 (19B)

Audio-video generation, **image-to-video only**. Only the **distilled** variant is compiled
(cfg 1.0, batch 1).

| Variant | CFG | Batch | Workflow | Static sizes (WxHxFrames) | Dynamic range |
|---------|-----|-------|----------|--------------------------|---------------|
| 19B **distilled** | 1.0 | 1 | i2v | 768x512x41, 1280x720x121, 1408x896x121 | 512px—768px—1408px, 41—121 frames |

Resolution and frame count are both dynamic. Audio tokens are computed from the frame count.

### Z-Image-Turbo (6B)

Image generation, cfg 1.0, batch 1, static sizes only.

| CFG | Batch | Static sizes |
|-----|-------|-------------|
| 1.0 (turbo) | 1 | 1024x1024, 1024x768, 768x1024 |

### LTX-Video 2.3 (22B)

Text-to-video with the full 22B LTXAV architecture (audio+video dual-stream attention).
LoRA-enabled **distilled** engines with dynamic resolution and frame count, batch 2, LoRA rank
1–256.

| Variant | CFG | Batch | Engine type | Static sizes (WxHxFrames) | Dynamic range |
|---------|-----|-------|-------------|--------------------------|---------------|
| 22B distilled FP8 + LoRA | 1.0, 4.0 | 2 (fixed) | t2v | 768x512x41, 1280x720x121, 1408x896x121 | 512px—768px—1408px, 9—121 frames |

### Qwen Image Edit (14B)

Image editing: a reference image and a text prompt produce the edited image. Text and image
tokens are joined into one sequence, with the text padded to 1536 tokens. Reference method:
`index_timestep_zero`. LoRA rank 1–256.

| Variant | CFG | Batch | Static sizes | Dynamic sizes | Ref images |
|---------|-----|-------|-------------|---------------|------------|
| BF16 + LoRA | 1.0 | 1 | 768, 1024, 1328, 1536 | All WxH from {512, 768, 1024, 1328, 1536} | 1 |

Inference needs `qlip_patch.py` in the engines directory.

### Wan 2.2 I2V (14B)

Image-to-video. Wan 2.2 has two experts: the high-noise transformer generates the initial video
and the low-noise transformer refines it.

**On H100** both experts are compiled, each with its own engines directory and
`QlipEnginesLoader`. Set `shared_memory="wan"` on both loaders: the two transformers never run at
the same time, so they can share one GPU memory pool.

| Variant | Transformer | CFG | Batch | Dynamic sizes | Frames |
|---------|------------|-----|-------|---------------|------------|
| FP8 + LoRA | High-noise | 1.0 | 1 | Up to 640x640 | 81 |
| FP8 + LoRA | Low-noise | 1.0 | 1 | Up to 640x640 | 81 |

**On RTX 5090** Wan 2.2 runs as a single expert: the low-noise transformer (NVFP4 + LoRA) plus our
low→high LoRA, which does the high-noise expert's job. See
[Wan 2.2 I2V — RTX 5090](benchmarks.md#wan-22-i2v-14b--rtx-5090).

### Krea 2 Turbo

Image generation, 8 steps, batch 1. FP8-static engines compiled on **H200**, with runtime LoRA
(rank up to 64). Needs ComfyUI `c2bcbecd`. Models: `krea2_turbo_bf16.safetensors`,
`qwen3vl_4b_bf16.safetensors`, `qwen_image_vae.safetensors` and optional LoRAs from
[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2).

| Variant | Static sizes | Dynamic range | TensorRT |
|---------|-------------|---------------|----------|
| FP8 + LoRA (`krea2-fp8_lora_trt_10_13`) | 1536x1536, 1536x864, 864x1536, 1024x1024 | 256px — 1024px — 1536px | 10.13 (`nvidia` install profile) |
| FP8 + LoRA (`krea2-fp8_lora`, legacy) | same | same | 10.15 |

## Precompiled engines

Set `hf_repo` on `Qlip Engines Loader` to one of the values below. The format is
`org/repo:path/to/engines`. The loader downloads the engine once and caches it.

Engines are GPU-specific: use the folder for your GPU (`H100`, `H200`, `GeForce-RTX-5090`, `B200`).

**H100 / H200**

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
| LTX-Video 2.3 22B Distilled FP8 + LoRA (t2v) | `TheStageAI/Elastic-LTX-2.3:models/H100/ltx-2.3-22b-distilled-fp8-t2v_lora` |
| Qwen Image Edit BF16 + LoRA | `TheStageAI/Elastic-Qwen-Image-Edit:models/H100/qwen-image-edit-bf16_lora` |
| Wan 2.2 I2V High-Noise FP8 + LoRA | `TheStageAI/Elastic-Wan2.2-I2V:models/H100/wan-i2v-high-noise-fp8_lora` |
| Wan 2.2 I2V Low-Noise FP8 + LoRA | `TheStageAI/Elastic-Wan2.2-I2V:models/H100/wan-i2v-low-noise-fp8_lora` |
| Krea 2 Turbo FP8 + LoRA — **H200**, TensorRT 10.13 (`nvidia` profile) | `TheStageAI/Elastic-Krea-2:models/H200/krea2-fp8_lora_trt_10_13` |
| Krea 2 Turbo FP8 + LoRA — **H200**, TensorRT 10.15 (legacy) | `TheStageAI/Elastic-Krea-2:models/H200/krea2-fp8_lora` |

**RTX 5090**

Engines marked *fp4attn* need the plugin setup in
[FP4-attention engines](installation.md#fp4-attention-engines-rtx-5090).

| Model | `hf_repo` value |
|-------|----------------|
| FLUX.2 Klein 9B Edit FP8-dynamic + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/GeForce-RTX-5090/klein-9b-fp8-dynamic-edit_lora` |
| FLUX.2 Klein 9B Edit FP8-static + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/GeForce-RTX-5090/klein-9b-fp8-static-edit_lora` |
| FLUX.2 Klein 9B Edit NVFP4 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/GeForce-RTX-5090/klein-9b-nvfp4-edit_lora` |
| FLUX.2 Klein 9B Edit NVFP4 + FP4-attention + LoRA (*fp4attn*) | `TheStageAI/Elastic-FLUX-2-Klein:models/GeForce-RTX-5090/klein-9b-nvfp4-fp4attn-edit_lora` |
| FLUX.2 Klein 9B Edit NVFP4 + FP4-attention, no LoRA (*fp4attn*) | `TheStageAI/Elastic-FLUX-2-Klein:models/GeForce-RTX-5090/klein-9b-nvfp4-fp4attn-edit-nolora` |
| Z-Image-Turbo FP8-dynamic + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/GeForce-RTX-5090/z-image-turbo-fp8-dynamic_lora` |
| Z-Image-Turbo NVFP4 + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/GeForce-RTX-5090/z-image-turbo-nvfp4_lora` |
| Z-Image-Turbo NVFP4 + FP4-attention + LoRA (*fp4attn*) | `TheStageAI/Elastic-Z-Image-Turbo:models/GeForce-RTX-5090/z-image-turbo-nvfp4-fp4attn_lora` |
| Z-Image-Turbo NVFP4 + FP4-attention, no LoRA (*fp4attn*) | `TheStageAI/Elastic-Z-Image-Turbo:models/GeForce-RTX-5090/z-image-turbo-nvfp4-fp4attn-nolora` |
| Wan 2.2 I2V NVFP4 + LoRA | `TheStageAI/Elastic-Wan2.2-I2V:models/GeForce-RTX-5090/wan-i2v-low-noise-nvfp4_lora` |
| Wan 2.2 I2V NVFP4 + FP4-attention + LoRA (*fp4attn*) | `TheStageAI/Elastic-Wan2.2-I2V:models/GeForce-RTX-5090/wan-i2v-low-noise-nvfp4-fp4attn_lora` |

**B200**

| Model | `hf_repo` value |
|-------|----------------|
| FLUX.2 Klein 9B Edit FP8-dynamic + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/B200/klein-9b-fp8-dynamic-edit_lora` |
| FLUX.2 Klein 9B Edit NVFP4 + LoRA | `TheStageAI/Elastic-FLUX-2-Klein:models/B200/klein-9b-nvfp4-edit_lora` |
| Z-Image-Turbo FP8-dynamic + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/B200/z-image-turbo-fp8-dynamic_lora` |
| Z-Image-Turbo NVFP4 + LoRA | `TheStageAI/Elastic-Z-Image-Turbo:models/B200/z-image-turbo-nvfp4_lora` |

Blackwell routing: models up to 14B run on the RTX 5090, 19B and larger on the B200. Wan 2.2 on
Blackwell is RTX 5090 only.

### Download engines manually

To use a local copy instead of `hf_repo`, download the engine folder with the `hf` CLI and point
`engines_path` at it. The old `huggingface-cli` is deprecated in `huggingface_hub` ≥ 1.x.

```bash
pip install click   # the hf CLI needs it; fixes "No module named 'click'"

# Example: FLUX.2 Klein 9B FP8 + LoRA
hf download TheStageAI/Elastic-FLUX-2-Klein \
    --local-dir ./engines/flux-klein \
    --include "models/H100/klein-9b-fp8_lora/*"
```

### Wan 2.2 on RTX 5090: get the low→high LoRA

The RTX 5090 Wan 2.2 workflow, `workflows/video_wan2_2_14b_i2v_5090-qlip.json`, loads our low→high
LoRA ([`wan2.2-i2v-low-to-high-lora.safetensors`](https://huggingface.co/TheStageAI/Elastic-Wan2.2-I2V/blob/main/models/GeForce-RTX-5090/wan2.2-i2v-low-to-high-lora.safetensors))
through `QlipLoraStack` from ComfyUI's `models/loras/`. Download it there:

```bash
python -c "
from huggingface_hub import hf_hub_download; import shutil
p = hf_hub_download('TheStageAI/Elastic-Wan2.2-I2V',
      'models/GeForce-RTX-5090/wan2.2-i2v-low-to-high-lora.safetensors')
shutil.copy(p, '/path/to/ComfyUI/models/loras/wan2.2-i2v-low-to-high-lora.safetensors')
print('LoRA installed')"
```

`QlipEnginesLoader` fetches the engine itself from its `hf_repo`.

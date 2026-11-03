# Benchmarks

Measured speedups of Qlip engines vs eager PyTorch and other accelerators.

All numbers are for one image or video, batch 1, **warm run** (second run, engines already
loaded). The benchmarked engines include runtime LoRA support, which costs about 5–15% compared
with no-LoRA engines (see [LoRA overhead](#lora-overhead)).

### H100 / H200

H100 numbers were measured on torch 2.8.0. The current `requirements.txt` pins torch 2.9.1; the
engines run the same on it.

**Image generation, H100** (1024x1024, 20 steps for FLUX, 8 steps for Z-Image, cfg 1)

| Model | Method | Time (s) | Speedup |
|-------|--------|----------|---------|
| **FLUX.2 Klein 9B** | Eager (PyTorch) | 3.743 | 1.0x |
| | torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 3.064 | 1.22x |
| | **Qlip BF16 + LoRA** | 2.944 | 1.27x |
| | **Qlip FP8 + LoRA** | 2.215 | **1.69x** |
| **Z-Image-Turbo** | Eager (PyTorch) | 1.142 | 1.0x |
| | torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 0.999 | 1.14x |
| | [SGLDiffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion) | 0.920 | 1.24x |
| | **Qlip BF16 + LoRA** | 0.957 | 1.19x |
| | **Qlip FP8 + LoRA** | 0.773 | 1.48x |
| | **Qlip FP8 + Cache** | 0.565 | **2.02x** |

**LTX-Video 2, H100** (1280x720, 121 frames, 8 sampler steps + 3 upsampling steps)

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 7.829 | 1.0x |
| torch.compile ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) | 5.330 | 1.47x |
| **Qlip BF16 + LoRA** | 6.020 | 1.30x |
| **Qlip FP8 + LoRA** | 5.051 | **1.55x** |

**LTX-Video 2.3 22B, H100** (1280x720, 121 frames, t2v, 8 sampler steps + 3 upsampling steps,
cfg 1.0). Transformer time only.

| Method | Transformer time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 15.188 | 1.0x |
| **Qlip FP8 + LoRA** | 8.537 | **1.78x** |

**Qwen Image Edit, H100** (1328x1328, 4 steps, cfg 1.0, 1 reference image). Transformer time only.

| Method | Transformer time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 3.135 | 1.0x |
| **Qlip BF16 + LoRA** | 2.389 | **1.31x** |

**Wan 2.2 I2V, H100** (640x640, 81 frames, 4 steps, cfg 1.0, both experts). Transformer time only.

| Method | Transformer time (s) | Speedup |
|--------|---------------------|---------|
| Eager (PyTorch) | 18.573 | 1.0x |
| **Qlip FP8 + LoRA** | 12.035 | **1.54x** |

**Krea 2 Turbo, H200** (1536x1536, 8 steps). Sampler time, median of images 2–8 from
`tools/krea_check.py` (8 prompts, LoRA switched after the 4th). The TensorRT 10.13 engine was
measured in a separate run on the same setup.

| Method | Sampler time (s) | Speedup |
|--------|------------------|---------|
| Eager (PyTorch) | 5.00 | 1.0x |
| **Qlip FP8 + LoRA** (TensorRT 10.13) | 3.41 | 1.47x |
| **Qlip FP8 + LoRA** (TensorRT 10.15) | 3.10 | 1.61x |
| **Qlip FP8 + LoRA** (TensorRT 10.15) + QlipCache + QlipProgressive | 1.85 | **2.70x** |

### Blackwell (RTX 5090 / B200)

Blackwell engines target RTX 5090 (`sm_120a`) and B200 (`sm_100`). They add **NVFP4** weights
(about 4× smaller than BF16) and, on the 5090, FP4 attention. FLUX.2 Klein is the **distilled**
variant: 4 sampling steps, cfg 1.0, full warm generation at 1024x2048.

#### FLUX.2 Klein 9B (edit) — RTX 5090

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 10.144 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 7.919 | 1.28x |
| **Qlip FP8-static + LoRA** | 7.520 | 1.35x |
| **Qlip NVFP4 + LoRA** | 6.058 | 1.67x |
| **Qlip NVFP4 + FP4-attention + LoRA** | 3.299 | 3.07x |
| **Qlip NVFP4 + FP4-attention (no LoRA)** | **2.810** | **3.61x** |

The fastest eager path, NVFP4 `.safetensors` + SageAttention3, takes 3.226 s. The Qlip NVFP4 +
FP4-attention engine is on par with LoRA (3.299 s) and about 13% faster without it (2.810 s).

#### Z-Image-Turbo 6B — RTX 5090

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 2.333 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 1.644 | 1.42x |
| **Qlip NVFP4 + LoRA** | 1.143 | 2.04x |
| **Qlip NVFP4 + FP4-attention + LoRA** | 1.017 | 2.29x |
| **Qlip NVFP4 + FP4-attention (no LoRA)** | 0.884 | 2.64x |

#### Wan 2.2 I2V (14B) — RTX 5090

The low→high LoRA lets the low-noise expert do the high-noise expert's job, so the pipeline runs
one transformer instead of two. The LoRA is applied at runtime on top of the NVFP4 +
FP4-attention engine. The 14B model fits on the 32 GB card with headroom at FP4; BF16 fits too,
at half the speed.

| 480×480, 81 frames, cfg 1.0 | Time / clip | Speedup |
|---|---|---|
| BF16 + LoRA (eager PyTorch) | 38.1 s | 1.0× |
| **Qlip NVFP4 + FP4-attention + LoRA** | **18.1 s** | **2.1×** |

Median per-clip wall time over a fixed 20-prompt batch, full pipeline (sampler + VAE decode +
save), warm, first clip excluded. Workflow:
[`workflows/video_wan2_2_14b_i2v_5090-qlip.json`](../workflows/video_wan2_2_14b_i2v_5090-qlip.json).
The plain NVFP4 engine is slower than the FP4-attention one but needs no plugin build.

#### FLUX.2 Klein 9B (edit) — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 2.05 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 1.59 | 1.29x |
| **Qlip NVFP4 + LoRA** | 1.27 | 1.61x |

#### Z-Image-Turbo 6B — B200

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager (PyTorch) | 0.611 | 1.0x |
| **Qlip FP8-dynamic + LoRA** | 0.493 | 1.24x |
| **Qlip NVFP4 + LoRA** | 0.435 | 1.40x |

FP4 attention is RTX 5090 only; B200 keeps FP8/BF16 attention.

### LoRA overhead

A `+ LoRA` engine always pays for its extra per-layer MatMuls, even when no LoRA is applied: LoRA
is a compile-time graph input, so the engine still gets a zero-filled `lora_packed` tensor. On the
Klein NVFP4 + FP4-attention engine this costs **3.299 s → 2.810 s = −0.489 s (~15%)**, so a no-LoRA
engine is always the fastest for a given precision.

Applying the same 0.49 s to the other Klein engines gives projected no-LoRA times of about 7.43 s
(FP8-dynamic), 7.03 s (FP8-static) and 5.57 s (NVFP4). Only the NVFP4 + FP4-attention no-LoRA engine
is actually built and measured.

### Reference: external NVFP4 checkpoints in eager PyTorch (RTX 5090)

These are other projects' pre-quantized checkpoints run in eager PyTorch, not Qlip engines. Each is
measured plain and with SageAttention3 (`sageattn3_blackwell`, the eager FP4 attention kernel for
consumer Blackwell, built from source). "Compiled" means `torch.compile` on the SageAttention3 op
only (its `allow_compile` path), not on the whole model.

**FLUX.2 Klein 9B NVFP4 (edit), 1024x2048, 4 steps** —
[`black-forest-labs/FLUX.2-klein-9b-nvfp4`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4)
(`flux-2-klein-9b-nvfp4.safetensors`, gated). Speedup vs the BF16 eager baseline (10.144 s).

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager NVFP4 `.safetensors` | 5.934 | 1.71x |
| Eager NVFP4 `.safetensors` + SageAttention3 | 3.226 | 3.14x |
| Eager NVFP4 `.safetensors` + SageAttention3 (compiled) | 3.262 | 3.11x |

On Klein, compiling the SageAttention3 op gives no gain.

**Z-Image-Turbo SVDQuant FP4 (r128)** —
[`nunchaku-ai/nunchaku-z-image-turbo`](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo)
(`svdq-fp4_r128-z-image-turbo.safetensors`, `svdq-fp4_r32-z-image-turbo.safetensors`). Speedup vs
the BF16 eager baseline (2.333 s).

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Eager SVDQuant FP4 `.safetensors` | 0.988 | 2.36x |
| Eager SVDQuant FP4 `.safetensors` + SageAttention3 | 0.951 | 2.45x |
| Eager SVDQuant FP4 `.safetensors` + SageAttention3 (compiled) | 0.842 | 2.77x |

These checkpoints load only through the nunchaku custom node and the `nunchaku` runtime package.
A plain `UNETLoader` fails with `KeyError: noise_refiner.0.attention.to_k.weight`, because the
checkpoint stores unfused `to_q/to_k/to_v` while ComfyUI's Z-Image expects fused `attention.qkv`.

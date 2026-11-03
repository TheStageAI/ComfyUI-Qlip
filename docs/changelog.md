# Changelog

**2026-08-04 — Acceleration nodes that need no precompiled engine**

- **QlipAutoSparse:** int8 dynamic block-sparse attention on any DiT, one knob. A diversity selector
  (similarity minus redundancy) picks the blocks per step; at 70% sparsity quality stays at the
  dense level (checked with DOVER).
- **QlipCache:** step caching with EasyCache / TaylorSeer / HiCache predictors and a separate cache
  per CFG branch, plus a DBCache-style `block` mode that also helps few-step distilled models.
- **QlipProgressive:** early denoising steps on a downscaled latent, with an `auto` ladder that has
  no knobs. Handles nested multimodal latents (MiniMax H3: video is downscaled, audio passes
  through).
- **QlipCompile** (+ **QlipQuantConfig**): per-block `torch.compile` with optional FP8/NVFP4 scaled
  GEMM. No ONNX export, cold start in seconds, no recompile on resolution change,
  `weights_policy=release` for a large VRAM cut.
- The four stack multiplicatively: **5.58× on Krea-2 Turbo, 5.09× on Z-Image-Turbo, 4.02× on
  MiniMax H3 video+audio** (RTX 5090), **2.27× on non-distilled Wan 2.2** (H100), all vs eager.

**2026-07-02 — Blackwell engines and the Wan 2.2 low→high LoRA**

- Engines recompiled for Blackwell with **NVFP4** weights (about 4× smaller than BF16) and **FP4
  attention** on the RTX 5090:
  - **FLUX.2 Klein 9B (edit):** RTX 5090 (FP8-dynamic/static, NVFP4, NVFP4 + FP4-attn) and B200
    (FP8-dynamic, NVFP4). Up to **3.6×** over BF16 eager on the 5090.
  - **Z-Image-Turbo 6B:** RTX 5090 and B200 (FP8-dynamic, NVFP4, NVFP4 + FP4-attn).
  - **Wan 2.2 I2V 14B:** RTX 5090 (NVFP4, NVFP4 + FP4-attn), encrypted, LoRA-enabled.
- **Wan 2.2 low→high LoRA:** lets the low-noise expert do the high-noise expert's job, so Wan 2.2
  runs as a single transformer. **18.1 s/clip vs 38.1 s BF16 (~2.1×)** at 480×480 / 81 frames.
  Published as [`wan2.2-i2v-low-to-high-lora.safetensors`](https://huggingface.co/TheStageAI/Elastic-Wan2.2-I2V/tree/main/models/GeForce-RTX-5090).
- **Blackwell install:** `install.py` automates the cu130 ordering (torch cu130 → ComfyUI
  requirements under a constraints file → device stack → `qlip.core[blackwell] --no-deps`).

**2026-04-15 — Wan 2.2 I2V, shared memory, runtime patches**

- **Wan 2.2 I2V (14B):** image-to-video, FP8 + LoRA, two experts (high-noise + low-noise), dynamic
  shapes up to 640x640.
- **Shared memory:** the `shared_memory` input of `QlipEnginesLoader`. Loaders with the same group
  name share one GPU memory pool (size = max, not sum), so the two Wan 2.2 transformers do not
  double VRAM.
- **Custom model patches:** a `qlip_patch.py` next to the engine files is loaded at runtime, with
  `patch_signatures(dm)` and `patch_caller(dm)` hooks.
- **LTX-Video 2.3 (22B):** text-to-video, FP8 distilled + LoRA, dynamic shapes 512px–1408px, 9–121
  frames.
- **Qwen Image Edit:** image editing, BF16 + LoRA, dynamic shapes 512px–1536px, reference image
  support.

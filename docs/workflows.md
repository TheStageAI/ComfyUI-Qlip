# Workflows and model downloads

Ready-to-use workflows, scripts that download the model weights, and the basic graph patterns.

Ready-to-use workflows are in [`workflows/`](../workflows/):

| File | Description | ComfyUI commit |
|------|-------------|----------------|
| `flux-klein.json` | FLUX.2 Klein image generation (H100 engines) | `048dd2f3` |
| `video_ltx2_i2v_distilled.json` | LTX-2 19B image-to-video | `048dd2f3` |
| `z-image-turbo.json` | Z-Image-Turbo image generation (H100 engines) | `048dd2f3` |
| `z-image-turbo-api.json` | Z-Image-Turbo with engines and timers, API format (for scripts) | `b615af1c` |
| `video_ltx_2_3_t2v-qlip.json` | LTX-Video 2.3 22B text-to-video | `b615af1c` |
| `ltx2.3-full-qlip.json` | LTX-Video 2.3 with engines, LoRA and `QlipAutoSparse`; uses the `ClownSampler_Beta` custom sampler | `b615af1c` |
| `qwen-image-edit.json` | Qwen Image Edit | `b615af1c` |
| `video_wan2_2_14B_i2v-qlip.json` | Wan 2.2 I2V 14B, two experts (H100) | `b615af1c` |
| `video_wan2_2_14b_i2v_5090-qlip.json` | Wan 2.2 I2V 14B, one expert + low→high LoRA (RTX 5090) | `b615af1c` |
| `flux_klein_edit.json` | FLUX.2 Klein image edit (RTX 5090 / B200 engines) | `b615af1c` |
| `z-image-turbo-5090.json` | Z-Image-Turbo (RTX 5090 / B200 engines) | `b615af1c` |

Model download scripts are in [`scripts/`](../scripts/). They skip files that are already
downloaded, so you can re-run them. Set `HF_HUB_CACHE` to change the Hugging Face cache location.

| Script | Downloads |
|--------|-------------|
| `download_flux_klein_models.sh` | FLUX.2 Klein: diffusion model, text encoder, VAE. Needs a Hugging Face token |
| `download_ltx_2_models.sh` | LTX-2 19B: checkpoints, text encoder, LoRAs, upscaler |
| `download_z_image_turbo_models.sh` | Z-Image-Turbo: diffusion model, text encoder, VAE, LoRA |
| `download_ltx_2_3_models.sh` | LTX-Video 2.3 22B: checkpoint, Gemma text encoder, LoRAs, upscaler |
| `download_qwen_image_edit.sh` | Qwen Image Edit: diffusion model, text encoder, LoRA |
| `download_wan_i2v_models.sh` | Wan 2.2 I2V 14B: high- and low-noise experts, text encoder, CLIP vision, VAE, lightx2v 4-step LoRAs |

```bash
export COMFYUI_PATH=/path/to/ComfyUI
bash custom_nodes/ComfyUI-Qlip/scripts/download_z_image_turbo_models.sh   # any script above
```

FLUX.2 Klein is gated: accept its license on Hugging Face and log in first (`hf auth login`, or set
`HF_TOKEN`; `huggingface-cli login` is deprecated).

### Graph patterns

Engines without LoRA:

```
UNETLoader / CheckpointLoaderSimple (model.safetensors)
  -> Qlip Engines Loader (engines_path=...)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

Engines with LoRA:

```
Qlip LoRA Stack (lora.safetensors, strength=1.0)  ─────────────────────┐
                                                                         |
UNETLoader / CheckpointLoaderSimple (model.safetensors)                  |
  -> Qlip Engines Loader (engines_path=..., lora_stack=^)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

Several LoRAs, chained through `prev_stack`:

```
Qlip LoRA Stack (style_lora.safetensors, 0.8)
  -> Qlip LoRA Stack (detail_lora.safetensors, 0.5, prev_stack=^)
    -> Qlip Engines Loader (engines_path=..., lora_stack=^)
```

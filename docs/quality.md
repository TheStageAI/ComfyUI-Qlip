# Quality

How close Qlip engines stay to the original model. Each engine is compared with the same model in
eager PyTorch (**Original**), in the same ComfyUI graph, with the same prompts and seeds. Speed
numbers are on the [Benchmarks](benchmarks.md) page.

- [How we measure](#how-we-measure)
- [Z-Image-Turbo 6B](#z-image-turbo-6b)
- [FLUX.2 Klein 9B](#flux2-klein-9b)
- [Krea 2 Turbo](#krea-2-turbo)

## How we measure

**Prompts.** All 1,632 prompts of [PartiPrompts](https://huggingface.co/datasets/nateraw/parti-prompts)
(12 categories) and all 200 prompts of [DrawBench](https://huggingface.co/datasets/shunk031/DrawBench)
(11 categories). One image per prompt; the seed is 42 + the prompt's index, the same for Original
and the engine.

**Graph.** The ComfyUI graph of the model's workflow, with and without **Qlip Engines Loader**; the
text encoder, VAE and sampler are the same in both runs. Each run uses the ComfyUI commit the engines
need (see [Installation](installation.md)).

**Metrics.** Each metric is averaged within a category, then over the categories of the dataset.

| Metric | What it measures | Range |
|---|---|---|
| **ARNIQA** | No-reference image quality (`koniq10k` weights) | 0–100, higher is better |
| **CLIP IQA** | No-reference image quality from CLIP (`"quality"` prompt pair) | 0–100, higher is better |
| **VQA Faithfulness** | Share of images for which BLIP VQA (`Salesforce/blip-vqa-base`) answers "yes" to "Does the image depict: {prompt}?" | 0–100, higher is better |
| **PSNR** | Pixel similarity of the engine's image to Original's image with the same seed, dB | higher is closer |
| **SSIM** | Structural similarity of the engine's image to Original's image with the same seed | 0–1, higher is closer |

PSNR and SSIM measure how close the engine's image is to Original's, not how good either image is.
These models generate in 4–8 steps, and a small numeric change can move objects in the frame while the
image stays just as good, so PSNR and SSIM can be low even when the other metrics match.

**Is a VQA difference real?** VQA gives one yes/no per image. We count the prompts where only Original
gets a "yes" and those where only the engine does, and test the difference with McNemar's test. A
difference is called real when p < 0.05.

## Z-Image-Turbo 6B

1024x1024, 8 steps, `res_multistep` / simple, shift 3, cfg 1.0.

| Metric | H100: Qlip FP8 + LoRA | H100: Original | RTX 5090: Qlip NVFP4 + LoRA | RTX 5090: Original |
|---|---|---|---|---|
| ARNIQA (PartiPrompts) | 66.1 | 65.8 | 66.3 | 65.8 |
| ARNIQA (DrawBench) | 67.9 | 67.5 | 67.7 | 67.6 |
| CLIP IQA (PartiPrompts) | 84.3 | 83.6 | 83.0 | 83.5 |
| CLIP IQA (DrawBench) | 86.1 | 86.2 | 85.0 | 86.2 |
| VQA Faithfulness (PartiPrompts) | 84.3 | 85.2 | 86.0 | 85.2 |
| VQA Faithfulness (DrawBench) | 65.7 | 65.9 | 59.0 | 65.2 |
| PSNR (PartiPrompts) | 18.27 | N/A | 16.24 | N/A |
| PSNR (DrawBench) | 17.10 | N/A | 15.28 | N/A |
| SSIM (PartiPrompts) | 0.768 | N/A | 0.711 | N/A |
| SSIM (DrawBench) | 0.741 | N/A | 0.683 | N/A |

Engines: `models/H100/z-image-turbo-fp8_lora` and `models/GeForce-RTX-5090/z-image-turbo-nvfp4_lora`,
no LoRA connected.

- **FP8 on H100 keeps the quality of Original.** ARNIQA and CLIP IQA are within 0.7 points, and the VQA
  differences are not real (PartiPrompts: 36 prompts only Original, 29 only FP8, p = 0.46; DrawBench:
  7 and 6, p = 1.0).
- **NVFP4 on RTX 5090 keeps image quality and PartiPrompts faithfulness**, and loses some faithfulness
  on DrawBench. ARNIQA and CLIP IQA are within 1.2 points, and on PartiPrompts the VQA difference is not
  real (32 and 41, p = 0.35). On DrawBench VQA drops from 65.2 to 59.0, and this difference is real:
  16 prompts are "yes" only for Original and 4 only for NVFP4 (p = 0.012). The drop is spread over 9
  of the 11 categories; Misspellings and Positional go up.

## FLUX.2 Klein 9B

Distilled model, text-to-image. 1024x1024, 4 steps, euler, `Flux2Scheduler`, cfg 1.0.

| Metric | H100: Qlip FP8 + LoRA | H100: Original |
|---|---|---|
| ARNIQA (PartiPrompts) | 67.2 | 67.2 |
| ARNIQA (DrawBench) | 68.1 | 68.0 |
| CLIP IQA (PartiPrompts) | 85.7 | 85.5 |
| CLIP IQA (DrawBench) | 86.6 | 86.4 |
| VQA Faithfulness (PartiPrompts) | 88.9 | 89.2 |
| VQA Faithfulness (DrawBench) | 71.3 | 73.3 |
| PSNR (PartiPrompts) | 24.04 | N/A |
| PSNR (DrawBench) | 23.25 | N/A |
| SSIM (PartiPrompts) | 0.884 | N/A |
| SSIM (DrawBench) | 0.874 | N/A |

Engine: `models/H100/klein-9b-fp8_lora`, no LoRA connected.

- **FP8 keeps the quality of Original.** ARNIQA and CLIP IQA are within 0.2 points, and the VQA
  differences are not real (PartiPrompts: 14 prompts only Original, 11 only FP8, p = 0.69; DrawBench:
  6 and 2, p = 0.29).
- **The images are the closest to Original of the three models:** PSNR 24 dB, SSIM 0.88.

## Krea 2 Turbo

1536x1536, 8 steps, euler / simple, cfg 1.0. Measured on H100 with the H200 engines
`models/H200/krea2-fp8_lora_trt_10_13`.

| Metric | H100: Qlip FP8 + LoRA | H100: Original |
|---|---|---|
| ARNIQA (PartiPrompts) | 62.5 | 62.4 |
| ARNIQA (DrawBench) | 62.9 | 63.0 |
| CLIP IQA (PartiPrompts) | 76.0 | 76.2 |
| CLIP IQA (DrawBench) | 76.4 | 77.0 |
| VQA Faithfulness (PartiPrompts) | 87.6 | 86.8 |
| VQA Faithfulness (DrawBench) | 68.9 | 68.0 |
| PSNR (PartiPrompts) | 17.90 | N/A |
| PSNR (DrawBench) | 17.53 | N/A |
| SSIM (PartiPrompts) | 0.746 | N/A |
| SSIM (DrawBench) | 0.739 | N/A |

Krea 2 engines need a LoRA of rank 32–64 (see [README](../README.md)), so the engine run has the
official `krea2_retroanime.safetensors` (rank 32) connected with `strength` 0. Its contribution is
zero; Original runs without a LoRA.

- **FP8 keeps the quality of Original.** ARNIQA and CLIP IQA are within 0.6 points, and the VQA
  differences are not real (PartiPrompts: 16 prompts only Original, 26 only FP8, p = 0.16; DrawBench:
  4 and 6, p = 0.75).

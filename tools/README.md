# ComfyUI-Qlip tools

Standalone scripts that help you tune and check ComfyUI-Qlip. They are not ComfyUI nodes. Run
them from your ComfyUI root, so `import comfy` and the model paths resolve.

| Script | Use it to |
|---|---|
| [`fit_spectrum.py`](#fit_spectrumpy-fit-the-spectrum-for-qlipprogressive) | Fit `speed_A`/`speed_beta` for the `spectral` mode of `QlipProgressive` |
| [`krea_check.py`](#krea_checkpy-end-to-end-check-on-krea-2-turbo) | Run ComfyUI-Qlip configurations end to end on Krea 2 Turbo and record timings |
| [`qlip_report.py`](#qlip_reportpy-and-qlip_collagepy-qlip-agent-report) | Build the QLIP Agent report from qlip-arena results |
| [`qlip_collage.py`](#qlip_reportpy-and-qlip_collagepy-qlip-agent-report) | Build a side-by-side collage of the key frontier configs |

## `fit_spectrum.py`: fit the spectrum for QlipProgressive

`QlipProgressive` with `backbone_mode = spectral` (SPEED, arXiv 2605.18736) decides when to grow
the grid from the model's data power spectrum `P(omega) = A * |omega|^(-beta)`. The built-in
`(A, beta)` is Krea-2's fit; **other models need their own fit**. The `Qlip Spectrum Fit` node
does the same inside a workflow.

```bash
# from the ComfyUI root
venv/bin/python custom_nodes/ComfyUI-Qlip/tools/fit_spectrum.py \
    --model mymodel \
    --unet  models/diffusion_models/mymodel.safetensors \
    --clip  models/text_encoders/mymodel_te.safetensors \
    --clip-type FLUX \
    --n 3 --size 1024 --steps 8
```

The script runs `--n` real generations, radially averages the power spectrum of each final latent,
normalizes it by SNR and fits `A` and `beta` on a log-log scale. It prints both values and writes
`<model>_spectrum.txt`.

Then set on the `QlipProgressive` node:

- `backbone_mode = spectral`
- `speed_A = <A>`
- `speed_beta = <beta>`

Leaving `speed_A`/`speed_beta` at 0 uses the built-in Krea-2 fit:
`A=0.024118  beta=2.549520  R^2=0.9968`.

Run the fit once per model, offline. The earlier in-engine autofit took the clean latent at
`sigma~0.31` instead of 0 and underestimated `A`. `P(0.5)` then fell below `delta`, the switch time
went to 0 and the ladder stayed at low resolution. A few clean offline generations give a stable
prior.

## `krea_check.py`: end-to-end check on Krea 2 Turbo

Runs ComfyUI-Qlip configurations on Krea 2 Turbo through the ComfyUI HTTP API. For each
configuration it builds an API workflow, restarts ComfyUI (the license binding and compiled state
live in the process), runs N prompts with a LoRA swap halfway, and records the per-image sampling
time (from `QlipTimer`), the images, the cache report and any error.

```bash
python tools/krea_check.py --comfy /workspace/ComfyUI --python /workspace/venv/bin/python \
    --configs eager engine engine_cache_prog compile_fp8 --n 8 --out /workspace/results/new
```

For A/B runs, `--pythonpath` prepends a qlip source tree (a local build) and `--nodes` points
`custom_nodes/ComfyUI-Qlip` at another checkout.

It needs these files in `ComfyUI/models`, all from
[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2):

- `diffusion_models/krea2_turbo_bf16.safetensors`
- `text_encoders/qwen3vl_4b_bf16.safetensors`
- `vae/qwen_image_vae.safetensors`
- `loras/krea2_retroanime.safetensors`, `loras/krea2_softwatercolor.safetensors`

The engine configurations use `TheStageAI/Elastic-Krea-2:models/H200/krea2-fp8_lora_trt_10_13`
(TensorRT 10.13, the `nvidia` profile), which `QlipEnginesLoader` downloads on first use.

## `qlip_report.py` and `qlip_collage.py`: QLIP Agent report

The [QLIP Agent](../AGENTS.md) calls these at the end of its search. You can also run them
yourself on a qlip-arena `results/` folder. They only read the exported JSONs and never regenerate
or re-judge.

```bash
# report.json, REPORT.md, frontier.png, params.png
python tools/qlip_report.py --results ~/krea2_engine/qlip-arena/results \
    --baseline krea2-eager --tol 0.35 --out ~/krea2_engine/report --gpu H100 --model krea2

# collage.png: eager | recommended | quality leader | max-speed point
python tools/qlip_collage.py --store $QLIP_ARENA_ROOT --report <out>/report.json \
    --out <out>/collage.png --rows 3 --seed 1000
```

With `--fpcharts <fpcharts_data.json>` (the normal case, see `AGENTS.md` §4), the recommended
config is the fastest faithful config that passes the arena's defect gate and is statistically
tied with the best-quality one. Without it, the recommendation is the fastest config with
mean LPIPS ≤ `--tol`. Matplotlib is optional: without it the script still writes the JSON and
`REPORT.md` and skips the PNGs.

What each output file means: [`READING_THE_AGENT_REPORT.md`](READING_THE_AGENT_REPORT.md). The
arena's own pages and metrics are explained in the qlip-arena repo, in `docs/READING_THE_REPORT.md`
and `docs/METRICS.md`.

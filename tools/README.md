# ComfyUI-Qlip tools

Standalone helpers, run from your ComfyUI root (so `import comfy` and the model
paths resolve). Not ComfyUI nodes.

## `fit_spectrum.py` — spectrum fit for QlipProgressive `spectral` backbone

QlipProgressive's `spectral` backbone (SPEED, arXiv 2605.18736) decides when to
grow the progressive grid from the model's data power spectrum
`P(omega) = A * |omega|^(-beta)`. Krea-2's `(A, beta)` is built in; **any other
model needs its own fit** — this produces it.

```bash
# from the ComfyUI root
venv/bin/python custom_nodes/ComfyUI-Qlip/tools/fit_spectrum.py \
    --model mymodel \
    --unet  models/diffusion_models/mymodel.safetensors \
    --clip  models/text_encoders/mymodel_te.safetensors \
    --clip-type FLUX \
    --n 3 --size 1024 --steps 8
```

It runs `--n` real generations, radial-averages each final latent's power
spectrum, SNR-normalizes, and log-log fits `A`/`beta`. It prints the two values
and writes `<model>_spectrum.txt`.

Then on the **QlipProgressive** node set:

- `backbone_mode = spectral`
- `speed_A = <A>`
- `speed_beta = <beta>`

(Leaving `speed_A`/`speed_beta` at 0 uses the built-in Krea-2 fit.)

Krea-2 reference: `A=0.024118  beta=2.549520  R^2=0.9968`.

**Why offline, not the old in-engine autofit:** online fitting harvested the
clean latent at `sigma~0.31` rather than 0, so it under-estimated `A`; `P(0.5)`
then fell below `delta`, the switch time went to 0, and the ladder stuck at low
resolution. A few clean offline generations are a stable prior — run this once
per model.
```

## Reading the agent's `report/`

See `READING_THE_AGENT_REPORT.md` in this folder; the arena's own pages and every metric are explained in the qlip-arena repo, `docs/READING_THE_REPORT.md` and `docs/METRICS.md`.

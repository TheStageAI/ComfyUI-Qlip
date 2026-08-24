"""Offline power-spectrum fit for QlipProgressive's `spectral` backbone (SPEED,
arXiv 2605.18736).

The `spectral` backbone grows the progressive grid exactly when a rung's Nyquist
band leaves the noise floor, computed from the model's data power spectrum
P(omega) = A * |omega|^(-beta). Krea-2's (A, beta) is built in; ANY OTHER model
needs its own fit, produced by this script, then pasted into the QlipProgressive
node's `speed_A` / `speed_beta` inputs.

What it does: runs N real generations, captures each FINAL clean latent x0 (the
"data" latent), radial-averages its power spectrum, SNR-normalizes by the
white-noise floor (H*W per Fourier mode), and log-log fits P = A*|omega|^(-beta)
on a mid frequency band. Prints (A, beta, R^2) and the exact node values to set.

Offline, not online: online in-engine fitting was removed because it harvested x0
at sigma~0.31 (not 0) and so under-estimated A → P(0.5) fell below delta → the
switch time t*=0 → the ladder stuck at low resolution. A handful of clean offline
generations is a stable prior; run it once per model.

Run it from your ComfyUI root (so `import comfy` and the model paths resolve):

  venv/bin/python custom_nodes/ComfyUI-Qlip/tools/fit_spectrum.py \
      --model krea2 \
      --unet models/diffusion_models/krea2_turbo_bf16.safetensors \
      --clip models/text_encoders/qwen3vl_4b_bf16.safetensors \
      --clip-type KREA2 \
      --n 3 --size 1024 --steps 8

Krea-2 reference result (n=3, 1024px): A=0.024118  beta=2.549520  R^2=0.9968.
"""
import argparse
import math
import sys

import torch

sys.path.insert(0, ".")


def radial_power(x):
    """x: [C,H,W] real latent. Returns (omega, P) radial-averaged power spectrum
    (raw, un-normalized; the SNR normalization happens once, after averaging)."""
    C, H, W = x.shape
    Xf = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    power = (Xf.real ** 2 + Xf.imag ** 2).mean(0)          # [H,W] avg over channels
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx,
                            indexing="ij")
    r_int = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).round().long()
    rmax = int(min(cy, cx))
    P = torch.zeros(rmax + 1)
    cnt = torch.zeros(rmax + 1)
    flat_r = r_int.flatten().clamp(max=rmax)
    P.index_add_(0, flat_r, power.flatten())
    cnt.index_add_(0, flat_r, torch.ones_like(power.flatten()))
    P = P / cnt.clamp(min=1)
    return torch.arange(rmax + 1).float(), P


DEFAULT_PROMPTS = [
    "A cinematic high-fashion editorial portrait of a woman in a black jacket, "
    "brutalist gallery",
    "A sunlit mountain landscape with a lake, golden hour, wide shot",
    "A vintage sports car on a coastal road, dusk, motion blur",
    "A cozy cafe interior, morning light, plants, wooden tables",
    "Abstract fluid swirls of teal and orange, macro detail",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model",
                    help="label for the model (used only in the printout / --out)")
    ap.add_argument("--unet", required=True,
                    help="diffusion model checkpoint, relative to the ComfyUI root")
    ap.add_argument("--clip", required=True,
                    help="text-encoder checkpoint, relative to the ComfyUI root")
    ap.add_argument("--clip-type", default="KREA2",
                    help="comfy CLIPType name (e.g. KREA2, FLUX, SD3)")
    ap.add_argument("--n", type=int, default=3, help="number of generations to pool")
    ap.add_argument("--size", type=int, default=1024, help="image side in pixels")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--prompt", action="append", default=None,
                    help="override the built-in prompts (repeatable)")
    ap.add_argument("--out", default=None,
                    help="write A/beta to this file (default: <model>_spectrum.txt)")
    args = ap.parse_args()

    import comfy.sample
    import comfy.sd

    clip_type = getattr(comfy.sd.CLIPType, args.clip_type)
    model = comfy.sd.load_diffusion_model(args.unet)
    clip = comfy.sd.load_clip(ckpt_paths=[args.clip], clip_type=clip_type)

    prompts = (args.prompt or DEFAULT_PROMPTS)[:args.n]
    if len(prompts) < args.n:
        raise SystemExit(f"need {args.n} prompts, only {len(prompts)} available "
                         f"— pass more with --prompt")

    hh = ww = args.size // 8
    lat = torch.zeros(1, 16, 1, hh, ww)
    accum_P = None
    omega = None
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    for i, p in enumerate(prompts):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(p))
        noise = torch.randn(1, 16, 1, hh, ww,
                            generator=torch.manual_seed(1000 + i))
        x0 = comfy.sample.sample(model, noise, args.steps, 1.0, args.sampler,
                                 args.scheduler, cond, neg, lat, denoise=1.0,
                                 disable_pbar=True, seed=1000 + i)
        xf = x0.detach().float().squeeze(0).squeeze(1)      # [C,H,W]
        om, P = radial_power(xf)
        accum_P = P if accum_P is None else accum_P + P
        omega = om
        print(f"  prompt {i}: captured x0 {tuple(xf.shape)}", flush=True)

    P = accum_P / len(prompts)
    # PHYSICAL normalization -> SNR spectrum. SPEED's P is a signal-to-noise power
    # ratio per frequency (P=1 marks the noise/signal crossover, and the switch
    # time expects P=O(1)). Flow-matching noise is standard Normal per latent
    # pixel; by Parseval its power is FLAT = H*W per Fourier mode (un-normalized
    # FFT). Divide the signal power by that noise floor to get SNR(omega) —
    # resolution-independent, no anchor tuning.
    P = P / float(hh * ww)
    # fit P = A * omega_n^(-beta) in NORMALIZED frequency omega_n = omega/omega_max,
    # on a mid band (skip DC and the Nyquist tail).
    omega_max = float(omega[-1])
    omega_n = omega / omega_max
    lo = max(2, len(omega) // 16)
    hi = int(len(omega) * 0.75)
    logw = torch.log(omega_n[lo:hi])
    logP = torch.log(P[lo:hi].clamp(min=1e-12))
    M = torch.stack([logw, torch.ones_like(logw)], 1)
    sol = torch.linalg.lstsq(M, logP.unsqueeze(1)).solution.squeeze(1)
    slope, intercept = sol[0].item(), sol[1].item()
    beta = -slope
    A = math.exp(intercept)
    pred = slope * logw + intercept
    ss_res = ((logP - pred) ** 2).sum()
    ss_tot = ((logP - logP.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    print(f"\n=== {args.model} spectrum fit "
          f"(n={len(prompts)}, {args.size}px, band [{lo},{hi}]) ===")
    print(f"A = {A:.6f}   beta = {beta:.6f}   R^2 = {r2:.4f}")
    if r2 < 0.9:
        print("WARNING: R^2 < 0.9 — fit is weak; add more prompts (--n) or check "
              "the model/clip/size are correct.")
    print("\n-> set these on the QlipProgressive node:")
    print(f"     backbone_mode = spectral")
    print(f"     speed_A       = {A:.6f}")
    print(f"     speed_beta    = {beta:.6f}")

    out = args.out or f"{args.model}_spectrum.txt"
    with open(out, "w") as f:
        f.write(f"A={A:.6f}\nbeta={beta:.6f}\nR2={r2:.4f}\n"
                f"n={len(prompts)}\nsize={args.size}\nmodel={args.model}\n")
    print(f"\nsaved -> {out}")
    print("FIT_DONE")


if __name__ == "__main__":
    main()

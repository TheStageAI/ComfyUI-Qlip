"""QlipSpectrumFit — in-graph power-spectrum fit for QlipProgressive's
``spectral`` backbone (SPEED, arXiv 2605.18736).

Replaces the manual ``tools/fit_spectrum.py`` CLI step: instead of running a
script and pasting ``speed_A``/``speed_beta`` into the QlipProgressive node,
wire MODEL + CLIP + LATENT through this node once. It runs a few real
generations, captures each final clean latent x0, radial-averages the power
spectrum, SNR-normalizes and log-log fits ``P = A * omega^(-beta)`` — then
returns the MODEL with the fit attached as ``model_options["qlip_spectrum"]``,
which QlipProgressive picks up automatically (its priority-2 source; manual
``speed_A``/``speed_beta`` inputs still override).

Why a LATENT input rather than a ``size`` widget: the spectrum depends on the
generation resolution (beta measured at 1024px differs from 1536px), so fitting
on the *same* latent you are about to sample guarantees the schedule matches
the run. It also gives the correct latent shape for any model family.

The fit costs n_samples full generations, ONCE: results are cached in-process
by (model, latent shape, sampling params) so re-queues of an unchanged graph
skip the work — and ComfyUI's own node cache skips re-execution entirely when
the inputs did not change.
"""
import hashlib
import importlib.util
import os

import torch

from .engine_loader import _validate_diffusion_model_input

# shared math lives in tools/fit_spectrum.py (also usable standalone as a CLI);
# tools/ is not a package, so load it by path.
_TOOLS_FIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tools", "fit_spectrum.py")
_spec = importlib.util.spec_from_file_location("qlip_fit_spectrum", _TOOLS_FIT)
_fitmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fitmod)
radial_power = _fitmod.radial_power
fit_power_law = _fitmod.fit_power_law
DEFAULT_PROMPTS = _fitmod.DEFAULT_PROMPTS

# (cache_key) -> (A, beta, r2)  — survives re-queues within one ComfyUI process
_FIT_CACHE = {}


class QlipSpectrumFit:
    """Fit the model's data power spectrum and feed it to QlipProgressive."""

    CATEGORY = "qlip"
    FUNCTION = "fit"
    RETURN_TYPES = ("MODEL", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("model", "speed_A", "speed_beta", "report")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:            # noqa: BLE001 — import-time fallback
            samplers, schedulers = ["euler"], ["simple"]
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to fit (and pass through)."}),
                "clip": ("CLIP", {"tooltip": "Text encoder for the probe prompts."}),
                "latent": ("LATENT", {
                    "tooltip": "Latent at YOUR generation resolution — the fit "
                               "is resolution-dependent, so wire the same "
                               "EmptyLatentImage the sampler uses."}),
            },
            "optional": {
                "n_samples": ("INT", {
                    "default": 3, "min": 1, "max": 5,
                    "tooltip": "Probe generations to pool. 3 is enough "
                               "(Krea-2 reference fit used 3, R^2 0.997)."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "seed": ("INT", {"default": 1000, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "prompts": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "One probe prompt per line. Empty = built-in "
                               "diverse set."}),
            },
        }

    def fit(self, model, clip, latent, n_samples=3, steps=8,
            sampler_name="euler", scheduler="simple", seed=1000, prompts=""):
        _validate_diffusion_model_input(model, "QlipSpectrumFit")
        import comfy.sample

        # normalize the latent exactly like KSampler does: fix the channel count
        # of an empty latent and add the temporal dim for 5D-latent models
        # (Krea-2 / Qwen-Image are [B,C,1,H,W]; a raw 4D EmptyLatentImage would
        # skip the 2x2 patchify and crash the first projection).
        lat = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        prompt_list = [p.strip() for p in prompts.splitlines() if p.strip()] \
            or list(DEFAULT_PROMPTS)
        prompt_list = (prompt_list * n_samples)[:n_samples] \
            if len(prompt_list) < n_samples else prompt_list[:n_samples]

        # cache key: model identity + everything that changes the fit
        try:
            dm = model.model.diffusion_model
            model_tag = f"{type(dm).__name__}:{sum(1 for _ in dm.parameters())}"
        except Exception:            # noqa: BLE001
            model_tag = "unknown"
        key = hashlib.sha1(repr((
            model_tag, tuple(lat.shape), n_samples, steps, sampler_name,
            scheduler, seed, tuple(prompt_list))).encode()).hexdigest()

        if key in _FIT_CACHE:
            A, beta, r2 = _FIT_CACHE[key]
            print(f"[QlipSpectrumFit] cached fit reused: A={A:.6f} "
                  f"beta={beta:.6f} R^2={r2:.4f}")
        else:
            print(f"[QlipSpectrumFit] fitting spectrum: {n_samples} probe "
                  f"generation(s) at {tuple(lat.shape)} ...", flush=True)
            neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            zero_lat = torch.zeros_like(lat[:1])
            accum_P, omega = None, None
            for i, p in enumerate(prompt_list):
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(p))
                noise = torch.randn(zero_lat.shape,
                                    generator=torch.manual_seed(seed + i))
                x0 = comfy.sample.sample(
                    model, noise, steps, 1.0, sampler_name, scheduler,
                    cond, neg, zero_lat, denoise=1.0, disable_pbar=True,
                    seed=seed + i)
                # -> [C', H, W]: fold any extra dims (batch, video frames) into
                # channels; the radial spectrum averages over dim 0 anyway.
                xf = x0.detach().float()
                H, W = xf.shape[-2], xf.shape[-1]
                xf = xf.reshape(-1, H, W)
                om, P = radial_power(xf)
                accum_P = P if accum_P is None else accum_P + P
                omega = om
                print(f"[QlipSpectrumFit]   probe {i + 1}/{n_samples} done",
                      flush=True)
            P = accum_P / len(prompt_list)
            # SNR-normalize by the white-noise floor (H*W per Fourier mode) —
            # same physics as tools/fit_spectrum.py.
            P = P / float(lat.shape[-2] * lat.shape[-1])
            A, beta, r2, lo, hi = fit_power_law(omega, P)
            _FIT_CACHE[key] = (A, beta, r2)
            print(f"[QlipSpectrumFit] fit: A={A:.6f} beta={beta:.6f} "
                  f"R^2={r2:.4f} (band [{lo},{hi}])")
            if r2 < 0.9:
                print("[QlipSpectrumFit] WARNING: R^2 < 0.9 — weak fit; raise "
                      "n_samples or check model/clip/latent.")

        patched = model.clone()
        patched.model_options = {**patched.model_options,
                                 "qlip_spectrum": {"A": float(A),
                                                   "beta": float(beta)}}
        report = (f"A={A:.6f} beta={beta:.6f} R^2={r2:.4f} "
                  f"(n={n_samples}, latent {tuple(lat.shape)}). "
                  f"QlipProgressive picks this up automatically via "
                  f"model_options['qlip_spectrum'] — set backbone_mode=spectral "
                  f"and leave speed_A/speed_beta at 0.")
        return (patched, float(A), float(beta), report)


NODE_CLASS_MAPPINGS = {"QlipSpectrumFit": QlipSpectrumFit}
NODE_DISPLAY_NAME_MAPPINGS = {"QlipSpectrumFit": "Qlip Spectrum Fit"}

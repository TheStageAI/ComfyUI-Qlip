"""Qlip Progressive — low-resolution early steps via a model-call hook.

Thin ComfyUI router: ALL logic (interpolation rules, layout adapters for
plain/nested/flat-packed latents, the auto ladder with sigma backbone +
latent-size floor + x̂0-stability accelerator) lives in the licensed qlip
package — ``qlip.inference.progressive_core.ProgressiveEngine``. This node
only wires the engine into ComfyUI's official
``set_model_unet_function_wrapper`` (chaining with any wrapper set earlier,
e.g. QlipCache) and parses the timestep.

Sampling through the engine's low-resolution path runs under the standard
qlip paid-session gate, inherited from the loaded Qlip engines when
present.
"""

from .engine_loader import _validate_diffusion_model_input


class QlipProgressive:
    """Early denoising steps at low latent resolution — as a model hook."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Any DiT; keep your own "
                          "sampler — this is a model hook, not a sampler."}),
                "enable": ("BOOLEAN", {"default": True}),
                "low_scale": ("FLOAT", {"default": 0.5, "min": 0.25,
                              "max": 0.9, "step": 0.05, "tooltip":
                              "[BOTH auto & sigma] The STARTING low-res rung "
                              "(latent side scale). auto climbs from here to full "
                              "res; sigma stays here until switch_at. 0.25 = most "
                              "aggressive/fastest, but the ×0.25 rung is where the "
                              "edge VEIL is born → 0.5 starts one rung up = "
                              "cleaner edges at some speed cost. (Was ignored in "
                              "auto before; now live.)"}),
                "switch_at": ("FLOAT", {"default": 0.5, "min": 0.1,
                              "max": 0.9, "step": 0.05, "tooltip":
                              "[SIGMA mode ONLY] Fraction of the sigma range spent "
                              "at low resolution before the single switch to full. "
                              "Ignored in auto (auto climbs by its backbone_mode "
                              "schedule instead)."}),
            },
            "optional": {
                "switch_mode": (["auto", "sigma"], {"default": "auto",
                                "tooltip": "auto = ADAPTIVE ladder that climbs "
                                "×low_scale → … → full across the σ-range (uses "
                                "backbone_mode + speed_delta). sigma = ONE fixed "
                                "switch: low_scale until switch_at, then full "
                                "(uses low_scale + switch_at; ignores "
                                "backbone_mode). Most users want auto."}),
                "stab_threshold": ("FLOAT", {"default": 0.08, "min": 0.01,
                                   "max": 0.5, "step": 0.01, "tooltip":
                                   "Reserved / internal; currently unused."}),
                "verify_sigma": ("FLOAT", {"default": 0.0, "min": 0.0,
                                 "max": 1.0, "step": 0.05, "tooltip":
                                 "[BOTH modes] Verify (full-res correction) steps. Run a "
                                 "FULL-RES step while sigma/first_sigma >= this "
                                 "to clean contour noise born at low res. "
                                 "DEFAULT 0 = OFF = fastest (~2x on Krea) and "
                                 "clean on MOST prompts. Turn it ON only when a "
                                 "specific prompt shows contour noise / a "
                                 "translucent VEIL along garment edges. IMPORTANT "
                                 "for the veil: use ~0.9, NOT 1.0. Verify only "
                                 "corrects a step whose sigma/first_sigma >= this, "
                                 "and the veil is born on the aggressive ×0.25 "
                                 "structure steps (ratios ~1.0, 0.96, 0.90 at "
                                 "1536²). 1.0 catches only the FIRST of them → "
                                 "veil survives; 0.9 catches all three → veil "
                                 "cleared (costs ~2 extra full-res steps). Lower "
                                 "= cleaner+slower, higher = faster+risk."}),
                "carry_prev": ("BOOLEAN", {"default": True, "tooltip":
                               "[BOTH modes] Carry the previous step's full-res prediction as "
                               "the band-split signal. ON = original (cleaner "
                               "smooth areas). Turn OFF if you see a "
                               "semi-transparent VEIL / double contour along "
                               "garment or silhouette edges — off removes it "
                               "(input becomes a pure nearest-downscale)."}),
                "up_mode": (["bilinear", "bicubic", "nearest-exact", "edge"],
                            {"default": "bilinear", "tooltip":
                             "[BOTH modes] How the model output is upscaled back to full res. "
                             "bilinear = default. bicubic = slightly sharper. "
                             "nearest = hardest (usually worse). edge = "
                             "EDGE-MASKED: hard upscale on silhouette edges + "
                             "bilinear on smooth areas — targets the translucent "
                             "VEIL along garment edges without blocking flats. "
                             "Try this if you see the veil; free (no speed "
                             "cost)."}),
                "backbone_mode": (["empirical", "spectral"],
                                  {"default": "empirical", "tooltip":
                                   "[AUTO mode ONLY] How the auto ladder decides "
                                   "WHEN to grow the grid. empirical = fixed sigma "
                                   "thresholds (0.9→×0.25, 0.75→×0.5), the "
                                   "original. spectral = SPEED (arXiv 2605.18736): "
                                   "switch from the data power spectrum "
                                   "(principled, resolution-independent, grows "
                                   "earlier so edges lock less = less veil). Uses "
                                   "the fit (A,beta) for the current model. For BEST "
                                   "results fit YOUR model at YOUR resolution: run "
                                   "`tools/fit_spectrum.py` (2-3 samples) and set "
                                   "speed_A/speed_beta below. A baked-in default "
                                   "exists but may be wrong for your model/resolution "
                                   "— don't rely on it (leaving A/beta=0 uses it)."}),
                "speed_delta": ("FLOAT", {"default": 0.01, "min": 0.001,
                                "max": 0.5, "step": 0.005, "tooltip":
                                "[AUTO + spectral ONLY] noise-dominated tolerance. "
                                "Smaller = stay low-res longer (faster, more "
                                "veil risk); larger = grow earlier (cleaner, "
                                "slower). 0.01 = SPEED default."}),
                "speed_A": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0,
                            "step": 0.001, "tooltip":
                            "[AUTO + spectral] Power-spectrum amplitude A for THIS "
                            "model. Run `tools/fit_spectrum.py` on your model (at "
                            "your resolution) and paste its A here. 0 = fall back to "
                            "a baked-in default (may not match your model — fitting "
                            "yourself is recommended, esp. off 1024px)."}),
                "speed_beta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0,
                               "step": 0.01, "tooltip":
                               "[AUTO + spectral] Power-spectrum exponent beta for "
                               "THIS model (paste fit_spectrum.py's beta, with "
                               "speed_A). Typical 1.9-2.6. 0 = baked-in default."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, low_scale=0.5, switch_at=0.5,
              switch_mode="auto", stab_threshold=0.08, verify_sigma=0.0,
              carry_prev=True, up_mode="bilinear",
              backbone_mode="empirical", speed_delta=0.01,
              speed_A=0.0, speed_beta=0.0):
        _validate_diffusion_model_input(model, "QlipProgressive")
        patched = model.clone()
        if not enable:
            return (patched,)

        try:
            from qlip.inference.progressive_core import ProgressiveEngine
        except ImportError as exc:
            raise RuntimeError(
                "QlipProgressive requires the qlip package "
                f"(qlip.inference.progressive_core): {exc}") from exc

        # stable per-model tag so a QlipSpectrumFit result is cached &
        # reused across generations of the same model.
        try:
            dm = patched.model.diffusion_model
            model_tag = f"{type(dm).__name__}:{sum(1 for _ in dm.parameters())}"
        except Exception:
            model_tag = None

        # spectrum fit for `spectral` backbone. Priority:
        #   1. manual speed_A/speed_beta on this node (for OTHER models)
        #   2. an upstream qlip_spectrum in model_options
        #   3. None -> engine falls back to its built-in Krea-2 fit
        if speed_A > 0.0 and speed_beta > 0.0:
            fit_A, fit_beta = float(speed_A), float(speed_beta)
        else:
            spec = patched.model_options.get("qlip_spectrum")
            fit_A = float(spec["A"]) if spec else None
            fit_beta = float(spec["beta"]) if spec else None

        engine = ProgressiveEngine(
            switch_mode=switch_mode, low_scale=low_scale,
            switch_at=switch_at, stab_threshold=stab_threshold,
            correct_sigma=float(verify_sigma),
            carry_prev=bool(carry_prev), up_mode=str(up_mode),
            backbone_mode=str(backbone_mode),
            speed_delta=float(speed_delta), model_tag=model_tag,
            speed_A=fit_A, speed_beta=fit_beta)
        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            if prev_wrapper is not None:
                base_apply = apply_model

                def apply_model(x, t, **c):     # noqa: F811 — chained inner
                    return prev_wrapper(base_apply, {
                        "input": x, "timestep": t, "c": c,
                        "cond_or_uncond": args.get("cond_or_uncond", [0])})
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]
            try:
                sig = float(timestep.reshape(-1)[0])
            except Exception:
                return apply_model(x, timestep, **c)
            return engine.step(
                x, timestep, sig, c, apply_model,
                lane_key=tuple(args.get("cond_or_uncond", [0])))

        patched.set_model_unet_function_wrapper(unet_wrapper)
        print(f"[QlipProgressive] enabled: mode={switch_mode} (licensed "
              f"qlip engine) — early model calls run on a downscaled "
              f"latent; nested multimodal latents supported; your sampler "
              f"is untouched.")
        return (patched,)

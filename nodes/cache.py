"""Qlip Cache — step caching (skip denoising steps) for extra speed.

Third acceleration axis on top of Qlip engines: reuse the model's last real
output when the trajectory moves slowly. Wraps the UNet call via ComfyUI's
official `set_model_unet_function_wrapper` (no monkey-patching), so it works
over FP8/NVFP4/CASA engines and eager alike.

Only useful on MANY-step configs (LTX-2.3 dev 15+, FLUX Klein base 50). On
distilled 4–8 step models there's almost no inter-step redundancy — the node
warns and effectively no-ops.
"""

import torch

from .engine_loader import _validate_diffusion_model_input
from ..utils.diffusion_cache import CacheController
from ..utils.block_cache import BlockCacheController

# the controller from the most recent Qlip Cache node, so the Report node can
# read stats without a model input (model.clone() would not carry it anyway)
_LAST_CTRL = None
_BLOCK_CTRL = None   # active block-mode controller (wraps the SHARED diffusion model)


class QlipCache:
    """Enable EasyCache-style step skipping. Separate cache per CFG branch;
    audio is never cached (cheap, and modality desync breaks lip-sync)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model (many-step "
                          "configs benefit; distilled 4–8 step models barely)."}),
                "enable": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0,
                              "step": 0.01, "tooltip": "step mode: accumulated-"
                              "error budget before a forced recompute. block "
                              "mode: rel-L1 hidden-state diff at the Fn "
                              "boundary below which the middle blocks are "
                              "skipped (0.05–0.1 typical). Higher = more skips "
                              "= faster & riskier."}),
                "mode": (["step", "block"], {"default": "step", "tooltip":
                         "step = skip whole denoising steps (wrapper around "
                         "the model call; works over ANY engine). block = "
                         "DBCache-style: first fn_blocks always compute, "
                         "middle blocks are skipped via a cached residual, "
                         "last bn_blocks refine — possible because qlip "
                         "compiles each block as its own engine. block mode "
                         "also helps few-step distilled models."}),
            },
            "optional": {
                "method": (["easycache", "taylor", "hermite"],
                           {"default": "easycache", "tooltip":
                            "How skipped steps are predicted. easycache = reuse "
                            "the last residual — cheap, robust, and in our "
                            "measurements the best quality/speed on every model "
                            "so far (Krea-2, MiniMax-H3, Ideogram-4): the "
                            "DEFAULT, start here. taylor = TaylorSeer, "
                            "extrapolate output from history (can overshoot). "
                            "hermite = HiCache, damped extrapolation: order 1 "
                            "behaves like easycache; order >= 2 on many-step "
                            "models tends to re-decide the composition and add "
                            "a translucent mesh (measured on Ideogram-4: scene "
                            "replaced on 12-14/16 prompts) — try only if "
                            "easycache's texture tail fails the gate."}),
                "order": ("INT", {"default": 2, "min": 1, "max": 4, "tooltip":
                          "Extrapolation order for taylor/hermite. 2 is a good "
                          "default; higher = more history tensors cached."}),
                "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 20,
                                 "tooltip": "First steps always computed — they "
                                 "set composition/background. Do not lower much."}),
                "max_consecutive_skips": ("INT", {"default": 3, "min": 1, "max": 10,
                                          "tooltip": "Hard cap on skips in a row "
                                          "so error can't compound unbounded."}),
                "fn_blocks": ("INT", {"default": 8, "min": 1, "max": 64,
                              "tooltip": "block mode: first N blocks that "
                              "ALWAYS compute — their output is the probe "
                              "that decides skipping."}),
                "bn_blocks": ("INT", {"default": 0, "min": 0, "max": 64,
                              "tooltip": "block mode: last N blocks that "
                              "always compute (refinement tail)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, threshold=0.15, mode="step",
              method="hermite", order=2, warmup_steps=4,
              max_consecutive_skips=3, fn_blocks=8, bn_blocks=0):
        _validate_diffusion_model_input(model, "QlipCache")
        global _LAST_CTRL, _BLOCK_CTRL
        patched = model.clone()

        # block mode wraps the SHARED diffusion model's block forwards, so a
        # previous block controller must always be unwound first (also covers
        # switching block -> step and disable).
        if _BLOCK_CTRL is not None:
            _BLOCK_CTRL.uninstall()
            _BLOCK_CTRL = None

        if not enable:
            patched.model_options = dict(patched.model_options)
            # dropping our wrapper: cleanest is to just not add it on a fresh
            # clone. Nothing to remove since clone started from the base.
            return (patched,)

        if mode == "block":
            ctrl = BlockCacheController(
                threshold=threshold, fn_blocks=fn_blocks, bn_blocks=bn_blocks,
                warmup_steps=warmup_steps,
                max_consecutive_skips=max_consecutive_skips,
                method=method, order=order)
            dm = patched.model.diffusion_model
            info = ctrl.install(dm)
            _BLOCK_CTRL = ctrl
            _LAST_CTRL = ctrl

            def reset_wrapper(executor, *a, **kw):
                ctrl.reset()
                return executor(*a, **kw)
            try:
                import comfy.patcher_extension as pe
                patched.add_wrapper(pe.WrappersMP.SAMPLER_SAMPLE,
                                    reset_wrapper)
            except Exception:
                pass

            print(f"[QlipCache] block mode: {info['blocks']} blocks "
                  f"({info.get('container')}), "
                  f"Fn={info['fn']} Bn={info['bn']} "
                  f"middle={info['middle']} threshold={threshold} "
                  f"predictor={method}. Works over per-block qlip engines "
                  f"and eager alike; helps few-step models too.")
            return (patched,)

        ctrl = CacheController(warmup_steps=warmup_steps,
                               max_consecutive_skips=max_consecutive_skips,
                               threshold=threshold, method=method, order=order)
        _LAST_CTRL = ctrl

        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            """ComfyUI calls: wrapper(model.apply_model, {"input", "timestep",
            "c", "cond_or_uncond"}). The result is chunked per branch, so the
            batch here holds one row per entry in cond_or_uncond. We keep a
            separate cache lane per branch so cond/uncond never share a
            predictor (their trajectories differ — sharing one is the
            classic caching bug)."""
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]
            branches = args.get("cond_or_uncond", [0])

            if prev_wrapper is not None:
                _base = apply_model

                def apply_model(xx, tt, **cc):   # noqa: F811 — chained inner
                    return prev_wrapper(_base, {
                        "input": xx, "timestep": tt, "c": cc,
                        "cond_or_uncond": branches})

            # When cond and uncond are batched into ONE call (len>1), skipping
            # would have to be an all-or-nothing decision on the shared tensor.
            # We gate on the WHOLE batch using one lane keyed by the branch
            # set — correct, just coarser than per-branch. Separate calls
            # (the common LTX path) give one branch → true per-branch caching.
            key = "video_" + "_".join(map(str, branches))
            st = ctrl.lane(key)

            # sigma identifies the sampler step; timestep IS the sigma tensor
            # here. Multi-stage samplers (RES4LYF res_2s) call us several times
            # per step with the SAME sigma — only the FIRST such call opens a
            # new step and may be cached; the sub-calls must always run the
            # real model or the sampler's stage math breaks.
            try:
                sigma = float(timestep.reshape(-1)[0])
            except Exception:
                sigma = None
            new_step = st.begin(sigma)

            if st.prev_input is not None and st.prev_input.shape != x.shape:
                # resolution changed mid-run (e.g. QlipProgressive low-res
                # phase) — history is incomparable; start the lane fresh
                st.reset()
            din = 0.0 if st.prev_input is None \
                else float((x - st.prev_input).norm())

            can_skip = new_step and not st.should_compute(float(x.norm()), din)
            if can_skip:
                out = st.predict(x)
            else:
                out = apply_model(x, timestep, **c)
                st.record_real(x, out)
            return out

        patched.set_model_unet_function_wrapper(unet_wrapper)

        # reset lanes at the start of each sampling run so state doesn't leak
        # between generations
        def reset_wrapper(executor, *a, **kw):
            ctrl.reset()
            return executor(*a, **kw)
        try:
            import comfy.patcher_extension as pe
            patched.add_wrapper(pe.WrappersMP.SAMPLER_SAMPLE, reset_wrapper)
        except Exception:
            pass  # older ComfyUI: reset on first warmup step instead

        print(f"[QlipCache] enabled: method={method} order={order} "
              f"threshold={threshold} warmup={warmup_steps} "
              f"max_skips={max_consecutive_skips}. Best on many-step configs; "
              f"near-useless on distilled 4–8 step models.")
        return (patched,)


class QlipCacheReport:
    """Print cache hit-rate and the resulting compute ratio / ideal speedup."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {"trigger": ("*", {"tooltip": "Connect a post-sampler "
                         "output so this runs after generation. No model input "
                         "needed."})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)
    FUNCTION = "report"
    CATEGORY = "qlip"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def report(self, trigger=None):
        ctrl = _LAST_CTRL
        if ctrl is None:
            msg = "[QlipCache] no cache active — add a Qlip Cache node first."
            print(msg)
            return (msg,)
        s = ctrl.summary()
        if s["total"] == 0:
            msg = "[QlipCache] no steps recorded yet — generate once."
            print(msg)
            return (msg,)
        note = ("" if s["skipped_steps"] else
                "  (0 skips — either warmup covered the whole run, few steps, "
                "or a distilled model with no inter-step redundancy)")
        msg = (f"[QlipCache] {s['real_steps']} real / {s['skipped_steps']} "
               f"skipped of {s['total']} → compute ratio {s['compute_ratio']}, "
               f"ideal speedup {s['ideal_speedup']}x{note}\n"
               f"  (measure wall-clock with Qlip Timer for the real number)")
        print(msg)
        return (msg,)

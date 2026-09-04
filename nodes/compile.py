"""Qlip Compile — plan-first compilation of the diffusion model (Loom).

Fifth acceleration axis: the previous nodes cut the AMOUNT of work (sparse
attention, caching, progressive resolution); this one speeds up EXECUTION of
what remains. It discovers the transformer block stacks, compiles each block
class once through torch.compile, memoizes shape-pure glue (RoPE embedders)
and can put every large block Linear on the FP8 path (qlip quantization
scheme: e4m3 + amax scales, executed via native H100/Blackwell scaled GEMM
— no ONNX export, no engine build).

Vs whole-model torch.compile: same steady speed (Z-Image 1024²: 118 ms vs
118.4 tc-default, eager 134.7) but cold start is seconds instead of minutes
and RESOLUTION CHANGES NEED NO RECOMPILE (0.7 s vs 79–156 s). With fp8:
94.7 ms (1.42×), cosine 0.9957 to eager.

Install is LAZY and happens on the FIRST MODEL CALL (unet wrapper) — at
that point weights are guaranteed on the GPU and LoRA patches are applied
(the sampler-level hook fires too early: ComfyUI moves weights to the
device after it). The first run with the node therefore includes one-time
compilation (~10–20 s on top); every later run — and every new resolution —
is fast. With fp8 + calibrate-first-run, the first run doubles as the
calibration pass (max observer); scales freeze at its end and later runs
use static scales, like the TRT-engine deploy.

Compose order: QlipAutoSparse → QlipCompile → QlipCache → QlipProgressive.
Whatever attention/patches are installed when the first model call happens
is what gets compiled; a block that fails to compile permanently falls back
to eager — the run never breaks.
"""

import time

from .engine_loader import _validate_diffusion_model_input
from ..utils.loom import (attach, dequantize_linears, detach,
                          ensure_untouched_linears_cast, find_regions,
                          freeze_scales, memoize_by_shape, quantize_linears,
                          release_masters, start_calibration, stats,
                          unmemoize)

# shared diffusion models are mutated in place; ONE install per model may
# be active (MoE like Wan 2.2 runs two experts = two models, each with its
# own node — they must not unwind each other), keyed by the model object
_ACTIVE = {}


def _uninstall(dm=None):
    """Unwind the install for `dm` only, or for all models if dm is None."""
    for key in [k for k, e in list(_ACTIVE.items())
                if dm is None or e["dm"] is dm]:
        ent = _ACTIVE.pop(key)
        try:
            detach(ent["dm"])
            dequantize_linears([ent["dm"]])
            for m in ent.get("memoized", []):
                unmemoize(m)
            print("[QlipCompile] previous install unwound")
        except Exception as exc:   # noqa: BLE001
            print(f"[QlipCompile] uninstall warning: {exc}")


class QlipCompile:
    """Loom: per-block compilation + optional FP8, no sampler changes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Any DiT. Works over eager "
                          "and composes with the other Qlip nodes."}),
                "enable": ("BOOLEAN", {"default": True}),
                "quantize": (["none", "fp8", "fp4"], {"default": "none",
                             "tooltip": "fp8 = e4m3 scaled GEMM (H100+). "
                             "fp4 = nvfp4 (e2m1 + e4m3 block scales), "
                             "Blackwell sm_100+ only — falls back to fp8 "
                             "on older GPUs. Both use native tensor cores, "
                             "no ONNX export."}),
            },
            "optional": {
                "backend": (["default", "max-autotune"],
                            {"default": "default", "tooltip":
                             "torch.compile mode for the blocks. "
                             "max-autotune compiles much longer for ~1% "
                             "extra; default is the right choice."}),
                "act_scales": (["calibrate-first-run", "dynamic"],
                               {"default": "calibrate-first-run", "tooltip":
                                "fp8 only. calibrate-first-run: first "
                                "sampling run records amax (max observer), "
                                "then scales freeze static (fastest, "
                                "TRT-like). dynamic: per-call amax forever "
                                "(no calibration warmup, ~6% slower)."}),
                "force_resident": ("BOOLEAN", {"default": True, "tooltip":
                                    "Dynamic-VRAM (streamed) models are "
                                    "converted to fully resident before "
                                    "compilation — compiled steps must not "
                                    "be PCIe-bound (same effect as "
                                    "--highvram, but only for this model). "
                                    "Disable to keep weight streaming."}),
                "attention": (["comfy", "auto", "int8_fp8", "int8_fp16", "fp4"], {
                    "default": "comfy",
                    "tooltip": "Attention kernel inside the compiled blocks. "
                               "comfy = whatever ComfyUI uses (SDPA, or sage if "
                               "launched with --use-sage-attention; loom marks "
                               "it opaque). auto/int8_fp8/int8_fp16/fp4 = qlip's "
                               "own dependency-free SageAttention-class kernels "
                               "(int8 Q/K + fp8 P·V; fp4 = Blackwell, phase 2). "
                               "Applied to long unmasked self-attention only "
                               "(>= 2048 tokens); everything else stays on the "
                               "original path."}),
                "weights_policy": (["keep", "release"], {"default": "keep",
                                   "tooltip": "release: after fp8/fp4 "
                                   "quantization FREE the master weights — "
                                   "big VRAM win (e.g. fp4 on a 13 GB fp8 "
                                   "checkpoint nets ~-6.5 GB). IRREVERSIBLE "
                                   "until the checkpoint is reloaded; apply "
                                   "LoRAs before this node."}),
                "quant_config": ("QLIP_QUANT", {"tooltip":
                                 "Optional QlipQuantConfig node — full "
                                 "control (granularity, observer, calib "
                                 "runs, layer skips). When connected it "
                                 "overrides quantize/act_scales and turns "
                                 "quantization ON."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    def apply(self, model, enable=True, quantize="none", backend="default",
              act_scales="calibrate-first-run", quant_config=None,
              force_resident=True, weights_policy="keep", attention="comfy"):
        _validate_diffusion_model_input(model, "QlipCompile")
        patched = model.clone()
        from ..utils.qattn_router import (install_attention, polish_external_sage,
                                          uninstall_attention)
        # attention kernel selection is its own axis (process-wide: comfy's
        # optimized_attention) — applied even when compile is disabled, so the
        # kernel can be A/B'd on the eager model too.
        polish_external_sage()
        if attention != "comfy":
            install_attention(attention)
        else:
            uninstall_attention()
        if not enable:
            _uninstall(patched.model.diffusion_model)
            return (patched,)

        # DYNAMIC VRAM (weight streaming): a compiled model must not be
        # PCIe-bound — fusion can't speed up weight delivery, and fp8 barely
        # helps while comfy keeps streaming the bf16 masters. Convert to a
        # regular resident ModelPatcher via comfy's own delegate mechanism
        # (the per-model equivalent of --highvram).
        if force_resident and getattr(patched, "is_dynamic", lambda: False)():
            try:
                patched = patched.get_non_dynamic_delegate()
                print("[QlipCompile] dynamic-VRAM model converted to "
                      "resident (highvram-like) — no weight streaming "
                      "during compiled steps")
            except Exception as exc:   # noqa: BLE001
                print(f"[QlipCompile] could not disable dynamic VRAM "
                      f"({exc}) — weights will stream, steps may be "
                      f"PCIe-bound")

        # EAGER dtype guard: making the model resident (or comfy's newer
        # CastBiasWeightContext) can leave OUT-OF-REGION Linears (Krea-2
        # self.first/last/tmlp) with an fp32 master weight while activations
        # are bf16 -> `mat1 and mat2 must have the same dtype` on the very
        # first (warm-up) forward, BEFORE our deferred _install runs. Apply
        # the guard now, scoped to the real block regions so region Linears
        # (which get quantized later) are untouched. Idempotent with the
        # copy _install() re-applies.
        if enable:
            try:
                dm0 = patched.model.diffusion_model
                _regs0 = find_regions(dm0, min_repeat=2)
                ng = ensure_untouched_linears_cast(dm0, _regs0)
                if ng:
                    print(f"[QlipCompile] dtype-cast guard on {ng} "
                          f"out-of-region Linear(s) (pre-install)")
            except Exception as exc:   # noqa: BLE001 — never block arming
                print(f"[QlipCompile] pre-install cast guard skipped ({exc})")

        # STREAMED (dynamic-VRAM) models: their loader owns weight
        # placement (vbar pools, per-cycle set_weight) — on-the-fly
        # quantization/release corrupts neighbouring weights (observed on
        # MiniMax H3). If the model could not be made resident, compile
        # WITHOUT touching weights and point to the offline path.
        if (quantize != "none"
                and getattr(patched, "is_dynamic", lambda: False)()):
            print("[QlipCompile] streamed (dynamic-VRAM) model — on-the-fly "
                  "quantization is not supported here; falling back to "
                  "quantize=none (block compilation only). For this model "
                  "use the offline qlip-engines pipeline instead.")
            quantize = "none"
            weights_policy = "keep"

        state = {"installed": False, "runs": 0, "regs": None,
                 "frozen": False, "install_after": 0, "cold_msg": False,
                 "installed_at_run": 0}
        qc = dict(quant_config) if quant_config else None
        if qc is not None:
            quantize = "fp4" if qc.get("scheme") == "nvfp4" else "fp8"
            act_scales = ("dynamic" if qc.get("act_scales") == "dynamic"
                          else "calibrate-first-run")
        else:
            qc = {"weight_granularity": "per-tensor", "calib_runs": 1,
                  "observer": "max", "ema_decay": 0.8, "min_dim": 512,
                  "skip": ()}
        qc.setdefault("scheme",
                      "nvfp4" if quantize == "fp4" else "fp8_e4m3")
        mode = ("max-autotune-no-cudagraphs" if backend == "max-autotune"
                else "default")

        # COLD MODEL: if the model is not resident on the GPU yet, ComfyUI
        # finishes loading/patching weights DURING the first sampling run —
        # installing mid-load catches weights in transient states (observed:
        # a weight with its storage reinterpreted at half width). Rule: cold
        # model → first run is a plain warm-up, install on the next run.
        try:
            import comfy.model_management as mm

            def _same(lm):
                m = getattr(lm, "model", None)
                return (m is patched.model
                        or getattr(m, "model", None) is patched.model)

            warm = any(_same(lm) for lm in mm.loaded_models())
        except Exception:   # noqa: BLE001
            warm = False
        state["install_after"] = 0 if warm else 1

        def _install(dm, device):
            _uninstall(dm)
            t0 = time.time()
            try:
                regs = find_regions(dm, min_repeat=2)
                if not regs:
                    print("[QlipCompile] no block stacks found — model "
                          "left untouched")
                    return
                if quantize in ("fp8", "fp4"):
                    nq = quantize_linears(
                        regs[0]["modules"], device=device,
                        min_dim=qc["min_dim"],
                        granularity=qc["weight_granularity"],
                        skip=qc["skip"], scheme=qc["scheme"],
                        release_each=(weights_policy == "release"))
                    if (qc["scheme"] != "nvfp4"
                            and act_scales == "calibrate-first-run"
                            and qc["weight_granularity"] == "per-tensor"):
                        start_calibration(regs[0]["modules"],
                                          observer=qc["observer"],
                                          ema_decay=qc["ema_decay"])
                    print(f"[QlipCompile] {qc['scheme']}: {nq} linears "
                          f"on {device}")
                    if weights_policy == "release" and nq:
                        freed = release_masters(regs[0]["modules"])
                        print(f"[QlipCompile] masters released "
                              f"(interleaved) — quantized copies are now "
                              f"the only weights; irreversible until "
                              f"checkpoint reload")
                rep = attach(dm, regions=regs, mode=mode)
                memoized = []
                for _, m in dm.named_modules():
                    if "EmbedND" in type(m).__name__:
                        memoize_by_shape(m)
                        memoized.append(m)
                _ACTIVE[id(dm)] = {"dm": dm, "memoized": memoized}
                state["regs"] = regs
                print(f"[QlipCompile] attached "
                      f"{[(r['name'], r['blocks']) for r in rep['regions']]}"
                      f", rope memo x{len(memoized)}, setup "
                      f"{time.time() - t0:.1f}s. THIS run compiles as it "
                      f"goes (~10-20 s one-time); later runs and new "
                      f"resolutions are fast.")
            except Exception as exc:   # noqa: BLE001 — atomic: full rollback
                print(f"[QlipCompile] install failed ({exc}) — rolled "
                      f"back, running the model untouched")
                try:
                    detach(dm)
                    dequantize_linears([dm])
                except Exception:   # noqa: BLE001
                    pass
                # detach() strips the out-of-region dtype-cast guards, but the
                # force_resident conversion done at arming time is NOT rolled
                # back — so the untouched model would now crash on its fp32
                # first/last projections (mat1/mat2 dtype). Re-arm the guard.
                try:
                    ensure_untouched_linears_cast(dm, find_regions(dm, min_repeat=2))
                except Exception:   # noqa: BLE001
                    pass
                _ACTIVE.pop(id(dm), None)
                state["regs"] = None

        # install on the FIRST MODEL CALL: weights are on-device and LoRA
        # patches applied by then (a sampler-level hook fires too early);
        # the compute device is taken from the live input tensor, NOT from
        # the weights (manual-cast models keep master weights on CPU)
        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            if not state["installed"]:
                if state["runs"] >= state["install_after"]:
                    state["installed"] = True
                    state["installed_at_run"] = state["runs"]
                    _install(patched.model.diffusion_model,
                             args["input"].device)
                elif not state["cold_msg"]:
                    state["cold_msg"] = True
                    print("[QlipCompile] cold model load — this run is a "
                          "plain warm-up; compiling from the next run")
            fn = apply_model
            if prev_wrapper is not None:
                def fn(x, t, **c):   # noqa: F811 — chained inner
                    return prev_wrapper(apply_model, {
                        "input": x, "timestep": t, "c": c,
                        "cond_or_uncond": args.get("cond_or_uncond", [0])})
            return fn(args["input"], args["timestep"], **args["c"])

        patched.set_model_unet_function_wrapper(unet_wrapper)

        # run-end hook: freeze calibrated scales after the first run and
        # surface any per-block fallbacks
        def run_end_wrapper(executor, *args, **kwargs):
            out = executor(*args, **kwargs)
            state["runs"] += 1
            calib_done = (state["runs"] - state["installed_at_run"]
                          >= qc["calib_runs"])
            if (quantize == "fp8" and qc["scheme"] != "nvfp4"
                    and act_scales == "calibrate-first-run"
                    and state["installed"] and not state["frozen"]
                    and calib_done and state["regs"]):
                state["frozen"] = True
                nf = freeze_scales(state["regs"][0]["modules"])
                print(f"[QlipCompile] calibration done "
                      f"({qc['calib_runs']} run(s)) — {nf} static act "
                      f"scales frozen; next runs use the fast path")
            if state["regs"]:
                failed = stats(patched.model.diffusion_model)["failed"]
                if failed:
                    print(f"[QlipCompile] eager fallbacks: {failed}")
            return out

        try:
            import comfy.patcher_extension as pe
            patched.add_wrapper(pe.WrappersMP.SAMPLER_SAMPLE,
                                run_end_wrapper)
        except Exception:   # noqa: BLE001 — old ComfyUI: stay dynamic
            if quantize == "fp8" and act_scales == "calibrate-first-run":
                print("[QlipCompile] no sampler hook in this ComfyUI — "
                      "fp8 stays on dynamic scales")

        print(f"[QlipCompile] armed: quantize={quantize} backend={backend} "
              f"act_scales={act_scales}. Install happens on the first "
              f"model call.")
        return (patched,)


class QlipQuantConfig:
    """Quantization settings for QlipCompile — qlip.quantization terms
    (scheme / granularity / observer), executed by the Loom fp8 path."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheme": (["fp8_e4m3", "nvfp4"], {"default": "fp8_e4m3",
                           "tooltip": "qlip QSchemeType. fp8_e4m3 = H100+. "
                           "nvfp4 = e2m1 + per-16 e4m3 block scales, "
                           "Blackwell sm_100+ tensor cores (falls back to "
                           "fp8 elsewhere); granularity/observer/calib "
                           "settings don't apply to nvfp4 (its scales are "
                           "per-block by construction)."}),
                "weight_granularity": (["per-tensor", "per-channel"],
                                       {"default": "per-tensor", "tooltip":
                                        "qlip QuantGranularity. per-channel "
                                        "= rowwise scaled GEMM: weight "
                                        "scale per output channel, act "
                                        "scale per token — more accurate, "
                                        "slightly slower, no calibration "
                                        "needed."}),
                "act_scales": (["calibrate", "dynamic"],
                               {"default": "calibrate", "tooltip":
                                "per-tensor only. calibrate: observer "
                                "collects amax during calib_runs sampling "
                                "runs, then scales freeze static (TRT-like "
                                "deploy). dynamic: per-call amax forever."}),
                "calib_runs": ("INT", {"default": 1, "min": 1, "max": 20,
                               "tooltip": "How many sampling runs feed the "
                               "observer before scales freeze."}),
                "observer": (["max", "ema"], {"default": "max", "tooltip":
                             "qlip StatMinMaxObserver flavors: max = "
                             "running maximum (safe), ema = exponential "
                             "moving average (ignores rare spikes)."}),
                "ema_decay": ("FLOAT", {"default": 0.8, "min": 0.1,
                              "max": 0.99, "step": 0.01, "tooltip":
                              "EMA observer: scale = decay*new + "
                              "(1-decay)*current (qlip default 0.8)."}),
                "min_dim": ("INT", {"default": 512, "min": 16, "max": 8192,
                            "tooltip": "Linears with any side smaller than "
                            "this stay unquantized (tiny projections gain "
                            "nothing, lose accuracy)."}),
                "skip_layers": ("STRING", {"default": "", "tooltip":
                                "Comma-separated name substrings to keep "
                                "in high precision, e.g. 'qkv' or "
                                "'feed_forward.w2'."}),
            },
        }

    RETURN_TYPES = ("QLIP_QUANT",)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "make"
    CATEGORY = "qlip"

    def make(self, scheme, weight_granularity, act_scales, calib_runs,
             observer, ema_decay, min_dim, skip_layers):
        cfg = {
            "scheme": scheme,
            "weight_granularity": weight_granularity,
            "act_scales": act_scales,
            "calib_runs": int(calib_runs),
            "observer": observer,
            "ema_decay": float(ema_decay),
            "min_dim": int(min_dim),
            "skip": tuple(x.strip() for x in skip_layers.split(",")
                          if x.strip()),
        }
        print(f"[QlipQuantConfig] {cfg}")
        return (cfg,)

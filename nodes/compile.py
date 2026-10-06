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

from qlip.inference.loom import (
    attach,
    dequantize_linears,
    detach,
    ensure_persistent_compile_cache,
    ensure_untouched_linears_cast,
    find_regions,
    freeze_scales,
    memoize_by_shape,
    quantize_linears,
    release_masters,
    start_calibration,
    stats,
    unmemoize,
)

from ..utils.lora_peft import lora_mode_help
from .engine_loader import _validate_diffusion_model_input

LORA_MODE_HELP = lora_mode_help()

# shared diffusion models are mutated in place; ONE install per model may
# be active (MoE like Wan 2.2 runs two experts = two models, each with its
# own node — they must not unwind each other), keyed by the model object
_ACTIVE = {}
# id(shared base model of a dynamic-VRAM patcher) -> (base model, resident
# model override). ComfyUI clones of one checkpoint (LoraLoader, our merge,
# other nodes) share ``patcher.model``; keying the resident copy on it lets
# every clone reuse ONE resident model (patches applied at load, as comfy does
# for ordinary clones) instead of a new full copy per clone.
_RESIDENT = {}


def _resident_delegate(model):
    """Non-dynamic (resident) clone of ``model`` that reuses one resident
    model per base checkpoint across clones and node re-executions."""
    key = id(model.model)
    ent = _RESIDENT.get(key)
    override = ent[1] if ent is not None and ent[0] is model.model else None
    mp = model.clone(disable_dynamic=True, model_override=override)
    _RESIDENT[key] = (model.model, mp.get_clone_model_override())
    return mp


def _unwrap_compiled(dm):
    """Put the original blocks back in place of package-backed
    CompiledModules (their `model` is not a registered submodule, so the
    quantized linears inside are invisible to dequantize_linears until the
    wrappers are gone)."""
    import gc

    import torch
    from qlip.inference import CompiledModule

    n = 0
    for name, m in list(dm.named_modules()):
        if isinstance(m, CompiledModule):
            *parent, child = name.split(".")
            owner = dm.get_submodule(".".join(parent)) if parent else dm
            release = getattr(m.session, "release", None)
            if release is not None:
                release()  # runner held refs to the bound weights
            m.unload()
            setattr(owner, child, m.model)
            n += 1
    if n:
        if hasattr(dm, "_qlip_imanager"):
            del dm._qlip_imanager
        gc.collect()  # sessions sit in adapter reference cycles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _uninstall(dm=None):
    """Unwind the install for `dm` only, or for all models if dm is None."""
    for key in [k for k, e in list(_ACTIVE.items()) if dm is None or e["dm"] is dm]:
        ent = _ACTIVE.pop(key)
        try:
            if ent.get("aot") is not None:
                _unwrap_compiled(ent["dm"])
            detach(ent["dm"])
            # the PEFT side path subclasses the fp8 linear: take it off first,
            # or dequantize would restore the base class under stale LoRA
            # attributes and a reinstall would think the side path is still on
            from qlip.inference.loom.lora import uninstall_lora

            uninstall_lora([ent["dm"]])
            dequantize_linears([ent["dm"]])
            for m in ent.get("memoized", []):
                unmemoize(m)
            print("[QlipCompile] previous install unwound")
        except Exception as exc:  # noqa: BLE001
            print(f"[QlipCompile] uninstall warning: {exc}")


class QlipCompile:
    """Loom: per-block compilation + optional FP8, no sampler changes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Any DiT. Works over eager "
                        "and composes with the other Qlip nodes."
                    },
                ),
                "enable": ("BOOLEAN", {"default": True}),
                "quantize": (
                    ["none", "fp8", "fp4"],
                    {
                        "default": "none",
                        "tooltip": "fp8 = e4m3 scaled GEMM (H100+). "
                        "fp4 = nvfp4 (e2m1 + e4m3 block scales), "
                        "Blackwell sm_100+ only — falls back to fp8 "
                        "on older GPUs. Both use native tensor cores, "
                        "no ONNX export.",
                    },
                ),
            },
            "optional": {
                "backend": (
                    ["default", "max-autotune"],
                    {
                        "default": "default",
                        "tooltip": "torch.compile mode for the blocks. "
                        "max-autotune compiles much longer for ~1% "
                        "extra; default is the right choice.",
                    },
                ),
                "act_scales": (
                    ["calibrate-first-run", "dynamic"],
                    {
                        "default": "calibrate-first-run",
                        "tooltip": "fp8 only. calibrate-first-run: first "
                        "sampling run records amax (max observer), "
                        "then scales freeze static (fastest, "
                        "TRT-like). dynamic: per-call amax forever "
                        "(no calibration warmup, ~6% slower).",
                    },
                ),
                "force_resident": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Dynamic-VRAM (streamed) models are "
                        "converted to fully resident before "
                        "compilation — compiled steps must not "
                        "be PCIe-bound (same effect as "
                        "--highvram, but only for this model). "
                        "Disable to keep weight streaming.",
                    },
                ),
                "attention": (
                    ["comfy", "auto", "int8_fp8", "int8_fp16", "fp4"],
                    {
                        "default": "comfy",
                        "tooltip": "Attention kernel inside the compiled blocks. "
                        "comfy = whatever ComfyUI uses (SDPA, or sage if "
                        "launched with --use-sage-attention; loom marks "
                        "it opaque). auto/int8_fp8/int8_fp16/fp4 = qlip's "
                        "own dependency-free SageAttention-class kernels "
                        "(int8 Q/K + fp8 P·V; fp4 = Blackwell, phase 2). "
                        "Applied to long unmasked self-attention only "
                        "(>= 2048 tokens); everything else stays on the "
                        "original path.",
                    },
                ),
                "weights_policy": (
                    ["keep", "release"],
                    {
                        "default": "keep",
                        "tooltip": "release: after fp8/fp4 "
                        "quantization FREE the master weights — "
                        "big VRAM win (e.g. fp4 on a 13 GB fp8 "
                        "checkpoint nets ~-6.5 GB). IRREVERSIBLE "
                        "until the checkpoint is reloaded; apply "
                        "LoRAs before this node.",
                    },
                ),
                "engines_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional directory for ahead-of-time "
                        "Inductor packages (same workspace format "
                        "as qlip engines; QlipEnginesLoader loads "
                        "it too). Empty: pure JIT. Missing/stale: "
                        "JIT as usual, then the packages are built "
                        "once at the end of the first compiled run. "
                        "Ready: blocks load from the packages — no "
                        "compilation, no calibration run.",
                    },
                ),
                "shapes": (
                    ["auto", "dynamic"],
                    {
                        "default": "auto",
                        "tooltip": "JIT only. auto: torch's default — compile for the "
                        "first resolution, recompile once (dynamic) on the "
                        "first resolution / prompt-length change. dynamic: "
                        "symbolic token dims from the first compile, so "
                        "resolution changes never recompile. AOT packages "
                        "are always dynamic in the token dims.",
                    },
                ),
                "aot_autotune": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "engines_dir build only: Inductor max-autotune "
                        "(GEMM template search) while building the "
                        "packages. Paid once at build time, never at "
                        "load; changes the recipe (rebuild).",
                    },
                ),
                "lora_stack": (
                    "QLIP_LORA_STACK",
                    {
                        "tooltip": "Optional LoRA stack (QlipLoraStack). How it is "
                        "applied is set by lora_mode."
                    },
                ),
                "lora_mode": (
                    ["merge", "swap"],
                    {"default": "merge", "tooltip": LORA_MODE_HELP},
                ),
                "lora_max_rank": (
                    "INT",
                    {
                        "default": 256,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "swap only: rank budget of the side path per "
                        "layer (fused q/k/v layers get one budget per "
                        "slice). Must cover the largest LoRA (sum of "
                        "ranks when stacking). Changing it reinstalls.",
                    },
                ),
                "prefetch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Read the checkpoint into the page cache in "
                        "background threads as soon as this node runs "
                        "(overlaps with prompt encoding). Only matters "
                        "on a cold disk; a cached file is skipped.",
                    },
                ),
                "quant_config": (
                    "QLIP_QUANT",
                    {
                        "tooltip": "Optional QlipQuantConfig node — full "
                        "control (granularity, observer, calib "
                        "runs, layer skips). When connected it "
                        "overrides quantize/act_scales and turns "
                        "quantization ON."
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    def apply(
        self,
        model,
        enable=True,
        quantize="none",
        backend="default",
        act_scales="calibrate-first-run",
        quant_config=None,
        force_resident=True,
        weights_policy="keep",
        attention="comfy",
        engines_dir="",
        prefetch=True,
        shapes="auto",
        aot_autotune=False,
        lora_stack=None,
        lora_mode="merge",
        lora_max_rank=256,
    ):
        _validate_diffusion_model_input(model, "QlipCompile")
        from ..utils import lora_peft

        lora_stack = list(lora_stack or [])
        # DYNAMIC VRAM (weight streaming): a compiled model must not be
        # PCIe-bound — fusion can't speed up weight delivery, and fp8 barely
        # helps while comfy keeps streaming the bf16 masters. Convert to a
        # regular resident ModelPatcher via comfy's own delegate mechanism
        # (the per-model equivalent of --highvram).
        # One resident copy per base checkpoint (_resident_delegate): comfy's
        # get_non_dynamic_delegate() caches it on the patcher it was asked
        # on, and every re-execution (LoRA change, upstream node re-run)
        # hands us a NEW patcher — so each one made a new resident copy of
        # the whole model (Krea 2: +36 GB per LoRA change) with a new
        # diffusion_model object, and the in-place LoRA swap never matched.
        to_resident = False
        if enable and force_resident and getattr(model, "is_dynamic", lambda: False)():
            try:
                model = _resident_delegate(model)
                to_resident = True
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[QlipCompile] could not disable dynamic VRAM "
                    f"({exc}) — weights will stream, steps may be "
                    f"PCIe-bound"
                )
        if lora_mode == "merge" and lora_stack:
            # ComfyUI's own LoRA patching (same as LoraLoader upstream)
            model = lora_peft.merge(model, lora_stack)
        patched = model.clone()
        # everything that decides the install except the LoRA stack itself:
        # in swap mode a change of ONLY the stack is an in-place swap
        sig = (
            quantize,
            backend,
            act_scales,
            repr(quant_config),
            force_resident,
            weights_policy,
            attention,
            (engines_dir or "").strip(),
            shapes,
            bool(aot_autotune),
            lora_mode,
            int(lora_max_rank),
        )
        dm_now = patched.model.diffusion_model
        ent = _ACTIVE.get(id(dm_now))
        if (
            enable
            and lora_mode == "swap"
            and ent is not None
            and ent.get("sig") == sig
            and ent.get("state", {}).get("installed")
            and lora_peft.installed(dm_now)
        ):
            t0 = time.time()
            rep = lora_peft.swap(dm_now, lora_stack, patched)
            unet_w, run_end_w = ent["state"]["wrappers"]
            patched.set_model_unet_function_wrapper(unet_w)
            try:
                import comfy.patcher_extension as pe

                patched.add_wrapper(pe.WrappersMP.SAMPLER_SAMPLE, run_end_w)
            except Exception:  # noqa: BLE001
                pass
            print(
                f"[QlipCompile] LoRA swapped in place in "
                f"{time.time() - t0:.1f}s (no reinstall, no recompile): "
                f"{[(e['path'].rsplit('/', 1)[-1], e.get('strength', 1.0)) for e in lora_stack] or 'off'}"
                f" {rep if not lora_stack else ''}"
            )
            return (patched,)
        from ..utils.qattn_router import (
            install_attention,
            polish_external_sage,
            uninstall_attention,
        )

        # attention kernel selection is its own axis (process-wide: comfy's
        # optimized_attention) — applied even when compile is disabled, so the
        # kernel can be A/B'd on the eager model too.
        polish_external_sage()
        if attention != "comfy":
            install_attention(attention)
        else:
            uninstall_attention()
        # A re-executed node (new LoRA / strength / settings upstream) must
        # drop the PREVIOUS install now, not at the next first model call:
        # our fp8 copies keep the old (LoRA-patched) weight tensors alive, and
        # ComfyUI re-patches by allocating NEW tensors during the next load —
        # holding both doubles the model in VRAM for the switch (Flux2-dev:
        # 35 GB -> OOM next to any other GPU tenant).
        _uninstall(patched.model.diffusion_model)
        if not enable:
            return (patched,)

        if to_resident:
            print(
                "[QlipCompile] dynamic-VRAM model converted to "
                "resident (highvram-like) — no weight streaming "
                "during compiled steps"
            )

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
                    print(
                        f"[QlipCompile] dtype-cast guard on {ng} "
                        f"out-of-region Linear(s) (pre-install)"
                    )
            except Exception as exc:  # noqa: BLE001 — never block arming
                print(f"[QlipCompile] pre-install cast guard skipped ({exc})")

        # STREAMED (dynamic-VRAM) models: their loader owns weight
        # placement (vbar pools, per-cycle set_weight) — on-the-fly
        # quantization/release corrupts neighbouring weights (observed on
        # MiniMax H3). If the model could not be made resident, compile
        # WITHOUT touching weights and point to the offline path.
        if quantize != "none" and getattr(patched, "is_dynamic", lambda: False)():
            print(
                "[QlipCompile] streamed (dynamic-VRAM) model — on-the-fly "
                "quantization is not supported here; falling back to "
                "quantize=none (block compilation only). For this model "
                "use the offline qlip-engines pipeline instead."
            )
            quantize = "none"
            weights_policy = "keep"

        # COLD DISK: weights are still mmap'ed, not read. Warm the page
        # cache in the background now — ComfyUI encodes the prompt before
        # sampling loads the unet, so the read overlaps with that, and
        # parallel large reads run near the disk ceiling instead of the
        # page-fault rate of the loader (Flux2-dev: 0.47 -> ~1.2 GB/s).
        if prefetch:
            try:
                from qlip.inference.loom.prefetch import prefetch_module_weights

                h = prefetch_module_weights(patched.model.diffusion_model)
                if h is not None:
                    print(
                        f"[QlipCompile] prefetching weights in the "
                        f"background: {h.paths}"
                    )
            except Exception as exc:  # noqa: BLE001 — optimisation only
                print(f"[QlipCompile] weight prefetch skipped ({exc})")

        state = {
            "installed": False,
            "attached": False,
            "runs": 0,
            "regs": None,
            "frozen": False,
            "cold_msg": False,
            "installed_at_run": 0,
            "aot": False,
            "builder": None,
        }
        qc = dict(quant_config) if quant_config else None
        if qc is not None:
            quantize = "fp4" if qc.get("scheme") == "nvfp4" else "fp8"
            act_scales = (
                "dynamic"
                if qc.get("act_scales") == "dynamic"
                else "calibrate-first-run"
            )
        else:
            qc = {
                "weight_granularity": "per-tensor",
                "calib_runs": 1,
                "observer": "max",
                "ema_decay": 0.8,
                "min_dim": 512,
                "skip": (),
            }
        qc.setdefault("scheme", "nvfp4" if quantize == "fp4" else "fp8_e4m3")
        mode = "max-autotune-no-cudagraphs" if backend == "max-autotune" else "default"
        # fp8 per-tensor with calibrate-first-run: the calibration run is
        # EAGER and only the static graph is ever compiled. (Compiling the
        # calibration variant first and then recompiling after the freeze
        # paid two compiles, the first one thrown away.)
        calib = (
            quantize == "fp8"
            and qc["scheme"] != "nvfp4"
            and act_scales == "calibrate-first-run"
            and qc["weight_granularity"] == "per-tensor"
        )
        engines_dir = (engines_dir or "").strip()
        lora_rec = (
            lora_peft.lora_recipe(patched, lora_max_rank)
            if lora_mode == "swap"
            else None
        )
        recipe = None
        if engines_dir:
            from qlip.inference.loom import export as lx

            recipe = lx.make_recipe(
                lora=lora_rec,
                quant=None
                if quantize == "none"
                else {
                    "scheme": qc["scheme"],
                    "granularity": qc["weight_granularity"],
                    "min_dim": qc["min_dim"],
                    "skip": list(qc["skip"]),
                    "weights_only": False,
                },
                regions=None,
                extra={
                    "attention": attention,
                    "build": {"max_autotune": bool(aot_autotune)},
                },
            )
            ws_state = lx.workspace_status(engines_dir)
            if ws_state == "ready":
                have = lx.read_recipe(engines_dir) or {}
                same = all(
                    have.get(k) == recipe[k] for k in ("quant", "attention", "build")
                ) and bool(have.get("lora")) == bool(lora_rec)
                state["aot"] = same
                print(
                    f"[QlipCompile] engines_dir {engines_dir}: "
                    + (
                        "Inductor packages ready — loading them, no " "compilation"
                        if same
                        else "built for other quant settings — rebuilding"
                    )
                )
            else:
                print(
                    f"[QlipCompile] engines_dir {engines_dir}: {ws_state}"
                    f" — will build Inductor packages after warm-up"
                )

        def _resident():
            """Weights are final on the device: ComfyUI lists the model as
            loaded (sampling loads it BEFORE the first model call) and not
            as a partial lowvram load. Checked at the first model call —
            not at node time, when a fresh server has not loaded anything
            yet (that check used to turn every first run into a warm-up)."""
            try:
                import comfy.model_management as mm

                for lm in mm.loaded_models():
                    m = getattr(lm, "model", None)
                    if m is patched.model or getattr(m, "model", None) is patched.model:
                        return not getattr(patched.model, "model_lowvram", False)
            except Exception:  # noqa: BLE001
                pass
            return False

        def _attach(dm):
            rep = attach(
                dm, regions=state["regs"], mode=mode, mark_dynamic=(shapes == "dynamic")
            )
            state["attached"] = True
            print(
                f"[QlipCompile] attached "
                f"{[(r['name'], r['blocks']) for r in rep['regions']]}"
                f" — this run compiles as it goes; later runs and new "
                f"resolutions are fast (cache: "
                f"{ensure_persistent_compile_cache()})"
            )

        def _install(dm, device):
            _uninstall(dm)
            t0 = time.time()
            try:
                regs = find_regions(dm, min_repeat=2)
                if not regs:
                    print(
                        "[QlipCompile] no block stacks found — model " "left untouched"
                    )
                    return
                state["regs"] = regs
                if state["aot"]:
                    from qlip.inference.loom import export as lx

                    im = lx.load_workspace(dm, engines_dir, device=device)
                    _ACTIVE[id(dm)] = {
                        "dm": dm,
                        "memoized": [],
                        "aot": im,
                        "sig": sig,
                        "state": state,
                    }
                    if lora_mode == "swap":
                        lora_peft.swap(dm, lora_stack, patched, device)
                    state["frozen"] = True
                    print(
                        f"[QlipCompile] AOT: {im.n_blocks} blocks on "
                        f"Inductor packages, setup {time.time() - t0:.1f}s"
                    )
                    return
                # quantize EVERY block region. The historical JIT scope was
                # the largest region only, which left ComfyUI's quantized
                # ops (fp8mixed MixedPrecisionOps) in the other blocks —
                # opaque to Dynamo: graph breaks (Flux2 double blocks: 10-11),
                # so the resumed fragments recompiled on every resolution
                # change even with shapes="dynamic", and JIT/AOT numerics
                # differed. One scope for JIT and engines builds.
                qregs = regs
                qmods = [m for r in qregs for m in r["modules"]]
                if quantize in ("fp8", "fp4"):
                    nq = quantize_linears(
                        qmods,
                        device=device,
                        min_dim=qc["min_dim"],
                        granularity=qc["weight_granularity"],
                        skip=qc["skip"],
                        scheme=qc["scheme"],
                        release_each=(weights_policy == "release"),
                    )
                    state["calib_now"] = False
                    if calib:
                        # calibrated scales are cached per (weights, quant
                        # settings): any later process — restart, reboot —
                        # loads them and skips the calibration image
                        import os as _os

                        from qlip.inference.loom import export as lx

                        fp = lx.weights_fingerprint(
                            dm,
                            {
                                "qc": {k: v for k, v in qc.items() if k != "skip"},
                                "skip": list(qc["skip"]),
                                "calib_runs": qc["calib_runs"],
                            },
                        )
                        path = (
                            _os.path.join(lx.scale_cache_dir(), f"{fp}.safetensors")
                            if fp
                            else None
                        )
                        nl = lx.load_scales(dm, path) if path else 0
                        if nl:
                            state["frozen"] = True
                            print(
                                f"[QlipCompile] fp8 scales loaded from "
                                f"cache ({nl}), no calibration run"
                            )
                        else:
                            state["calib_now"] = True
                            state["scale_path"] = path
                            start_calibration(
                                qmods,
                                observer=qc["observer"],
                                ema_decay=qc["ema_decay"],
                            )
                    print(f"[QlipCompile] {qc['scheme']}: {nq} linears " f"on {device}")
                    if weights_policy == "release" and nq:
                        release_masters(qmods)
                        print(
                            "[QlipCompile] masters released "
                            "(interleaved) — quantized copies are now "
                            "the only weights; irreversible until "
                            "checkpoint reload"
                        )
                state["qmods"] = qmods
                if lora_mode == "swap":
                    n = lora_peft.install(dm, lora_rec, device)
                    lora_peft.swap(dm, lora_stack, patched, device)
                    print(
                        f"[QlipCompile] LoRA swap mode: side path on {n} "
                        f"linears (max rank {lora_max_rank}); later LoRA "
                        f"changes swap in place"
                    )
                if recipe is not None:
                    recipe["regions"] = [r["name"] for r in qregs]
                memoized = []
                for _, m in dm.named_modules():
                    if "EmbedND" in type(m).__name__:
                        memoize_by_shape(m)
                        memoized.append(m)
                _ACTIVE[id(dm)] = {
                    "dm": dm,
                    "memoized": memoized,
                    "sig": sig,
                    "state": state,
                }
                if state.get("calib_now"):
                    print(
                        f"[QlipCompile] fp8 calibration run (eager, "
                        f"{qc['calib_runs']} run(s)); the static graph "
                        f"compiles once, on the next run; scales are then "
                        f"cached for later processes"
                    )
                else:
                    _attach(dm)
                print(
                    f"[QlipCompile] install setup {time.time() - t0:.1f}s,"
                    f" rope memo x{len(memoized)}"
                )
            except Exception as exc:  # noqa: BLE001 — atomic: full rollback
                print(
                    f"[QlipCompile] install failed ({exc}) — rolled "
                    f"back, running the model untouched"
                )
                try:
                    detach(dm)
                    dequantize_linears([dm])
                except Exception:  # noqa: BLE001
                    pass
                # detach() strips the out-of-region dtype-cast guards, but the
                # force_resident conversion done at arming time is NOT rolled
                # back — so the untouched model would now crash on its fp32
                # first/last projections (mat1/mat2 dtype). Re-arm the guard.
                try:
                    ensure_untouched_linears_cast(dm, find_regions(dm, min_repeat=2))
                except Exception:  # noqa: BLE001
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
                if _resident():
                    state["installed"] = True
                    state["installed_at_run"] = state["runs"]
                    _install(patched.model.diffusion_model, args["input"].device)
                elif not state["cold_msg"]:
                    state["cold_msg"] = True
                    print(
                        "[QlipCompile] model not resident (lowvram / "
                        "streamed load) — this run is a plain warm-up; "
                        "installing on the next run"
                    )
            fn = apply_model
            if prev_wrapper is not None:

                def fn(x, t, **c):  # noqa: F811 — chained inner
                    return prev_wrapper(
                        apply_model,
                        {
                            "input": x,
                            "timestep": t,
                            "c": c,
                            "cond_or_uncond": args.get("cond_or_uncond", [0]),
                        },
                    )

            return fn(args["input"], args["timestep"], **args["c"])

        patched.set_model_unet_function_wrapper(unet_wrapper)

        # run-end hook: freeze calibrated scales after the calibration run(s),
        # attach the compiled path, build Inductor packages when asked, and
        # surface any per-block fallbacks
        def run_end_wrapper(executor, *args, **kwargs):
            b = state.get("builder")
            if (
                b is None
                and engines_dir
                and not state["aot"]
                and state["attached"]
                and state["frozen"]
                and state["regs"]
            ):
                # arm the capture for THIS run: one real call per block kind
                from qlip.inference.loom import export as lx

                try:
                    from qlip.compiler.inductor import InductorBuilderConfig

                    b = lx.WorkspaceBuilder(
                        patched.model.diffusion_model,
                        engines_dir,
                        regions=state["regs"],
                        recipe=recipe,
                        builder_config=InductorBuilderConfig(
                            max_autotune=bool(aot_autotune)
                        ),
                    ).arm()
                    state["builder"] = b
                except Exception as exc:  # noqa: BLE001
                    print(f"[QlipCompile] engines build not armed ({exc})")
                    state["builder"] = False
            out = executor(*args, **kwargs)
            state["runs"] += 1
            dm = patched.model.diffusion_model
            calib_done = state["runs"] - state["installed_at_run"] >= qc["calib_runs"]
            calib_now = state.get("calib_now", False)
            if (
                state["installed"]
                and not state["frozen"]
                and state["regs"]
                and (not calib_now or calib_done)
            ):
                state["frozen"] = True
                if calib_now:
                    nf = freeze_scales(state["qmods"])
                    msg = ""
                    if state.get("scale_path"):
                        from qlip.inference.loom import export as lx

                        try:
                            lx.save_scales(dm, state["scale_path"])
                            msg = f", cached in {state['scale_path']}"
                        except Exception as exc:  # noqa: BLE001
                            msg = f" (not cached: {exc})"
                    print(
                        f"[QlipCompile] calibration done "
                        f"({qc['calib_runs']} run(s)) — {nf} static act "
                        f"scales frozen{msg}"
                    )
                if not state["attached"]:
                    _attach(dm)
            if b and b.ready:
                state["builder"] = False
                t0 = time.time()
                try:
                    rep = b.build()
                    print(
                        f"[QlipCompile] Inductor packages built into "
                        f"{engines_dir} in {time.time() - t0:.1f}s — the "
                        f"next process loads them with no compilation"
                    )
                except Exception as exc:  # noqa: BLE001 — never break a run
                    print(
                        f"[QlipCompile] engines build failed ({exc}); "
                        f"JIT path unaffected"
                    )
            if state["regs"] and state["attached"]:
                failed = stats(dm)["failed"]
                if failed:
                    print(f"[QlipCompile] eager fallbacks: {failed}")
            return out

        state["wrappers"] = (unet_wrapper, run_end_wrapper)
        try:
            import comfy.patcher_extension as pe

            patched.add_wrapper(pe.WrappersMP.SAMPLER_SAMPLE, run_end_wrapper)
        except Exception:  # noqa: BLE001 — old ComfyUI: stay dynamic
            if calib:
                print(
                    "[QlipCompile] no sampler hook in this ComfyUI — "
                    "fp8 stays on dynamic scales"
                )
                calib = False

        print(
            f"[QlipCompile] armed: quantize={quantize} backend={backend} "
            f"act_scales={act_scales}"
            + (f" engines_dir={engines_dir}" if engines_dir else "")
            + (
                f" lora={lora_mode}x{len(lora_stack)}"
                if lora_stack or lora_mode == "swap"
                else ""
            )
            + ". Install happens on the first model call."
        )
        return (patched,)


class QlipQuantConfig:
    """Quantization settings for QlipCompile — qlip.quantization terms
    (scheme / granularity / observer), executed by the Loom fp8 path."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheme": (
                    ["fp8_e4m3", "nvfp4"],
                    {
                        "default": "fp8_e4m3",
                        "tooltip": "qlip QSchemeType. fp8_e4m3 = H100+. "
                        "nvfp4 = e2m1 + per-16 e4m3 block scales, "
                        "Blackwell sm_100+ tensor cores (falls back to "
                        "fp8 elsewhere); granularity/observer/calib "
                        "settings don't apply to nvfp4 (its scales are "
                        "per-block by construction).",
                    },
                ),
                "weight_granularity": (
                    ["per-tensor", "per-channel"],
                    {
                        "default": "per-tensor",
                        "tooltip": "qlip QuantGranularity. per-channel "
                        "= rowwise scaled GEMM: weight "
                        "scale per output channel, act "
                        "scale per token — more accurate, "
                        "slightly slower, no calibration "
                        "needed.",
                    },
                ),
                "act_scales": (
                    ["calibrate", "dynamic"],
                    {
                        "default": "calibrate",
                        "tooltip": "per-tensor only. calibrate: observer "
                        "collects amax during calib_runs sampling "
                        "runs, then scales freeze static (TRT-like "
                        "deploy). dynamic: per-call amax forever.",
                    },
                ),
                "calib_runs": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 20,
                        "tooltip": "How many sampling runs feed the "
                        "observer before scales freeze.",
                    },
                ),
                "observer": (
                    ["max", "ema"],
                    {
                        "default": "max",
                        "tooltip": "qlip StatMinMaxObserver flavors: max = "
                        "running maximum (safe), ema = exponential "
                        "moving average (ignores rare spikes).",
                    },
                ),
                "ema_decay": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.1,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "EMA observer: scale = decay*new + "
                        "(1-decay)*current (qlip default 0.8).",
                    },
                ),
                "min_dim": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "tooltip": "Linears with any side smaller than "
                        "this stay unquantized (tiny projections gain "
                        "nothing, lose accuracy).",
                    },
                ),
                "skip_layers": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated name substrings to keep "
                        "in high precision, e.g. 'qkv' or "
                        "'feed_forward.w2'.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("QLIP_QUANT",)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "make"
    CATEGORY = "qlip"

    def make(
        self,
        scheme,
        weight_granularity,
        act_scales,
        calib_runs,
        observer,
        ema_decay,
        min_dim,
        skip_layers,
    ):
        cfg = {
            "scheme": scheme,
            "weight_granularity": weight_granularity,
            "act_scales": act_scales,
            "calib_runs": int(calib_runs),
            "observer": observer,
            "ema_decay": float(ema_decay),
            "min_dim": int(min_dim),
            "skip": tuple(x.strip() for x in skip_layers.split(",") if x.strip()),
        }
        print(f"[QlipQuantConfig] {cfg}")
        return (cfg,)

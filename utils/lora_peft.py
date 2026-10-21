"""PEFT-style LoRA for compiled models (QlipCompile / Inductor packages).

Two ways to use a LoRA with qlip torch compile:

  merge  ComfyUI's own mechanism: the LoRA is added into the weights
         (comfy.sd.load_lora_for_models). No per-step cost; a change costs a
         weight re-patch (~8-9 s on Flux2-dev) and, with fp8, a re-calibration
         image.
  swap   a fixed-shape side path  y = W x + B (A x)  on every eligible
         Linear (qlip.inference.loom.lora). A change is an in-place copy into
         A/B (~1 s), never a recompile or a package rebuild; the side path
         costs ~+19% per step at rank 256.

This module is the ComfyUI glue for `swap`: ComfyUI's LoRA key map (so
diffusers-named adapters land on fused ComfyUI layers, e.g. Flux
`img_attn.qkv` row slices), per-module side-path ranks, and stack I/O.
"""
import os

from qlip.inference.loom import lora as ll

MIN_DIM = 16  # also the small projections LoRAs target (time/guidance in)


def lora_mode_help():
    return (
        "merge: the LoRA is merged into the weights by ComfyUI (no per-step "
        "cost; a change re-patches weights, ~10 s). swap: PEFT side path — a "
        "change is an in-place swap (~1 s, no recompile, no package rebuild), "
        "~+19% step cost. Do not ALSO load the same LoRA with LoraLoader."
    )


def comfy_key_map(model_patcher):
    import comfy.lora

    return ll.key_map_from_comfy(
        comfy.lora.model_lora_keys_unet(model_patcher.model, {})
    )


def lora_recipe(model_patcher, max_rank):
    """install_lora kwargs, also stored in the Inductor workspace recipe."""
    km = comfy_key_map(model_patcher)
    return {
        "max_rank": int(max_rank),
        "min_dim": MIN_DIM,
        "rank_for": ll.ranks_from_key_map(km, int(max_rank)),
    }


def installed(dm):
    return bool(ll.lora_modules([dm]))


def install(dm, recipe, device):
    if installed(dm):
        return 0
    return ll.install_lora(
        [dm],
        max_rank=recipe["max_rank"],
        device=device,
        min_dim=recipe["min_dim"],
        rank_for=recipe.get("rank_for"),
    )


def stack_key(stack):
    out = []
    for e in stack or []:
        p = e.get("path", "")
        try:
            mt = os.path.getmtime(p) if p else 0.0
        except OSError:
            mt = 0.0
        out.append((str(p), float(e.get("strength", 1.0)), mt))
    return tuple(out)


def swap(dm, stack, model_patcher, device=None):
    """Make `stack` the active adapter set (empty/None -> side path off)."""
    if not stack:
        n = ll.disable_lora([dm])
        return {"disabled": n}
    import comfy.utils

    entries = []
    for e in stack:
        sd = comfy.utils.load_torch_file(e["path"], safe_load=True)
        entries.append(
            {
                "tensors": sd,
                "strength": float(e.get("strength", 1.0)),
                "path": e["path"],
            }
        )
    rep = ll.swap_lora(
        [dm], entries, device=device, key_map=comfy_key_map(model_patcher)
    )
    if rep.get("unmatched_count"):
        print(
            f"[qlip] LoRA swap: {rep['unmatched_count']} adapter layer(s) "
            f"matched nothing, e.g. {rep['unmatched'][:3]}"
        )
    return rep


def merge(model_patcher, stack):
    """`merge` mode: ComfyUI's own LoRA patching, returns the patched model."""
    if not stack:
        return model_patcher
    import comfy.sd
    import comfy.utils

    m = model_patcher
    for e in stack:
        sd = comfy.utils.load_torch_file(e["path"], safe_load=True)
        m = comfy.sd.load_lora_for_models(
            m, None, sd, float(e.get("strength", 1.0)), 0
        )[0]
    return m

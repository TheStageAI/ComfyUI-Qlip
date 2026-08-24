"""Qlip Drafter — self-speculative sparse-attention draft on late denoising steps.

Early steps run full attention (they set structure); from ``split_step`` onward the
model's self-attention is DRAFTED with the licensed block-sparse kernel. One model,
no separate drafter, no training. Orthogonal to QlipProgressive (resolution) — stack
both. Licensed through the standard qlip session (same gate as QlipAutoSparse).

Keep your own sampler: this is a model hook (counts steps via the unet wrapper) plus
an attention rebind, like QlipAutoSparse.
"""

from .engine_loader import _validate_diffusion_model_input
from ..utils.helpers import _rebind_optimized_attention

_ORIG = None
_PATCHED = []
_MOD = None          # active SelfSpeculativeDrafter (None = disabled)


def _install(mod, orig_fn):
    global _PATCHED, _MOD
    _MOD = mod
    err_shown = [False]

    def routed(q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
        m = _MOD
        if (m is None or not m.drafting or skip_output_reshape
                or not m.matches(q, k, mask, skip_reshape)):
            if m is not None:
                m.n_full += 1
            return orig_fn(q, k, v, heads, mask=mask,
                           attn_precision=attn_precision,
                           skip_reshape=skip_reshape, **kw)
        try:
            return m.draft_attention(q, k, v, heads)
        except Exception as e:
            from qlip.inference.errors import LicensingError
            if isinstance(e, LicensingError):
                raise
            if not err_shown[0]:
                err_shown[0] = True
                import traceback
                print(f"[QlipDrafter] draft kernel failed on q={tuple(q.shape)} "
                      f"→ falling back to DENSE. First error:\n{e}")
                traceback.print_exc()
            m.n_full += 1
            return orig_fn(q, k, v, heads, mask=mask,
                           attn_precision=attn_precision,
                           skip_reshape=skip_reshape, **kw)

    _PATCHED = _rebind_optimized_attention(
        orig_fn, routed, extra_symbols=["optimized_attention_masked"])
    import comfy.ldm.modules.attention as A
    A.optimized_attention = routed


def _uninstall():
    global _MOD, _PATCHED
    _MOD = None
    if _ORIG is not None:
        import comfy.ldm.modules.attention as A
        A.optimized_attention = _ORIG
        for mod, sym, old in _PATCHED:
            try:
                mod.__dict__[sym] = old
            except Exception:
                pass
    _PATCHED = []


class QlipDrafter:
    """Self-speculative sparse-attention draft on late steps, licensed via qlip."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Any DiT; keep your own sampler "
                          "— this is a model hook, not a sampler."}),
                "enable": ("BOOLEAN", {"default": True}),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 1000,
                                "tooltip": "Number of sampler steps you run "
                                "(needed to place the split correctly)."}),
                "split_step": ("INT", {"default": 4, "min": 1, "max": 1000,
                               "tooltip": "Steps BEFORE this run full attention "
                               "(they set structure); steps FROM this on are "
                               "sparse-drafted. Higher = safer/less speedup, "
                               "lower = faster/more drift. 4 of 8 is a good "
                               "start."}),
            },
            "optional": {
                "sparsity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.95,
                             "step": 0.05, "tooltip": "Draft attention sparsity "
                             "on late steps (fraction of blocks dropped). 0.9 = "
                             "aggressive draft."}),
                "selector": (["topk", "diversity", "meansim"],
                             {"default": "topk", "tooltip": "Block selector for "
                             "the draft kernel."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, total_steps=8, split_step=4,
              sparsity=0.9, selector="topk"):
        _validate_diffusion_model_input(model, "QlipDrafter")
        global _ORIG

        if not enable:
            _uninstall()
            return (model,)

        try:
            from qlip.inference.drafter_core import (
                SelfSpeculativeDrafter, DrafterConfig)
        except ImportError as e:
            raise RuntimeError(
                "QlipDrafter requires the qlip package (the draft kernel runs "
                "through the licensed qlip session). Install qlip and log in "
                "with your TheStage token.") from e

        patched = model.clone()
        import comfy.ldm.modules.attention as A
        if _ORIG is None:
            _ORIG = A.optimized_attention

        mod = SelfSpeculativeDrafter(DrafterConfig(
            split_step=int(split_step), sparsity=float(sparsity),
            selector=selector))
        mod.reset()
        patched.model._qlip_drafter = mod
        _install(mod, _ORIG)

        # count denoising steps via the unet wrapper; flip drafting on at split
        state = {"step": 0}
        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            # a full model call = one denoising step (per lane). Use the first
            # lane's calls to advance; drafting turns on once split is reached.
            mod.drafting = (state["step"] >= int(split_step))
            state["step"] += 1
            if prev_wrapper is not None:
                return prev_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        patched.set_model_unet_function_wrapper(unet_wrapper)
        print(f"[QlipDrafter] enabled: full steps 0..{split_step-1}, "
              f"sparse-draft steps {split_step}..{total_steps-1} "
              f"(sparsity={sparsity}, selector={selector}), licensed qlip "
              f"session. Orthogonal to QlipProgressive — stack for more.")
        return (patched,)

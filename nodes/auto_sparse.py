"""Qlip Auto Sparse — one-knob block-sparse attention on any model.

Thin router: the actual int8 dynamic block-sparse attention (selectors +
Triton kernel) lives in the qlip package and runs through the standard qlip
licensed session — the same paid-session gate as the compiled engines, so
one login covers both. This node swaps ComfyUI's ``optimized_attention`` for
a router that sends long self-attention calls to
``qlip.inference.sparse_attention`` and passes everything else (text/cross
attention, masked calls, short sequences) to the original kernel untouched.
"""

from .engine_loader import _validate_diffusion_model_input
from ..utils.helpers import _rebind_optimized_attention

_ORIG = None
_PATCHED = []
_MOD = None          # active QlipAutoSparseAttention (None = disabled)


def _auto_report(mod):
    total = mod.n_sparse + mod.n_dense
    if total and total % 96 == 0:
        tag = ("" if mod.n_sparse else
               "  <-- ALL DENSE: kernel never engaged (see error above, "
               "masked/cross-attn calls, or sequences shorter than min_seq)")
        print(f"[QlipAutoSparse] sparse={mod.n_sparse} dense={mod.n_dense}{tag}")


def _install(mod, orig_fn):
    global _PATCHED, _MOD
    _MOD = mod
    err_shown = [False]

    def routed(q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
        m = _MOD
        if (m is None or skip_output_reshape
                or not m.matches(q, k, mask, skip_reshape)):
            if m is not None:
                m.n_dense += 1
                _auto_report(m)
            return orig_fn(q, k, v, heads, mask=mask,
                           attn_precision=attn_precision,
                           skip_reshape=skip_reshape, **kw)
        try:
            out = m.attention(q, k, v, heads)
            _auto_report(m)
            return out
        except Exception as e:
            from qlip.inference.errors import LicensingError
            if isinstance(e, LicensingError):
                raise
            if not err_shown[0]:
                err_shown[0] = True
                import traceback
                print(f"[QlipAutoSparse] kernel failed on q={tuple(q.shape)} "
                      f"heads={heads} → falling back to DENSE. First error:\n"
                      f"{e}")
                traceback.print_exc()
            m.n_dense += 1
            _auto_report(m)
            return orig_fn(q, k, v, heads, mask=mask,
                           attn_precision=attn_precision,
                           skip_reshape=skip_reshape, **kw)

    # also intercept optimized_attention_masked — Krea-2 (comfy.ldm.krea2.model)
    # imports it by name and never touches the plain optimized_attention, so
    # without this the sparse kernel stays all-dense on Krea-2.
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
        # _PATCHED is now a list of (module, symbol, old_fn) tuples
        for mod, sym, old in _PATCHED:
            try:
                mod.__dict__[sym] = old
            except Exception:
                pass
    _PATCHED = []


class QlipAutoSparse:
    """One-knob block-sparse attention, licensed through qlip."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Any DiT (LTX, Wan, etc.)."}),
                "enable": ("BOOLEAN", {"default": True}),
                "sparsity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95,
                             "step": 0.05, "tooltip": "Fraction of attention "
                             "blocks to DROP. 0.5 keeps the top 50% (safe); "
                             "0.7 is the validated fast point with the "
                             "diversity selector; 0.9 is a draft mode. Blocks "
                             "are picked dynamically per step."}),
            },
            "optional": {
                "selector": (["diversity", "topk", "meansim", "tau"],
                             {"default": "diversity",
                             "tooltip": "How blocks are picked. diversity "
                             "(DEFAULT, recommended) = similarity minus "
                             "redundancy, keeps distinct blocks; best DOVER "
                             "quality, beats top-k and even dense at the same "
                             "speed. topk = flat top-k by similarity. meansim "
                             "= cdf mass + self-similarity. tau = Sol-Attn "
                             "DYNAMIC budget: keep blocks above mean+tau*std of "
                             "each query block's scores (ignores `sparsity`; use "
                             "the `tau` knob) — budget adapts per query block."}),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,
                        "step": 0.1, "tooltip": "selector=tau only: threshold in "
                        "#-of-sigmas above each query block's mean score. Larger "
                        "= fewer blocks kept = faster/lower quality. 1.0 default "
                        "(Sol-Attn)."}),
                "correction": (["none", "sla", "vspace_block", "vspace_topk"],
                               {"default": "none",
                               "tooltip": "Attention mode. none = block-sparse "
                               "(fast, skips blocks). sla = ORIGINAL SLA: fast "
                               "LUT block-sparse kernel + linear-attention "
                               "compensator (the real fast+quality path — "
                               "recommended to test). vspace_block = block "
                               "kernel + V-space rank-d correction (moderate "
                               "sparsity). vspace_topk = per-token top-k + "
                               "V-space — best QUALITY but PURE PYTORCH (slow, "
                               "quality-probe only)."}),
                "d_rank": ("INT", {"default": 24, "min": 4, "max": 128,
                           "step": 4, "tooltip": "Rank of the V-space "
                           "correction subspace (both vspace_* modes). 24 "
                           "captured ~98% of the residual on Krea-2."}),
                "score_rank": ("INT", {"default": 32, "min": 8, "max": 128,
                              "step": 8, "tooltip": "vspace_topk only: rank of "
                              "the cheap SVD key-scorer. 32 ≈ exact top-k on "
                              "Krea-2; 16 drifts."}),
                "simthreshd1": ("FLOAT", {"default": 0.1, "min": -0.5, "max": 1.0,
                                "step": 0.05, "tooltip": "meansim only: "
                                "self-similarity threshold. Higher = fewer "
                                "blocks force-kept = more sparsity."}),
                "smooth_k": ("BOOLEAN", {"default": True, "tooltip":
                             "Key smoothing (de-mean) — usually improves "
                             "quality at no cost."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, sparsity=0.5, selector='diversity',
              correction='none', d_rank=24, score_rank=32,
              simthreshd1=0.1, smooth_k=True, tau=1.0):
        _validate_diffusion_model_input(model, "QlipAutoSparse")
        global _ORIG

        if not enable:
            _uninstall()
            return (model,)

        try:
            from qlip.inference.sparse_attention import (
                QlipAutoSparseAttention, SparseAttentionConfig)
        except ImportError as e:
            raise RuntimeError(
                "QlipAutoSparse requires the qlip package (the sparse kernel "
                "runs through the licensed qlip session). Install qlip and "
                "log in with your TheStage token.") from e

        patched = model.clone()
        import comfy.ldm.modules.attention as A
        if _ORIG is None:
            _ORIG = A.optimized_attention

        mod = QlipAutoSparseAttention(SparseAttentionConfig(
            sparsity=float(sparsity), selector=selector,
            correction=correction, d_rank=int(d_rank),
            score_rank=int(score_rank), tau=float(tau),
            simthreshd1=float(simthreshd1), smooth_k=smooth_k))

        patched.model._qlip_auto_sparse = mod
        _install(mod, _ORIG)
        corr = ("" if correction == "none"
                else f", correction={correction} (d_rank={d_rank}"
                     + (f", score_rank={score_rank}" if correction ==
                        "vspace_topk" else "") + ")")
        warn = ("  [vspace_topk is PURE PYTORCH — quality probe, not a speed "
                "win; no fused kernel yet]" if correction == "vspace_topk"
                else "")
        print(f"[QlipAutoSparse] enabled: sparsity={sparsity} "
              f"(keep ~{1.0-sparsity:.0%} of blocks, selector={selector}"
              f"{corr}), licensed qlip session. Sparsifies self-attn with "
              f"seq>={mod.cfg.min_seq}.{warn}")
        return (patched,)

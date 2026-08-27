"""Qlip Token Prune — drop low-salience tokens on mid denoising steps.

Sixth acceleration axis (licensed), orthogonal to the others: cuts the NUMBER OF
TOKENS the transformer block stack processes on a window of mid steps, restoring the
dropped tokens from the previous step's hidden state (Sol-Engine feature-norm
pruning). Best on long spatiotemporal sequences (VIDEO) where many refine-step tokens
are redundant; on short image sequences the Amdahl win is small. Stacks with
QlipAutoSparse/Drafter (those cut attention interactions; this cuts token count).

How it hooks: it wraps the diffusion model's transformer-block loop so the expensive
blocks run on the kept-token subset. This needs the model to expose a discoverable
`transformer_blocks` (or `blocks`) ModuleList — true for most DiTs (LTX, Flux, Krea).
If none is found the node prints a warning and stays a no-op (never breaks the run).
"""

from .engine_loader import _validate_diffusion_model_input

# Registry of live block-forward patches so we can cleanly REMOVE them when the
# node is disabled (or re-armed). Without this, disabling QlipTokenPrune left the
# first/last block monkey-patched AND a stale process-wide _PREV_HIDDEN buffer,
# so pruning damage (accumulating noise via compensation="prev") survived the
# toggle. Keyed by id(block) -> (block, original_forward).
_PATCHED_BLOCKS = {}


def _restore_all_patches():
    """Remove every live token-prune block-forward patch and clear the qlip
    process-wide prev-hidden buffer. Safe to call anytime (idempotent)."""
    for _bid, (blk, orig) in list(_PATCHED_BLOCKS.items()):
        try:
            blk.forward = orig
        except Exception:
            pass
    _PATCHED_BLOCKS.clear()
    try:
        import qlip.inference.token_prune_core as tp
        tp._PREV_HIDDEN.clear()
    except Exception:
        pass


def _find_block_list(dm):
    """Return the transformer-block ModuleList of a diffusion model, or None."""
    import torch
    for attr in ("transformer_blocks", "blocks", "double_blocks", "layers"):
        bl = getattr(dm, attr, None)
        if isinstance(bl, torch.nn.ModuleList) and len(bl) >= 2:
            return attr, bl
    # search one level into a `.transformer` / `.model` submodule
    for sub in ("transformer", "model", "diffusion_model"):
        m = getattr(dm, sub, None)
        if m is not None:
            for attr in ("transformer_blocks", "blocks", "layers"):
                bl = getattr(m, attr, None)
                if isinstance(bl, torch.nn.ModuleList) and len(bl) >= 2:
                    return f"{sub}.{attr}", bl
    return None, None


class QlipTokenPrune:
    """One knob: keep_ratio. Prunes tokens on a mid-step window; video lever."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "keep_ratio": ("FLOAT", {"default": 0.75, "min": 0.3, "max": 1.0,
                               "step": 0.05, "tooltip": "Fraction of tokens KEPT "
                               "on a pruned step. 0.75 = drop the least-salient "
                               "25%. 1.0 = off. Lower = faster, more quality risk."}),
                "method": (["l2sq", "l1", "linf", "var"], {"default": "l2sq",
                           "tooltip": "Saliency criterion per token. l2sq "
                           "(Sol-Engine default) = sum of squared features."}),
                "compensation": (["prev", "zero"], {"default": "prev",
                                 "tooltip": "How dropped tokens are filled: prev "
                                 "= previous step's hidden state (recommended); "
                                 "zero = zeros."}),
                "step_lo": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0,
                            "step": 0.05, "tooltip": "Prune only while the step "
                            "fraction is >= this (0=first step, 1=last). Keeps "
                            "early structure steps full."}),
                "step_hi": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0,
                            "step": 0.05, "tooltip": "Prune only while the step "
                            "fraction is <= this. Keeps late detail steps full."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, keep_ratio=0.75, method="l2sq",
              compensation="prev", step_lo=0.2, step_hi=0.8):
        _validate_diffusion_model_input(model, "QlipTokenPrune")
        # ALWAYS undo any previous token-prune patch + clear the process-wide prev
        # buffer before doing anything — this makes disable actually disable, and
        # re-arm start clean (no accumulated-noise carryover between runs).
        _restore_all_patches()
        patched = model.clone()
        if not enable or keep_ratio >= 0.999:
            print("[QlipTokenPrune] disabled — patches removed, prev buffer cleared.")
            return (patched,)

        try:
            from qlip.inference.token_prune_core import (
                TokenPruner, TokenPruneConfig)
        except ImportError as e:
            raise RuntimeError(
                "QlipTokenPrune requires the qlip package (licensed). Install "
                "qlip and log in with your TheStage token.") from e

        dm = patched.model.diffusion_model
        attr, block_list = _find_block_list(dm)
        if block_list is None:
            print("[QlipTokenPrune] no transformer-block ModuleList found on this "
                  "model — token pruning is a no-op here (works on standard DiTs).")
            return (patched,)

        try:
            model_tag = f"{type(dm).__name__}:{sum(1 for _ in dm.parameters())}"
        except Exception:
            model_tag = None
        pruner = TokenPruner(TokenPruneConfig(
            keep_ratio=float(keep_ratio), method=method,
            compensation=compensation, step_lo=float(step_lo),
            step_hi=float(step_hi)), model_tag=model_tag)
        pruner.reset()

        # step-fraction clock: the model wrapper sees each denoising step; we map
        # sigma -> fraction of the run (1 at first step, ~0 at last) so the prune
        # window [step_lo, step_hi] is well defined without knowing total steps.
        lane = {"first_sigma": None, "frac": 1.0}
        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            try:
                sig = float(args["timestep"].reshape(-1)[0])
                if lane["first_sigma"] is None or sig > lane["first_sigma"]:
                    lane["first_sigma"] = sig
                # frac = 1 - sigma/first  → 0 early (high sigma), 1 late
                lane["frac"] = 1.0 - sig / max(lane["first_sigma"], 1e-8)
            except Exception:
                pass
            if prev_wrapper is not None:
                return prev_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])
        patched.set_model_unet_function_wrapper(unet_wrapper)

        # ROBUST block-loop hook: patch the FIRST block to gather kept tokens and
        # the LAST block to scatter them back. Works for any `for b in blocks:
        # x = b(x, ...)` loop (LTX/Flux/Krea) — no model-specific loop patch.
        # The kept-index is chosen once per step (on the first block) from the
        # first positional tensor arg; every block then runs on the reduced set;
        # the last block restores the full token count. Non-tensor / mismatched
        # shapes fall through untouched (safe no-op for that call).
        import torch
        first_blk, last_blk = block_list[0], block_list[-1]
        of_first = first_blk.forward
        of_last = last_blk.forward
        st = {"idx": None, "full_shape": None}

        def _hidden_arg(args):
            for i, a in enumerate(args):
                if torch.is_tensor(a) and a.dim() == 3:
                    return i, a
            return None, None

        def first_forward(*args, **kw):
            st["idx"] = None
            i, x = _hidden_arg(args)
            if x is not None and pruner._should_prune(lane["frac"], x.shape[1]):
                keep = max(1, int(round(x.shape[1] * pruner.cfg.keep_ratio)))
                from qlip.inference.token_prune_core import feature_saliency
                sal = feature_saliency(x, pruner.cfg.method).mean(0)
                idx = torch.sort(torch.topk(sal, keep).indices).values
                st["idx"], st["full_shape"] = idx, x.shape
                args = list(args)
                args[i] = x.index_select(1, idx)
                args = tuple(args)
            return of_first(*args, **kw)

        def last_forward(*args, **kw):
            out = of_last(*args, **kw)
            idx = st["idx"]
            if idx is not None and torch.is_tensor(out) and out.dim() == 3 \
                    and out.shape[1] == idx.shape[0]:
                import qlip.inference.token_prune_core as tp
                prev = tp._PREV_HIDDEN.get(pruner.model_tag)
                B, _, C = out.shape
                N = st["full_shape"][1]
                if pruner.cfg.compensation == "prev" and prev is not None \
                        and prev.shape == (B, N, C):
                    full = prev.clone()
                else:
                    full = out.new_zeros(B, N, C)
                full.index_copy_(1, idx, out.to(full.dtype))
                tp._PREV_HIDDEN[pruner.model_tag] = full.detach()
                pruner.n_pruned += 1
                st["idx"] = None
                return full
            # full pass (no prune this step): refresh prev buffer
            if torch.is_tensor(out) and out.dim() == 3:
                import qlip.inference.token_prune_core as tp
                tp._PREV_HIDDEN[pruner.model_tag] = out.detach()
            return out

        first_blk.forward = first_forward
        last_blk.forward = last_forward
        # register so a later disable / re-arm can restore the originals
        _PATCHED_BLOCKS[id(first_blk)] = (first_blk, of_first)
        _PATCHED_BLOCKS[id(last_blk)] = (last_blk, of_last)
        patched.model._qlip_token_pruner = pruner

        print(f"[QlipTokenPrune] armed on {attr} ({len(block_list)} blocks): "
              f"keep {keep_ratio:.0%} on step window [{step_lo:.2f},{step_hi:.2f}], "
              f"method={method}, comp={compensation}. Licensed qlip session. "
              f"Video lever — orthogonal to sparse/progressive.")
        return (patched,)


NODE_CLASS_MAPPINGS = {"QlipTokenPrune": QlipTokenPrune}
NODE_DISPLAY_NAME_MAPPINGS = {"QlipTokenPrune": "Qlip Token Prune"}

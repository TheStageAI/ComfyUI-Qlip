"""Route ComfyUI's optimized_attention through qlip.inference.qattn
(dependency-free SageAttention-class kernels), plus the loom polish for the
external `--use-sage-attention` path (mark it opaque for Dynamo).

Mirrors comfy.ldm.modules.attention.attention_pytorch's reshape contract:
  skip_reshape=False : q,k,v are [B, S, H*D]   -> out [B, S, H*D]
  skip_reshape=True  : q,k,v are [B, H, S, D]  -> out [B, S, H*D]
                       (or [B, H, S, D] if skip_output_reshape)
Masked / short / odd-head-dim calls go to the original function untouched.
"""
import torch

from .helpers import _rebind_optimized_attention

_STATE = {"patched": [], "orig": None, "cfg": None, "n_q": 0, "n_dense": 0}


def polish_external_sage():
    """If ComfyUI runs with --use-sage-attention, wrap attention_sage with
    torch.compiler.disable ONCE so loom's Dynamo never tries to trace into
    sageattn's Python asserts (seen as trace-time AssertionErrors)."""
    try:
        import comfy.ldm.modules.attention as A
    except Exception:            # noqa: BLE001
        return False
    sage = getattr(A, "attention_sage", None)
    if sage is None or getattr(sage, "_qlip_opaque", False):
        return False
    if A.optimized_attention is not sage:
        return False
    wrapped = torch.compiler.disable(sage)
    wrapped._qlip_opaque = True          # noqa: SLF001
    _rebind_optimized_attention(sage, wrapped)
    A.attention_sage = wrapped
    print("[QlipCompile] external sage attention marked opaque for torch.compile")
    return True


def uninstall_attention():
    for mod, sym, old in _STATE["patched"]:
        try:
            setattr(mod, sym, old)
        except Exception:        # noqa: BLE001
            pass
    if _STATE["orig"] is not None:
        try:
            import comfy.ldm.modules.attention as A
            A.optimized_attention = _STATE["orig"]
        except Exception:        # noqa: BLE001
            pass
    _STATE.update(patched=[], orig=None, cfg=None)


def install_attention(mode, min_seq=2048):
    """mode: auto | int8_fp8 | int8_fp16 | fp4 (see qlip.inference.qattn)."""
    from qlip.inference.qattn import QAttnConfig, describe_backend, qlip_attention
    import comfy.ldm.modules.attention as A

    uninstall_attention()
    cfg = QAttnConfig(mode=mode, min_seq=int(min_seq))
    orig = A.optimized_attention
    _STATE.update(orig=orig, cfg=cfg, n_q=0, n_dense=0)

    def routed(q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
        if mask is not None:
            _STATE["n_dense"] += 1
            return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kw)
        if skip_reshape:
            b, h, s, d = q.shape
            qh, kh, vh = q, k, v
        else:
            b, s, hd = q.shape
            d = hd // heads
            qh, kh, vh = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
        if (s < cfg.min_seq or kh.shape[2] < cfg.min_seq or d not in cfg.head_dims
                or qh.dtype not in (torch.bfloat16, torch.float16)):
            _STATE["n_dense"] += 1
            return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kw)
        _STATE["n_q"] += 1
        out = qlip_attention(qh, kh, vh, cfg)                    # [B,H,S,D]
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(b, -1, heads * d)

    # Opaque for Dynamo: loom compiles the block that CALLS this; if Dynamo
    # inlines `routed` it guards on the mutable _STATE counters and recompiles
    # on every call (hit recompile_limit=64 -> whole block falls to eager).
    routed = torch.compiler.disable(routed)
    _STATE["patched"] = _rebind_optimized_attention(
        orig, routed, extra_symbols=["optimized_attention_masked"])
    A.optimized_attention = routed
    print(f"[QlipCompile] attention -> {describe_backend(cfg)}")
    return cfg


def attention_stats():
    return {"quantized_calls": _STATE["n_q"], "dense_calls": _STATE["n_dense"],
            "mode": (_STATE["cfg"].mode if _STATE["cfg"] else "comfy")}

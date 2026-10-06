"""Host-side node state that decides re-execution and VRAM — regressions found
on Krea 2 / H200 (2026-10-05)."""

import types


def test_auto_sparse_is_changed_is_stable(pack_module):
    """Always-NaN IS_CHANGED re-ran AutoSparse and everything downstream
    (QlipCompile) on every prompt: ~26 s/image instead of ~3 s."""
    m = pack_module("nodes.auto_sparse")
    cls = m.QlipAutoSparse
    old = (m._MOD, m._SIG, m._LAST_SIG)
    try:
        m._MOD, m._SIG = None, None
        kw = {"enable": True, "sparsity": 0.5, "selector": "diversity"}
        first = cls.IS_CHANGED(**kw)
        assert first == first, "first call must not be NaN (it costs a re-run)"
        # node executed: patch installed with these settings
        m._MOD, m._SIG = object(), m._LAST_SIG
        assert cls.IS_CHANGED(**kw) == first, "unchanged settings -> cached"
        changed = cls.IS_CHANGED(**dict(kw, sparsity=0.7))
        assert changed != first, "new settings -> re-run"
        # patch removed elsewhere (enable=False node): must re-run to restore
        m._MOD, m._SIG = None, m._SIG
        removed = cls.IS_CHANGED(**kw)
        assert removed != removed, "removed patch -> NaN"
    finally:
        m._MOD, m._SIG, m._LAST_SIG = old


class _FakePatcher:
    """Minimal ComfyUI ModelPatcher: clones share ``model``; a non-dynamic
    clone gets a resident copy unless ``model_override`` is given."""

    copies = 0

    def __init__(self, model, override=None):
        self.model = model
        self.override = override

    def clone(self, disable_dynamic=False, model_override=None):
        if model_override is None:
            _FakePatcher.copies += 1
            model_override = types.SimpleNamespace(copy_of=self.model)
        return _FakePatcher(self.model, model_override)

    def get_clone_model_override(self):
        return self.override


def test_compile_reuses_one_resident_copy(pack_module):
    """Each re-execution (LoRA change) used to make a new resident copy of the
    whole model: +36 GB per change on Krea 2."""
    compile_mod = pack_module("nodes.compile")
    base = object()
    _FakePatcher.copies = 0
    first = compile_mod._resident_delegate(_FakePatcher(base))
    second = compile_mod._resident_delegate(_FakePatcher(base))  # new clone, same base
    assert _FakePatcher.copies == 1
    assert second.override is first.override
    other = compile_mod._resident_delegate(_FakePatcher(object()))  # another checkpoint
    assert _FakePatcher.copies == 2 and other.override is not first.override

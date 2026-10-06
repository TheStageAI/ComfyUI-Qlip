"""Node registration and schemas — what ComfyUI reads when it loads the pack."""

import pytest

EXPECTED_NODES = {
    "QlipEnginesLoader",
    "QlipLoraStack",
    "QlipLoraSwitch",
    "QlipTimerStart",
    "QlipTimerStop",
    "QlipTimerReport",
    "QlipCache",
    "QlipCacheReport",
    "QlipAutoSparse",
    "QlipProgressive",
    "QlipSpectrumFit",
    "QlipCompile",
    "QlipQuantConfig",
    "QlipDrafter",
    "QlipTokenPrune",
}
REMOVED_NODES = {"QlipAutoPilot", "QlipSchedule", "QlipRestartSampler"}


def test_expected_nodes_registered(pack):
    assert set(pack.NODE_CLASS_MAPPINGS) == EXPECTED_NODES
    assert not REMOVED_NODES & set(pack.NODE_CLASS_MAPPINGS)


def test_every_node_has_a_display_name(pack):
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(pack.NODE_CLASS_MAPPINGS)
    for name in pack.NODE_DISPLAY_NAME_MAPPINGS.values():
        assert isinstance(name, str) and name


def _input_specs(cls):
    spec = cls.INPUT_TYPES()
    # "required" may be absent (e.g. QlipCacheReport has only an optional trigger)
    assert isinstance(spec, dict) and set(spec) <= {
        "required",
        "optional",
        "hidden",
    }, cls.__name__
    for section in ("required", "optional"):
        for inp, value in (spec.get(section) or {}).items():
            yield section, inp, value


@pytest.mark.parametrize("node", sorted(EXPECTED_NODES))
def test_node_schema(pack, node):
    cls = pack.NODE_CLASS_MAPPINGS[node]
    for section, inp, value in _input_specs(cls):
        assert isinstance(value, tuple) and value, f"{node}.{section}.{inp}"
        kind = value[0]
        # a type name ("MODEL", "INT", ...) or a list of choices (combo box)
        assert isinstance(kind, (str, list)), f"{node}.{inp}: {kind!r}"
        if isinstance(kind, list):
            assert kind, f"{node}.{inp}: empty choice list"
            opts = value[1] if len(value) > 1 else {}
            if "default" in opts:
                assert opts["default"] in kind, f"{node}.{inp}: default not a choice"
        elif len(value) > 1 and isinstance(value[1], dict):
            opts = value[1]
            lo, hi, default = opts.get("min"), opts.get("max"), opts.get("default")
            if None not in (lo, hi, default):
                assert lo <= default <= hi, f"{node}.{inp}: default out of range"

    assert isinstance(cls.RETURN_TYPES, tuple), node
    names = getattr(cls, "RETURN_NAMES", None)
    if names is not None:
        assert len(names) == len(cls.RETURN_TYPES), node
    assert callable(getattr(cls, cls.FUNCTION, None)), f"{node}.{cls.FUNCTION}"
    assert isinstance(getattr(cls, "CATEGORY", ""), str)


def test_function_accepts_every_declared_input(pack):
    """A widget the node declares but its FUNCTION does not accept makes
    ComfyUI fail at execution with an unexpected keyword argument."""
    import inspect

    for node, cls in pack.NODE_CLASS_MAPPINGS.items():
        sig = inspect.signature(getattr(cls, cls.FUNCTION))
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        params = set(sig.parameters)
        for _, inp, _ in _input_specs(cls):
            assert inp in params, f"{node}.{cls.FUNCTION} does not accept {inp!r}"

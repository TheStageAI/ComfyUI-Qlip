"""Standalone scripts: install.py (dry run) and tools/krea_check.py (graphs)."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("device", ["nvidia", "blackwell"])
def test_install_dry_run(repo_root, tmp_path, device):
    """The installer must plan a full install without touching the env."""
    (tmp_path / "requirements.txt").write_text("torch\n")
    out = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "install.py"),
            "--dry-run",
            "--device",
            device,
            "--comfy-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "torch" in out.stdout
    req = (
        "requirements_nvidia.txt"
        if device == "nvidia"
        else "requirements_blackwell.txt"
    )
    assert req in out.stdout


KREA_CONFIGS = [
    "eager",
    "eager_cache",
    "eager_cache_block",
    "eager_prog",
    "eager_prog_auto",
    "eager_sparse",
    "eager_prune",
    "engine",
    "engine_cache",
    "engine_prog",
    "engine_cache_prog",
    "release",
    "local_engine",
    "local_release",
    "compile_bf16",
    "compile_fp8",
    "compile_fp8_swap",
    "compile_fp8_aot",
    "compile_fp8_cache_prog",
    "compile_fp8_sparse",
]


@pytest.mark.parametrize("config", KREA_CONFIGS)
def test_krea_check_graph_is_wired(script, pack, config):
    """Every link points at an existing node, every node is a registered Qlip
    node or a ComfyUI core node, and the sampler's model is wired."""
    kc = script("tools/krea_check.py")
    g = kc.build(config, "krea2_retroanime.safetensors", 1024, "/engines")
    for nid, node in g.items():
        cls = node["class_type"]
        if cls.startswith("Qlip"):
            assert cls in pack.NODE_CLASS_MAPPINGS, f"{config}: {cls}"
        for inp, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2:
                assert value[0] in g, f"{config}: {nid}.{inp} -> {value[0]}"
    assert "model" in g["21"]["inputs"], config

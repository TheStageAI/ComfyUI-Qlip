#!/usr/bin/env python3
"""One-command, device-aware installer for ComfyUI-Qlip.

Why this exists
---------------
The hard part of installing Qlip alongside ComfyUI is never the qlip code — it is
the **torch build**. Three parties each want a different torch in the same venv:

  * ComfyUI's ``requirements.txt`` installs an unpinned CUDA-12 ``torch``.
  * The target device needs a specific build — cu12 on Hopper/Ada, **cu130** on
    Blackwell (RTX 5090 / B200), cu13 on Jetson Thor.
  * Older qlip wheels over-pinned ``torch<=2.9.1`` / ``numpy<2`` (now relaxed).

Any plain ``pip install`` can silently replace torch and break the CUDA engines.
This script removes the guesswork: it **detects the device**, installs the right
torch FIRST, then installs ComfyUI's requirements under a **constraints file** so
they can't move torch, then installs the matching ``qlip.core`` extra.

Usage
-----
    # from the ComfyUI root, with its venv active:
    python custom_nodes/ComfyUI-Qlip/install.py

    # options:
    python .../install.py --device blackwell   # force a profile (skip detection)
    python .../install.py --comfy-root /path/to/ComfyUI
    python .../install.py --dry-run            # print the plan, install nothing

Profiles: nvidia (Hopper/Ada, CUDA 12) · blackwell (RTX 5090 sm_120a / B200 sm_100,
CUDA 13). One code path; the only thing that changes between them is the CUDA
runtime stack (torch build + TensorRT + CuPy).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

THESTAGE_INDEX = ("https://thestage.jfrog.io/artifactory/api/pypi/"
                  "pypi-thestage-ai-staging/simple")

# Per-device install plan. `torch` is installed on its own first (from PyTorch's
# index, because the +cuXXX local build is not on PyPI); everything else follows.
PROFILES = {
    "nvidia": {
        "label": "Hopper / Ada (CUDA 12)",
        "torch": ["torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1"],
        "torch_index": "https://download.pytorch.org/whl/cu128",
        "constraints": ["torch==2.9.1", "torchvision==0.24.1",
                        "torchaudio==2.9.1"],
        "req": "requirements_nvidia.txt",
        "extra_no_deps": False,
    },
    "blackwell": {
        "label": "Blackwell RTX 5090 / B200 (CUDA 13)",
        "torch": ["torch==2.12.0", "torchvision==0.27.0"],  # torchaudio not on cu130
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "torch_pre": True,
        "constraints": ["torch==2.12.0+cu130", "torchvision==0.27.0",
                        "torchaudio==2.11.0", "numpy==2.4.6"],
        "req": "requirements_blackwell.txt",
        # qlip.core installed separately with --no-deps (its metadata would pull cu12).
        "extra_no_deps": True,
    },
}


def run(cmd, dry):
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.check_call(cmd)


def detect_device() -> str:
    """Best-effort device profile from the live machine (nvidia vs blackwell)."""
    # NVIDIA GPU: read the compute capability via nvidia-smi.
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True).strip().splitlines()
            caps = [float(c) for c in out if c.strip()]
            # sm_100 (B200 = 10.0) and sm_120 (RTX 5090 = 12.0) are Blackwell.
            if caps and max(caps) >= 10.0:
                return "blackwell"
            return "nvidia"
        except (subprocess.CalledProcessError, ValueError):
            return "nvidia"
    print("!! no NVIDIA GPU detected — defaulting to 'nvidia'.\n"
          "   Override with --device {nvidia,blackwell} if wrong.")
    return "nvidia"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Device-aware ComfyUI-Qlip installer")
    ap.add_argument("--device", choices=list(PROFILES),
                    help="force a profile instead of auto-detecting")
    ap.add_argument("--comfy-root", default=None,
                    help="ComfyUI root (default: two levels up from this file, "
                         "i.e. custom_nodes/ComfyUI-Qlip/..)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without installing")
    args = ap.parse_args()

    dev = args.device or detect_device()
    prof = PROFILES[dev]
    comfy_root = args.comfy_root or os.path.abspath(os.path.join(here, "..", ".."))
    pip = [sys.executable, "-m", "pip"]

    print(f"== ComfyUI-Qlip installer ==")
    print(f"device profile : {dev}  ({prof['label']})"
          + ("  [forced]" if args.device else "  [auto-detected]"))
    print(f"comfy root     : {comfy_root}")
    print(f"python         : {sys.executable}")
    if not os.path.isfile(os.path.join(comfy_root, "requirements.txt")):
        print(f"!! {comfy_root}/requirements.txt not found — is --comfy-root correct?")
        if not args.dry_run:
            sys.exit(1)
    print()

    # 1) torch FIRST, from PyTorch's index (the +cuXXX build isn't on PyPI).
    print("[1/3] torch (device build, before anything else can pin it)")
    cmd = pip + ["install"]
    if prof.get("torch_pre"):
        cmd.append("--pre")
    cmd += prof["torch"]
    if prof["torch_index"]:
        cmd += ["--index-url", prof["torch_index"]]
    run(cmd, args.dry_run)
    print()

    # 2) ComfyUI's own requirements, under a constraints file that pins torch so
    #    ComfyUI's unpinned `torch` line can't replace the build from step 1.
    print("[2/3] ComfyUI requirements (torch pinned via constraints)")
    cons_path = None
    if prof["constraints"]:
        fd, cons_path = tempfile.mkstemp(prefix="qlip_keep_", suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(prof["constraints"]) + "\n")
        print(f"  (constraints: {', '.join(prof['constraints'])})")
    cmd = pip + ["install", "-r", os.path.join(comfy_root, "requirements.txt")]
    if cons_path:
        cmd += ["-c", cons_path]
    run(cmd, args.dry_run)
    print()

    # 3) the device stack + qlip, from this node's requirements_<device>.txt.
    #    (We install from the requirements file rather than `pip install .[extra]`
    #    on purpose: a ComfyUI node is a set of files under custom_nodes, NOT a
    #    pip-buildable package — `pip install .` makes setuptools try to build a
    #    wheel from the flat node layout and fails.)
    #    torch was already installed in step 1 and is held pinned by the
    #    constraints file; the requirements file does not list torch.
    print("[3/3] device stack + qlip.core  (requirements_%s.txt)" % dev)
    req = os.path.join(here, prof["req"])
    cmd = pip + ["install", "-r", req]
    if cons_path:
        cmd += ["-c", cons_path]
    run(cmd, args.dry_run)
    if prof["extra_no_deps"]:
        # blackwell: the device requirements file deliberately does NOT install
        # qlip (it would pull a cu12 torch). Install qlip.core LAST, with --no-deps
        # so its metadata can't move the cu130 torch/numpy from steps 1-3.
        cmd = pip + ["install", "qlip.core[blackwell]", "--no-deps",
                     "--extra-index-url", THESTAGE_INDEX]
        if cons_path:
            cmd += ["-c", cons_path]
        run(cmd, args.dry_run)
    print()

    print("== done. verify the stack: ==")
    print("  python -c \"import torch, qlip; print('torch', torch.__version__); "
          "print('qlip OK')\"")
    print("Then set your TheStage token:  thestage config set --access-token <TOKEN>")


if __name__ == "__main__":
    main()

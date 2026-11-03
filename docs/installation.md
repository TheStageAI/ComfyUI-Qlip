# Installation

Install ComfyUI-Qlip, set your token and run a first workflow. For the short version, see the [README](../README.md#quick-start).

## Install with install.py

These steps take you from an empty machine to a running Qlip workflow.

### Prerequisites

- An NVIDIA GPU: Hopper / Ada (CUDA 12), or Blackwell RTX 5090 / B200 (CUDA 13).
- Python 3.10+. **Blackwell needs Python 3.11+**: the `blackwell` profile pins `numpy==2.4.6`.
- A TheStage access token from [app.thestage.ai](https://app.thestage.ai). The Qlip runtime
  checks it on import.
- ComfyUI at the commit your model needs:

  | Models | ComfyUI commit |
  |---|---|
  | FLUX.2 Klein, LTX-Video 2, Z-Image-Turbo (H100 engines) | `048dd2f3` |
  | LTX-Video 2.3, Qwen Image Edit, Wan 2.2 I2V, **all Blackwell engines** | `b615af1c` |
  | Krea 2 Turbo | `c2bcbecd` (v0.32.0, the first version with Krea 2) |

  Engines are tied to the block structure of the ComfyUI version they were compiled against, so
  compile and inference must use the same commit. `b615af1c` also runs the `048dd2f3` models.

### 1. Install

From an empty directory (for an existing ComfyUI, activate its venv and start at the `git clone`
of ComfyUI-Qlip):

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git checkout b615af1c          # pick the commit from the table above
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

git clone https://github.com/TheStageAI/ComfyUI-Qlip custom_nodes/ComfyUI-Qlip
python custom_nodes/ComfyUI-Qlip/install.py
```

`install.py` detects your GPU (`nvidia` or `blackwell`) and installs, in this order:

1. the matching torch build;
2. ComfyUI's `requirements.txt` under a constraints file, so it cannot replace that torch;
3. the device stack (`requirements_nvidia.txt` / `requirements_blackwell.txt`) and `qlip.core`.

> **Do not run ComfyUI's `pip install -r requirements.txt` yourself, especially on Blackwell.**
> It installs a CUDA 12 torch over the cu130 torch the Blackwell engines need. `install.py`
> already installs those requirements with torch pinned.

Useful flags:

```bash
python custom_nodes/ComfyUI-Qlip/install.py --device blackwell   # skip GPU detection
python custom_nodes/ComfyUI-Qlip/install.py --dry-run            # print the plan, install nothing
```

### 2. Set your token

```bash
thestage config set --access-token <YOUR_API_TOKEN>
```

The installer already installs the `thestage` CLI. Without a valid token, `import qlip` fails
with `RuntimeError: [PyArmor] Validation failed`, which looks like a broken install but is not.

### 3. Verify

```bash
python -c "import torch, tensorrt, qlip; \
  print('torch', torch.__version__); print('trt', tensorrt.__version__); print('qlip OK')"
```

On Blackwell, expect `torch 2.12.0+cu130`, `trt 10.15.1.29` and `qlip OK`.

### 4. Launch ComfyUI and run a workflow

```bash
python main.py --listen 0.0.0.0 --port 8188
```

Always activate the same venv before you launch. Then open a workflow from
[`workflows/`](../workflows/), for example `z-image-turbo-5090.json` on an RTX 5090. The
`Qlip Engines Loader` in it has `hf_repo` set, so the engine downloads on the first run. If you do
not have the model weights yet, see [Workflows and model downloads](workflows.md).

The first run loads the engine and takes a few seconds longer. Later runs reuse it.

## FP4-attention engines (RTX 5090)

FP4-attention engines (*fp4attn* in the tables) are the fastest RTX 5090 engines. They contain a
`QlipFP4Attention` plugin node, so they load only after the `fp4attn` plugin is built. Plain
FP8 / NVFP4 engines need none of this.

Requirements:

- RTX 5090 or another consumer Blackwell GPU (`sm_120a`). The plugin does not build or run on H100
  (`sm_90`) or B200 (`sm_100`).
- The `blackwell` install profile (PyTorch 2.12+cu130, `tensorrt-cu12` ≥ 10.15) and the CUDA 13
  toolkit (`nvcc`).

`qlip` ships the plugin source and builds it for you:

```bash
# 1. Download the TensorRT C++ headers and CUTLASS source, check nvcc/ninja,
#    and add the build variables to the venv's activate script.
qlip-setup-plugins          # or: python -m qlip.plugins setup

# 2. Re-activate the venv so those variables are set.
source venv/bin/activate

# 3. Build and register the plugin. The first build takes ~30-90 s of nvcc and is cached.
python -c "import torch; from qlip.plugins.fp4attn import ensure_plugin_registered; \
           ensure_plugin_registered(); print('fp4attn plugin OK')"
```

If something is missing, `ensure_plugin_registered()` prints the exact fix. At inference,
`QlipEnginesLoader` calls it automatically, so once the plugin is built, fp4attn engines load like
any other. The full build reference is in the installed package, in `qlip/plugins/BUILD.md` and
`qlip/plugins/README.md`. To find it:

```bash
python -c "import qlip, pathlib; print(pathlib.Path(qlip.__file__).parent / 'plugins')"
```

The plugin compiles its FP4 attention kernel into its own `.so`, so it does not need the
SageAttention package. SageAttention3 is only used by the eager attention path (KJNodes
`Patch Sage Attention KJ`), not by these engines.

## Manual installation

Use this only to debug a step of `install.py`, which runs the same sequence. Install torch first
from PyTorch's index (the `+cuXXX` builds are not on PyPI), then the requirements under a
constraints file that keeps torch fixed. Run from the ComfyUI root with the venv active.

**Hopper / Ada (CUDA 12):**

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
printf 'torch==2.9.1\ntorchvision==0.24.1\ntorchaudio==2.9.1\n' > /tmp/keep.txt
pip install -r requirements.txt -c /tmp/keep.txt            # ComfyUI's own
pip install -r custom_nodes/ComfyUI-Qlip/requirements_nvidia.txt -c /tmp/keep.txt
```

**Blackwell RTX 5090 / B200 (CUDA 13):** install `qlip.core` last with `--no-deps`; otherwise its
metadata pulls a cu12 torch over the cu130 build.

```bash
pip install --pre torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130
printf 'torch==2.12.0+cu130\ntorchvision==0.27.0\ntorchaudio==2.11.0\nnumpy==2.4.6\n' > /tmp/keep.txt
pip install -r requirements.txt -c /tmp/keep.txt            # ComfyUI's own
pip install -r custom_nodes/ComfyUI-Qlip/requirements_blackwell.txt -c /tmp/keep.txt
pip install "qlip.core[blackwell]" --no-deps \
    --extra-index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-staging/simple
```

After the `--no-deps` step, a later `pip install` may print
`ERROR: ... qlip-core requires cvxpy / Cython / scikit-learn / thop, which are not installed`. This is
harmless for inference: those packages back qlip's model-analysis tooling (ANNA), not the engine
runtime. To silence it, run `pip install cvxpy Cython scikit-learn thop`.

Then set the token and verify as in [Install with install.py](#2-set-your-token).

### Optional extras

```bash
# Flash Attention for Hopper GPUs
pip install flash-attn --no-build-isolation

# Nodes used in the comparison benchmarks (run from ComfyUI/custom_nodes)
git clone https://github.com/kijai/ComfyUI-KJNodes              # torch.compile node
git clone https://github.com/sgl-project/ComfyUI_SGLDiffusion    # SGLang acceleration
pip install "sglang[diffusion]"
```

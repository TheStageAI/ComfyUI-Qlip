---
name: qlip-model-compiler
description: Compile diffusion models (image/video) to optimized Qlip engines for ComfyUI with LoRA support, FP8 quantization, and dynamic shapes. Handles model analysis, patching, compilation scripts, bash wrappers, and inference-time node integration.
---

# Qlip Model Compiler

Compile any ComfyUI-compatible diffusion model into optimized Qlip engines with runtime LoRA support.

## Before You Start

You need from the user:
1. **ComfyUI workflow JSON** (expanded — all subgraphs must be unpacked so every node is visible in the JSON)
2. **Model checkpoint path** on the target server
3. **LoRA file path** (if LoRA support is needed)
4. **Target GPU** (H100, B200, L40S, RTX 5090)
5. **Target resolutions and batch sizes**
6. **Server SSH access** (if remote compilation)
7. **Path to ComfyUI installation** — you need to read ComfyUI source code and custom_nodes

From the workflow JSON you determine: model loader type, block class names, text encoder, VAE, sampler settings, CFG value.

**You must have read access to:**
- ComfyUI source code (`comfy/ldm/`, `comfy/model_*.py`, `comfy/sample.py`)
- `custom_nodes/` — model code may live here, not in ComfyUI core
- The model's transformer and block classes to analyze forward() signatures and return types

## Step 0: Analyze the Model

Before writing any code, you MUST understand the model architecture. Do this:

### Read ComfyUI source code

The model's Python class lives in ComfyUI or a custom_node. Find it:
1. From the workflow JSON, identify the model loader node (e.g., `UNETLoader`, `CheckpointLoaderSimple`)
2. Find what `model_name` / `ckpt_name` is loaded
3. Read the model class in ComfyUI source: `comfy/ldm/` or in `custom_nodes/`
4. Find the **transformer class** (the `diffusion_model` attribute of the loaded model)
5. Find the **block class** — what's inside `transformer.blocks` / `transformer.layers` / `transformer.transformer_blocks`

### Read block.forward()

This is the function that will be compiled. Read it carefully and check:

1. **What arguments does it receive?** List all params with types (tensor, None, dict, dataclass, tuple, etc.)
2. **What does it return?** Single tensor, tuple of tensors, or can it return None in some branches?
3. **Does it use custom C++/CUDA ops?** (especially RoPE — `comfy_kitchen`, `torch.view_as_complex`, `torch.polar`)
4. **Are weights fused?** (e.g., `attention.qkv` instead of separate `to_q/to_k/to_v`)

### Read the caller

Find the function that calls block.forward() (usually `forward_orig` or `_process_transformer_blocks`):
1. **What does it pass to each block?** Are arguments tensors or Python objects?
2. **Does it compute modulation/conditioning outside the blocks?**
3. **Does it do anything between blocks that would break if blocks become compiled engines?**

### Check the workflow for weight offloading

**CRITICAL**: After the model is loaded and Qlip engines replace transformer blocks, nothing in the workflow should try to offload or reload the model weights. Look for nodes like:
- `ModelMemoryFree` / `FreeModelMemory`
- Any node that calls `model_patcher.unpatch_model()` or moves the model to CPU

Qlip engines are compiled binary — if something offloads them, they cannot be restored. The engines must stay in GPU memory for the entire inference.

### Check block return values

If block.forward() can return `None` in some code paths (e.g., conditional branches), this is a problem — ONNX export cannot handle optional tensor outputs. You need to patch the block to always return tensors (e.g., return zero tensor instead of None).

## Compilation Pipeline

**Order is critical. Do not reorder.**

```
1. Load model (ComfyUI API)
2. Quantize (FP8, if needed — BEFORE LoRA)
3. Setup LoRA (patches block signatures)
4. Patch blocks (model-specific, BEFORE setup_modules)
5. Setup compiler (NvidiaCompileManager + NvidiaBuilderConfig)
6. Patch caller (model-specific, AFTER setup_modules)
7. Calibration inference (shape collection)
8. Set dynamic axes + shape profiles
9. Patch RoPE for export (temporary)
10. Compile
11. Restore RoPE
12. Save lora_config.json
```

## Python Script Structure

### Imports

```python
import argparse
import sys
from pathlib import Path

import torch

# Qlip compiler
from qlip.compiler.nvidia import (
    NvidiaBuilderConfig,
    NvidiaCompileManager,
    NVIDIA_FLOAT_W8A8,
    NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC,
)
from qlip.quantization import QuantizationManager

# LoRA (elastic_models)
from elastic_models.diffusers.lora import LoRAManager, QlipLoraModule
```

### Model Loading

For UNETLoader models (most image models):
```python
import comfy.sd
unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
model_patcher = comfy.sd.load_diffusion_model(unet_path, model_options={})
transformer = model_patcher.model.diffusion_model
```

For checkpoint models (model + VAE + CLIP bundled):
```python
model_patcher, clip, vae = comfy.sd.load_checkpoint_guess_config(ckpt_path)
transformer = model_patcher.model.diffusion_model
```

Extract latent info:
```python
latent_format = model_patcher.model.model_config.latent_format
vae_downscale = getattr(latent_format, "spacial_downscale_ratio",
                        getattr(latent_format, "spatial_downscale_ratio", 8))
patch_size = getattr(transformer, "patch_size", 2)
```

### FP8 Quantization

```python
def get_quantizable_modules(transformer, skip_first=0, skip_last=0):
    modules = []
    blocks = transformer.blocks  # or .layers, .transformer_blocks — model-dependent
    for i, block in enumerate(blocks):
        if skip_first <= i < len(blocks) - skip_last:
            for name, mod in block.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    modules.append(mod)
    return modules

modules = get_quantizable_modules(transformer, skip_first=1, skip_last=1)
qmanager = QuantizationManager()

# Dynamic (no calibration needed):
qmanager.setup_modules(modules, calibration_iterations=0,
    activations_scale_offset_dtype=torch.bfloat16,
    weights_scale_offset_dtype=torch.bfloat16,
    **NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC)

# Static (requires calibration inference):
qmanager.setup_modules(modules, calibration_iterations=10,
    activations_scale_offset_dtype=torch.bfloat16,
    weights_scale_offset_dtype=torch.bfloat16,
    **NVIDIA_FLOAT_W8A8)
# Then run calibration_iterations x run_calibration_inference(...)
```

### LoRA Setup

```python
configs = LoRAManager.infer_config(lora_path)
for c in configs:
    blocks = getattr(transformer, c.block_prefix)
    manager = LoRAManager(c, device="cuda", dtype=torch.bfloat16)
    manager.load_from_safetensors(lora_path, strength=1.0)
    packed = [manager.pack_block(f"{c.block_prefix}.{i}", max_rank)
              for i in range(len(blocks))]
    # IMPORTANT: fill with small random noise, NOT zeros
    for pt in packed:
        pt.uniform_(-0.01, 0.01)
    QlipLoraModule.setup(transformer, c.block_prefix, c, packed, max_rank=max_rank)
```

### Compiler Setup

```python
nvidia_config = NvidiaBuilderConfig(
    builder_flags={"BF16", "WEIGHT_STREAMING"},  # add "FP8" if quantized
    io_dtype="base",
    io_dtype_per_tensor={"pe_name": "FP32"},  # RoPE tensors in FP32 if needed
    creation_flags={"STRONGLY_TYPED"},
    profiling_verbosity="DETAILED",
)

cm = NvidiaCompileManager(transformer, workspace=engines_dir)

# By block type:
cm.setup_modules(module_types=(MyBlockClass,), builder_config=nvidia_config, dtype=torch.bfloat16)

# Or by explicit names (when multiple block types with different signatures exist):
cm.setup_modules(modules=[f"blocks.{i}" for i in range(n_blocks)],
                 builder_config=nvidia_config, dtype=torch.bfloat16)
```

### Calibration & Shape Collection

```python
for module in cm.modules:
    module._collect_shapes = True

run_calibration_inference(model_patcher, clip, prompt="test prompt",
                          width=1024, height=1024, steps=2)

for module in cm.modules:
    module._collect_shapes = False
```

### Dynamic Axes & Profiles

```python
dynamic_axes = {
    "x": {0: "batch_size", 1: "seq_len"},
    "lora_packed": {1: "lora_rank"},
}

# Token calculation: img_tokens = (w // (vae_downscale * patch_size)) * (h // (vae_downscale * patch_size))

profiles = []
for w, h in sizes:
    tokens = calc_tokens(w, h)
    profiles.append({
        "min": {"batch_size": 1, "seq_len": tokens + txt_len, "lora_rank": 1},
        "opt": {"batch_size": 1, "seq_len": tokens + txt_len, "lora_rank": 32},
        "max": {"batch_size": 1, "seq_len": tokens + txt_len, "lora_rank": max_rank},
    })

for module in cm.modules:
    module.ioconfig.set_axes_profiles(dynamic_axes, profiles)
```

### Compile

```python
restore_rope = patch_rope_for_export()
transformer.to("cpu")
torch.cuda.empty_cache()

cm.compile(cpu_offload=True, dynamo=True, keep_compiled=False,
           recompile_existing=False, dump_onnx=False)
restore_rope()
```

### Save LoRA Config

```python
import json

def save_lora_config_json(path, configs):
    entries = []
    for cfg in configs:
        entries.append({
            "name": cfg.name,
            "block_prefix": cfg.block_prefix,
            "num_blocks": cfg.num_blocks,
            "max_features": cfg.max_features,
            "layers": [{"name": l.name, "out_features": l.out_features,
                        "in_features": l.in_features} for l in cfg.layers],
        })
    with open(path, "w") as f:
        json.dump({"configs": entries}, f, indent=2)

save_lora_config_json(engines_dir / "lora_config.json", lora_configs)
```

## Model Download Script Pattern

Always create a download script that fetches the model, text encoder, VAE, and LoRA to the correct ComfyUI directories. From the workflow JSON, identify what files are needed.

```bash
#!/bin/bash
set -e

COMFYUI_PATH="${COMFYUI_PATH:-./ComfyUI}"
HF_TOKEN="${HF_TOKEN:-$(python3 -c "from huggingface_hub import HfFolder; print(HfFolder.get_token() or '')" 2>/dev/null)}"

mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"

download_url() {
    local url=$1 output_dir=$2 description=$3
    local filename=$(basename "$url")
    echo "Downloading: $description"
    if [ -f "$output_dir/$filename" ]; then
        echo "  Already exists, skipping"
        return 0
    fi
    curl -L -C - -H "Authorization: Bearer $HF_TOKEN" -o "$output_dir/$filename" "$url"
    echo "  Done: $output_dir/$filename"
}

# Diffusion model
download_url \
    "https://huggingface.co/ORG/MODEL/resolve/main/model.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "Model Name"

# Text encoder
download_url \
    "https://huggingface.co/ORG/MODEL/resolve/main/text_encoder.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "Text Encoder Name"

# VAE
download_url \
    "https://huggingface.co/ORG/MODEL/resolve/main/vae.safetensors" \
    "$COMFYUI_PATH/models/vae" \
    "VAE Name"

# LoRA (for compilation)
download_url \
    "https://huggingface.co/ORG/LORA/resolve/main/lora.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "LoRA Name"

echo "All models downloaded to $COMFYUI_PATH/models/"
```

Get the exact HuggingFace URLs from the model card or by browsing the repo files. For gated models, remind the user to accept the license on HuggingFace first.

## Bash Wrapper Script Pattern

```bash
#!/bin/bash
set -e

COMFYUI_PATH="${COMFYUI_PATH:-./ComfyUI}"
LORA_PATH="${LORA_PATH:-$COMFYUI_PATH/models/loras/my_lora.safetensors}"
TEXT_ENCODER="encoder_name.safetensors"
SIZES="--sizes 1024x1024 1024x768"
DYNAMIC="--dynamic-size-range 512 768 1024"
LORA_RANK="--min-lora-rank 1 --opt-lora-rank 32 --max-lora-rank 128"
COMMON="--comfyui-path $COMFYUI_PATH --text-encoder $TEXT_ENCODER $SIZES $DYNAMIC"
FP8="--quantize --static --calibration-iterations 10 --skip-first-blocks 1 --skip-last-blocks 1"

# BF16 + LoRA
python compile_my_model.py --model model.safetensors --engines-dir ./engines/bf16-lora \
  $COMMON --lora "$LORA_PATH" $LORA_RANK

# FP8 + LoRA
python compile_my_model.py --model model.safetensors --engines-dir ./engines/fp8-lora \
  $COMMON $FP8 --lora "$LORA_PATH" $LORA_RANK
```

## When Does a Block Need Patching?

Read block.forward() and ask these questions:

### 1. Non-tensor arguments?

| Input type | Patch needed? | Solution |
|---|---|---|
| `None` | No | Qlip auto-strips |
| `dict` (transformer_options) | No | Qlip auto-strips |
| **Dataclass** | Yes | Stack fields into tensor |
| **Tuple with non-tensor** (cos, sin, bool) | Yes | Stack tensors, store non-tensor as block attribute |
| **Custom Python object** | Yes | Expand to raw tensors in caller |

### 2. Modulation computed outside blocks?

If caller passes dataclass/tuple to blocks → **two patches**:
- Block patch (BEFORE setup_modules) — baked into engine
- Caller patch (AFTER setup_modules) — needed at compile AND inference time

### 3. RoPE compatibility?

RoPE (Rotary Position Embeddings) is the most common source of export failures. Check for ALL of these:

| What to grep for in the model code | Problem | Fix |
|---|---|---|
| `comfy_kitchen.apply_rope` | C++ custom op, not exportable | Replace with pure PyTorch `_apply_rope` temporarily during export |
| `torch.view_as_complex` | Complex tensor ops not supported in ONNX | Replace with real-valued `[cos, sin]` stack + explicit view/contiguous |
| `torch.polar` | Complex tensor construction | Same as above — replace with cos/sin computation |
| `torch.view_as_real` | Complex→real conversion | Replace with explicit reshape |

**Universal RoPE patch pattern** (temporary, for export only):
```python
def patch_rope_for_export():
    import comfy.ldm.flux.math as flux_math
    orig_apply_rope = flux_math.apply_rope
    orig_apply_rope1 = flux_math.apply_rope1
    flux_math.apply_rope = flux_math._apply_rope    # Pure PyTorch
    flux_math.apply_rope1 = flux_math._apply_rope1  # Pure PyTorch
    def restore():
        flux_math.apply_rope = orig_apply_rope
        flux_math.apply_rope1 = orig_apply_rope1
    return restore
```

If the model uses its own RoPE implementation (not from `comfy.ldm.flux.math`), you need to write a custom patch that replaces complex ops with real-valued equivalents. The key rule: **no complex dtypes anywhere in the forward path during export**.

### 4. Fused QKV weights + LoRA?

Model has `attention.qkv` but LoRA has `attention.to_q/to_k/to_v` → unfuse into separate Linear layers.

### 5. Batch size symbolic?

`view(batch * seq, hidden)` with both dynamic → fix batch at export, compile separate engines per batch.

### 6. Block returns None in some branches?

If `block.forward()` has conditional branches where some return `None` or skip outputs:
```python
if condition:
    return output
else:
    return None  # ← ONNX cannot handle this
```

Patch to always return a tensor (e.g., zeros with the expected shape). The engine signature must be deterministic.

### 7. Caller needs patching?

If caller passes non-tensor data to compiled blocks → add caller patch AFTER setup_modules + in inference nodes.

### 8. Weight offloading in the workflow?

Check if ANY node in the workflow after model loading might offload or unload model weights. Qlip engines are compiled binaries in GPU memory — they cannot be offloaded and restored like PyTorch weights. If a node calls `model.unpatch_model()` or moves to CPU, engines break silently.

Warn the user if you see `ModelMemoryFree`, weight offloading nodes, or any mechanism that evicts the model after loading.

## Adding Inference Support

When a model needs caller-level patches at inference time, add to `utils/helpers.py`:

```python
def is_my_model(dm):
    return type(dm).__name__ == "MyModelClass"

def patch_my_model_caller(dm):
    # Patch forward_orig or process_transformer_blocks
    ...
```

And in `nodes/engine_loader.py` → `_apply_model_patches()`:

```python
from ..utils import is_my_model, patch_my_model_caller
if is_my_model(dm):
    patch_my_model_caller(dm)
```

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| ONNX export fails on custom op | RoPE uses C++ kernel | `patch_rope_for_export()` |
| ONNX export fails on complex dtype | `torch.view_as_complex` / `torch.polar` | Replace with real-valued cos/sin — no complex dtypes during export |
| ONNX export fails on None output | Block returns None in some branch | Patch to always return tensor (zeros if needed) |
| Engine gets wrong inputs | kwargs not filtered | Check `_allowed_params` in QlipLoraModule |
| FP8 quality bad | Fused QKV one shared scale | `--unfuse-qkv` + `--skip-first-blocks 1` |
| LoRA no effect | Key format mismatch | `_convert_lora_format()` auto-detects Kohya/XLabs/diffusers |
| Slow with LoRA disabled | Full-rank zero tensor | Use rank=1 zero tensor |
| Shape mismatch | Resolution outside profile range | Check min/max in profiles |
| Engines silently broken after run | Weight offloading node in workflow | Remove any node that offloads model after Qlip engines are loaded |
| Model class not found | Model code in custom_nodes, not ComfyUI core | Check `custom_nodes/` directory for the model implementation |

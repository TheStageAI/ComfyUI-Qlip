# ComfyUI-Qlip

GPU-accelerated inference for diffusion models in ComfyUI.

Qlip compiles transformer blocks into optimized TensorRT engines, delivering significant speedups with full runtime LoRA support. Engines are compiled once and reused across all inference runs.

## Supported Models

| Model | Variants | Status |
|-------|----------|--------|
| **FLUX.2 Klein** | 9B, Base 9B, 4B, Base 4B | Supported |
| **LTX-Video 2** (LTXAV) | 19B Distilled | Supported |

More models coming soon — stay tuned for updates.

## Quick Start

### 1. Install

Clone or symlink into your ComfyUI custom nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/TheStageAI/ComfyUI-Qlip qlip_nodes
```

### 2. Use in ComfyUI

Connect `Qlip Engines Loader` after your model loader:

```
UNETLoader
  -> Qlip Engines Loader (engines_path=./engines/..., with_lora=False)
    -> Guider -> Sampler -> VAEDecode -> SaveImage
```

## Nodes

### Qlip Engines Loader

Loads pre-compiled engines and replaces transformer blocks at runtime. Caches engines across runs — first load takes a few seconds, subsequent runs are instant.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model from UNETLoader |
| `with_lora` | BOOLEAN | Yes | | `True` if engines were compiled with `--lora` |
| `engines_path` | STRING | No | `""` | Path to directory with `.engine` files |
| `hf_repo` | STRING | No | `""` | HuggingFace repo with engines (alternative to local path) |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack from `Qlip LoRA Stack` node(s) |
| `lora_config` | QLIP_LORA_CONFIG | No | | Config from `Qlip LoRA Config` node |
| `max_rank` | INT | No | `128` | Maximum LoRA rank (must match compilation) |

**Output:** `MODEL`

### Qlip LoRA Stack

Builds a chainable list of LoRA entries. Each node adds one LoRA file. Chain multiple nodes via `prev_stack` to stack LoRAs.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `lora_path` | STRING | Yes | `""` | Path to LoRA `.safetensors` file |
| `strength` | FLOAT | Yes | `1.0` | Strength multiplier (`-10.0` to `10.0`) |
| `prev_stack` | QLIP_LORA_STACK | No | | Previous stack to extend |

**Output:** `QLIP_LORA_STACK`

### Qlip LoRA Config

Loads LoRA config from JSON (auto-generated during compilation as `lora_config.json`). Ensures inference config matches compilation exactly.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `config_path` | STRING | No | `""` | Path to `lora_config.json` |

**Output:** `QLIP_LORA_CONFIG`

### Qlip LoRA Switch

Enables or disables LoRA at runtime without reloading engines. Use after `Qlip Engines Loader` with `with_lora=True`.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model with loaded engines |
| `enable` | BOOLEAN | Yes | `True` | Enable or disable LoRA |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack to load (when enabling) |
| `max_rank` | INT | No | `128` | Maximum LoRA rank |

**Output:** `MODEL`

## Workflows

### Basic (no LoRA)

```
UNETLoader (model.safetensors)
  -> Qlip Engines Loader (engines_path=..., with_lora=False)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

### With LoRA

```
Qlip LoRA Config (lora_config.json)  ──────────────────────────────┐
                                                                    │
Qlip LoRA Stack (lora.safetensors, strength=1.0)  ───────────────┐ │
                                                                  │ │
UNETLoader (model.safetensors)                                    │ │
  -> Qlip Engines Loader (with_lora=True, lora_stack=↑, config=↑)
    -> BasicGuider / CFGGuider
      -> SamplerCustomAdvanced
        -> VAEDecode -> SaveImage
```

### Multi-LoRA stacking

```
Qlip LoRA Stack (style_lora.safetensors, 0.8)
  -> Qlip LoRA Stack (detail_lora.safetensors, 0.5, prev_stack=↑)
    -> Qlip Engines Loader (with_lora=True, lora_stack=↑)
```

Multiple LoRAs are stacked — their ranks concatenate. Total rank must fit within `--max-lora-rank` used at compilation time.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- TensorRT 10+
- `qlip` package
- `elastic_models` package (only needed with LoRA)

## LoRA Details

LoRA weights are **runtime inputs** to compiled engines, not baked into weights. The `lora_packed` tensor shape `[num_layers, rank, max_features, 2]` holds A and B matrices. The rank dimension is dynamic, allowing different LoRA files without recompilation.

**Caching behavior:**
- Engines loaded once per path, reused across runs
- LoRA weights hot-swapped in-place when stack changes
- Same LoRA between runs — no work at all
- LoRA removed — packed tensors zeroed, no engine reload

**Constraints:**
- `max_rank` at inference must be <= `--max-lora-rank` at compilation
- Engines compiled without `--lora` cannot accept LoRA at runtime
- Engines compiled with `--lora` work both with and without LoRA
- LyCORIS / LoKR format is not supported

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Shape mismatch errors | Check that `with_lora` matches how engines were compiled |
| Slow first run | Normal — engine loading takes a few seconds on first use |
| Engines don't work after GPU change | Engines are GPU-specific — recompile after changing hardware or TensorRT version |

# ComfyUI-Qlip

GPU-accelerated inference for diffusion models in ComfyUI using Qlip engines.

## Skills

- `skills/qlip-model-compiler/SKILL.md` — compile new models, write patching code, create bash scripts

## Structure

```
nodes/engine_loader.py  — QlipEnginesLoader, QlipLoraStack, QlipLoraSwitch
nodes/timer.py          — QlipTimerStart, QlipTimerStop, QlipTimerReport
utils/helpers.py        — engine loading, model detection, inference patches
```

## Rules

- Compilation order: Quantization → LoRA → Block patches → setup_modules → Caller patches → Calibration → Compile
- Block patches baked into engines at compile time; caller patches apply at inference
- LoRA disabled = rank-1 zero tensor (minimal overhead)
- RoPE custom ops must be replaced with pure PyTorch for export

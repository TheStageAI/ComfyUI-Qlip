# ComfyUI-Qlip

Accelerated inference for diffusion models in ComfyUI using Qlip engines.

## Skills

- `skills/qlip-model-compiler/SKILL.md` — full guide for compiling new models (7 phases: investigation → patching → testing → BF16 compilation → FP8 quantization → benchmarking → inference integration)

## Structure

```
nodes/engine_loader.py  — QlipEnginesLoader, QlipLoraStack, QlipLoraSwitch
nodes/timer.py          — QlipTimerStart, QlipTimerStop, QlipTimerReport
utils/helpers.py        — engine loading, model detection, inference patches
```

## Critical Rules

- Compilation order: Quantization → Block patches → LoRA → setup_modules → Caller patches → Calibration → Compile
- Block patches BEFORE LoRA (LoRA wraps the patched forward)
- Avoid patching block.forward if possible — qlip auto-strips None/dict args
- RoPE patch must also patch local references in model files (not just flux_math module)
- dynamic_axes must only contain args that survive calibration (not None/dict args)
- LoRA `lora_packed` is always FIRST positional arg — handled automatically by QlipLoraModule.setup()
- BF16 weights required for compilation — never use FP8 checkpoints as input
- FP8 for H100/B200/Ada+; INT8 for A100

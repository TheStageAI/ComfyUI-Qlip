# Compile your own model

Build Qlip engines for a model, GPU or resolution range that the precompiled engines do not cover.

Compile your own engines when your model is not in the precompiled list, when you need engines for
another GPU (engines are hardware-specific), or when you want custom resolution ranges or LoRA
support.

### With Claude Code (recommended)

This repository includes an [agent skill](../skills/qlip-model-compiler/SKILL.md) for
[Claude Code](https://claude.ai/claude-code). The agent analyzes the workflow and the model source,
writes the download, compile and benchmark scripts, runs the compilation and, if needed, adds
inference support to the nodes.

Give it:

1. **The ComfyUI workflow JSON.** Expand all subgraphs and groups before exporting, so every node
   is in the file.
2. **The path to ComfyUI**, so the agent can read the model source.
3. **The model path** and, if needed, **the LoRA path**.
4. **The target GPU and resolutions.**
5. **SSH access** to the server, if you compile remotely.

Example prompt:

```
I want to compile my model with Qlip.

Workflow: /path/to/my_workflow.json
ComfyUI: /path/to/ComfyUI
Model: my_model.safetensors (UNETLoader, in models/diffusion_models/)
LoRA: my_lora.safetensors (in models/loras/)
Text encoder: my_encoder.safetensors (CLIPLoader)
VAE: my_vae.safetensors
Target: H100, 512-1024px, FP8 quantization, LoRA support
```

### By hand

The [compilation skill reference](../skills/qlip-model-compiler/SKILL.md) has the full API: imports,
function signatures, patching patterns and bash wrapper templates.

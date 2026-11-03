# Troubleshooting

Common problems and their fixes.

| Problem | Solution |
|---------|----------|
| `RuntimeError: [PyArmor] Validation failed` on `import qlip`, or `Nvidia support is not available` when loading an engine | The TheStage token is missing or invalid. Run `thestage config set --access-token <TOKEN>` and retry; do not reinstall qlip |
| `install.py` stops with `the blackwell profile needs Python 3.11+` | Create the venv with Python 3.11 or newer |
| Blackwell engines fail after installing another package | A `pip install` replaced the cu130 torch with a CUDA 12 build. Re-run `install.py` |
| An fp4attn engine fails to load (`QlipFP4Attention` plugin not found) | Run the plugin setup in [FP4-attention engines](installation.md#fp4-attention-engines-rtx-5090), re-activate the venv and restart ComfyUI |
| Shape mismatch errors | The engine does not match the model or the input shape. LoRA-enabled engines need `lora_config.json` in the engines directory; check the model's static/dynamic size range |
| Engines do not work after a GPU change | Engines are GPU-specific. Use the folder for your GPU or recompile |
| Engines fail after a ComfyUI update | Engines are tied to the ComfyUI commit they were compiled against. Check out the commit from the [Prerequisites](installation.md#prerequisites) table |
| Slow first run | Expected: the first run loads the engine (a few seconds) |
| FP8 quality looks bad | Recompile with `--unfuse-qkv` and `--skip-first-blocks 1 --skip-last-blocks 1` |
| LoRA has no effect | Check that `lora_config.json` exists in the engines directory and the LoRA rank is ≤ the compiled maximum rank |
| `flash_attn 3 package is not installed` | Optional, Hopper only: `pip install flash-attn --no-build-isolation` |
| `No module named 'click'` when running `hf` | `pip install click` |

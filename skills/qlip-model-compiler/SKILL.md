---
name: qlip-model-compiler
description: Accelerate diffusion models by compiling transformer blocks with TheStage AI Qlip Framework. Covers model investigation, patching, compilation, quantization, benchmarking, and LoRA support.
---

# Acceleration of Diffusion Pipelines with TheStage AI Qlip Framework

## Goal

Compile transformer blocks inside diffusion architectures into optimized engines. The target blocks are typically the full transformer block (QKV + attention + projection + MLP) which benefits most from kernel fusion. During this work we may encounter operations that need patching before compilation.

## Reference Documentation

Read these pages when you need API details beyond what's in this skill:

- **Get Started**: https://docs.thestage.ai/qlip/docs/source/get_started.html
- **Quantization API**: https://docs.thestage.ai/qlip.core/docs/source/qlip.quantization_api.html
- **Nvidia Compilation & Inference API**: https://docs.thestage.ai/qlip.core/docs/source/qlip.deploy_nvidia_api.html
- **Quantization Tutorial**: https://docs.thestage.ai/tutorials/source/quantization_tutorial.html
- **TheStage AI Platform**: https://app.thestage.ai (API tokens, model access)

## Prerequisites

From the user:
1. **ComfyUI workflow JSON** (expanded — all subgraphs unpacked so every node is visible)
2. **Path to ComfyUI installation** (to read model source code and custom_nodes)
3. **Model checkpoint path** (MUST be BF16/FP16 — **NEVER FP8**. Even if the workflow JSON references an FP8 checkpoint, you MUST find and download the BF16/FP16 version of the same model. We handle all quantization ourselves via ANNA/Qlip — starting from an already-quantized FP8 checkpoint would double-quantize and destroy quality.)
4. **LoRA file path** (if LoRA support is needed)
5. **Target GPU** and **target resolutions**
6. **Server SSH access** (if remote compilation)

## How to Work

1. **Default flow**: Write all scripts (download, compile, bash wrapper), provide commands for the user to run. Or run them yourself if the user provides SSH access. Do NOT run tests/checks upfront — just implement.

2. **Debugging flow**: If something fails, THEN isolate the problem with targeted tests (single block export, dynamo trace, single block compile, etc. — see Phase 3). Use these tests to diagnose, not as mandatory steps before every compilation.

3. **Workflow analysis**: Always read the workflow JSON to check for weight offloading nodes that would break compiled engines. Warn the user proactively if found.

---

## Phase 1: Model Investigation

### 1.1 Find the transformer architecture

From the workflow JSON, identify the model loader node. **If the workflow references an FP8 checkpoint** (e.g. `model_fp8_e4m3fn.safetensors`), find the BF16/FP16 version from the same HuggingFace repo. The download script and compile script must use BF16/FP16 — we quantize ourselves.

Then read the model source code:
- ComfyUI core models: `comfy/ldm/`
- **Custom node models: `custom_nodes/`** — many models are NOT in ComfyUI core. The model class may live entirely inside a custom node package. Check the workflow JSON for non-standard node types and trace them to their `custom_nodes/` directory.

Find:
- **Transformer class** — the `diffusion_model` attribute of the loaded model
- **Block class** — what's inside `transformer.blocks` / `transformer.layers` / `transformer.transformer_blocks`
- **Block count** — `len(transformer.transformer_blocks)`

### 1.2 Analyze block.forward()

Read the block's forward method. Document:

| Question | What to look for |
|----------|-----------------|
| **Input arguments** | List all params with types (tensor, None, dict, dataclass, tuple) |
| **Return values** | Single tensor, tuple of tensors, or conditional None? |
| **RoPE implementation** | `comfy_kitchen`, `torch.view_as_complex`, `torch.polar`? |
| **Fused weights** | `attention.qkv` vs separate `to_q/to_k/to_v`? |
| **Non-tensor inputs** | Dataclasses, tuples with bools, Python objects? |

### 1.3 Compute seq_len formula

**You MUST derive the correct `compute_seq_len(width, height, ...)` formula from the model source code.** Do NOT guess — read how the model converts pixel dimensions to token sequence length.

The formula involves multiple compression stages:
1. **VAE compression** — reduces spatial (÷8 typical) and temporal (÷4 for video) dimensions
2. **Patch embedding** — further reduces by `patch_size` (e.g. `(1,2,2)` or `(2,2)`)
3. **Additional tokens** — text tokens, image tokens (CLIP), reference tokens

**How to derive:** Find `patch_embedding` in the model's `forward_orig`:
```python
# Example from Wan2.2:
# forward_orig does: x = self.patch_embedding(x)  where x is [B, C, T, H, W] latent
# patch_size = (1, 2, 2) — defined in model __init__
# VAE compresses: temporal ÷4, spatial ÷8
# So: latent shape = [B, C, frames//4, height//8, width//8]
# After patch_embedding: tokens = (frames//4) * (height//8 // 2) * (width//8 // 2)
```

**Common mistake:** Using pixel `num_frames` directly instead of latent frames. Video VAEs compress temporally (typically ÷4):
```python
# WRONG:
t = num_frames // patch_t  # 81 // 1 = 81

# CORRECT:
latent_t = (num_frames + 3) // 4  # VAE temporal compression: 81 → 21
t = latent_t // patch_t            # 21 // 1 = 21
```

**Verification:** Run one calibration inference and check the log:
```
Add shape profile: {'x': ((2, 32760, 5120), ...)}
```
`32760` is the real seq_len. Verify: `21 * 52 * 30 = 32760` for 480x832, 81 frames. If your formula gives a different number, it's wrong.

**Template function — adapt per model:**
```python
def compute_seq_len(width, height, num_frames=None, patch_size=(2, 2),
                    vae_spatial=8, vae_temporal=4):
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    h = (latent_h + patch_size[-2] - 1) // patch_size[-2]
    w = (latent_w + patch_size[-1] - 1) // patch_size[-1]
    if num_frames is not None:
        latent_t = (num_frames + vae_temporal - 1) // vae_temporal
        patch_t = patch_size[0] if len(patch_size) == 3 else 1
        t = (latent_t + patch_t - 1) // patch_t
        return t * h * w
    return h * w
```

### 1.4 Analyze the caller

Find the function that calls block.forward() (usually `forward_orig` or `_forward`):
- What does it pass to each block?
- Does it compute modulation/conditioning outside blocks as non-tensor objects?

### 1.4 Analyze the workflow for weight offloading

**CRITICAL**: Read through ALL nodes in the workflow JSON. Look for any node that might unload, offload, or free model weights AFTER the model is loaded. Compiled engines are binary GPU objects — if any node calls `unpatch_model()`, moves the model to CPU, or frees GPU memory, the engines will be destroyed and cannot be restored.

Look for nodes like:
- `FreeModelMemory` / `ModelMemoryFree`
- `UnloadModel`
- Any node with "offload", "free", "unload" in its name
- Any custom node that manipulates model memory

If found — warn the user that these nodes must be removed from the workflow when using compiled engines.

### 1.5 Check block return values

If block.forward() can return `None` in some code paths — ONNX export will fail. Patch to always return tensors.

---

## Phase 2: Pre-compilation Patching

### 2.1 RoPE patching

Custom C++/CUDA RoPE ops cannot be exported to ONNX. Common patterns:

| Pattern | Problem | Fix |
|---------|---------|-----|
| `comfy_kitchen.apply_rope1` | C++ custom op | Replace with pure PyTorch `_apply_rope1` |
| `torch.view_as_complex` | Complex dtype not in ONNX | Replace with real-valued [cos, sin] operations |
| `torch.polar` | Complex construction | Replace with cos/sin computation |

**Critical**: Check HOW the model imports RoPE. If it does `from comfy.ldm.flux.math import apply_rope1` (direct import), patching `flux_math.apply_rope1` alone is NOT enough — the model file has a **local copy** of the reference:

```python
def patch_rope_for_export():
    import comfy.ldm.flux.math as flux_math
    # Also import the MODEL module that uses apply_rope1
    import comfy.ldm.my_model.model as model_module

    orig = flux_math.apply_rope1
    pure_fn = flux_math._apply_rope1

    flux_math.apply_rope1 = pure_fn
    model_module.apply_rope1 = pure_fn  # Patch the local reference too!

    def restore():
        flux_math.apply_rope1 = orig
        model_module.apply_rope1 = orig
    return restore
```

### 2.2 Non-tensor argument handling

Qlip auto-strips `None` and `dict` arguments. You do NOT need to patch for these.

**Do NOT patch block.forward** unless absolutely necessary. If a block receives `timestep_zero_index=None` or `transformer_options={}`, qlip handles it.

Only patch when the block receives **structured non-tensor objects** (dataclasses, tuples with bools) that qlip cannot auto-strip:

| Input type | Qlip handles? | Action needed |
|-----------|---------------|---------------|
| `None` | Yes, auto-stripped | None |
| `dict` | Yes, auto-stripped | None |
| Dataclass | **No** | Stack fields into tensor, patch block + caller |
| Tuple with non-tensor `(cos, sin, bool)` | **No** | Stack tensors, store non-tensor as block attribute |
| Custom Python object | **No** | Expand to raw tensors in caller |

### 2.3 Caller patching (when needed)

If the caller passes non-tensor data to blocks, patch the caller to convert before calling.

Caller patches are applied **AFTER setup_modules** and must also be applied at **inference time** in `nodes/engine_loader.py → _apply_model_patches()`.

### 2.4 Golden rule for writing patches

**Your patches must not introduce new compilation problems.** Before writing any patch, verify it doesn't violate the Compiler Limitations (see below). Common mistakes:

- Writing `self._attribute` reads inside a compiled block → **L11**: value baked as constant
- Using `tensor.item()` or `int(tensor)` inside compiled block → dynamo error
- Introducing `reshape(a * b, -1, c)` with two dynamic dims → **L5**: TRT can't resolve
- Creating new dynamic axes that TRT can't relate → **L6**: symbol mismatch
- Using Python `if/else` on runtime values → branch baked at trace time
- Slicing with runtime Python int → split position frozen

**Self-check before committing a patch**: read through the patched forward and for every line ask: "will dynamo trace this correctly? will TRT handle the shapes?" If in doubt — move the logic to the caller patch (Python runtime) instead.

### 2.5 Fused QKV + LoRA

If model has fused `attention.qkv` but LoRA has `attention.to_q/to_k/to_v` — unfuse into separate Linear layers before LoRA setup. This also improves FP8 quality (separate scales per projection).

---

## Compiler Limitations Checklist

Before writing ANY compilation code, check the block source for these constraints. Each one will cause a compilation failure if not addressed.

### L1: No custom C++/CUDA ops in the traced forward path

ONNX export fails on any op not in the standard PyTorch/ONNX registry.

**Check**: `grep -r 'comfy_kitchen\|torch.ops\.' <model_file>.py`

**Fix**: Replace with pure PyTorch equivalent temporarily during export. Also patch local references if the model does `from module import func`.

### L2: No complex dtypes

`torch.view_as_complex`, `torch.polar`, `torch.view_as_real` — none have ONNX support.

**Check**: `grep -r 'view_as_complex\|torch.polar\|view_as_real' <model_file>.py`

**Fix**: Replace with real-valued cos/sin operations.

### L3: No non-tensor inputs to compiled blocks

Qlip auto-strips `None` and `dict`. Everything else must be a tensor.

**Check**: Read `block.forward()` signature. Any dataclass, named tuple, Python object, or tuple containing non-tensors?

**Fix**: Stack tensor parts into one tensor, store non-tensor parts as block attributes set by caller.

### L4: No conditional None returns from compiled blocks

ONNX requires deterministic output types. A block that returns `None` in some branches will fail.

**Check**: Read `block.forward()` — any `return None` or conditional `return`?

**Fix**: Patch to always return tensor (zeros if needed).

### L5: No reshape with two unknown dimensions

`reshape(batch * X, -1, dim)` where both `batch` and `X` are dynamic → TRT cannot resolve.

**Check**: `grep -r 'reshape\|\.view(' <block_file>.py` — look for products of dimensions.

**Fix**: Fix batch at export time. Compile separate engines per batch size.

### L6: Dynamic axes symbols are independent

TRT cannot prove that `symbol_a + symbol_b == symbol_c`. If a block concatenates two inputs and uses the result with a third input that has a separate symbol → dimension mismatch.

**Check**: Does block.forward() do `cat(input_A, input_B)` and then use the result in attention/matmul with `input_C`? If all three have different dynamic dims → FAIL.

**Fix**: Restructure — move concat to caller, pass single concatenated tensor to block. Then `joint` and `PE` share one symbol.

### L7: io_dtype must match actual data types

If PE is FP32 but gets multiplied with BF16 data inside block → might be reject type mismatch.

**Check**: Does the caller cast PE to `x.dtype` before passing to blocks? Or does PE stay FP32?

**Fix**: Only use `io_dtype_per_tensor={"pe": "FP32"}` if PE actually stays FP32 through the block. If caller casts PE to BF16 → do NOT override.

### L8: LoRA lora_packed must be FIRST positional argument

`QlipLoraModule.setup()` prepends `lora_packed` to block.forward() signature. If you patch block.forward() AFTER LoRA setup → LoRA signature lost.

**Check**: Is block patching happening after `QlipLoraModule.setup()`?

**Fix**: Block patches BEFORE LoRA. Or avoid patching block.forward() entirely — let qlip auto-strip None/dict args.

### L9: Quantization calibration runs unpatched model

Static FP8 calibration calls `forward_orig()` which doesn't pass `lora_packed`. If LoRA setup already patched signatures → calibration fails.

**Check**: Is LoRA setup before quantization?

**Fix**: Quantization FIRST, then LoRA, then block patches.

### L10: Block signature must match at inference too

If block.forward() was patched at compile time (e.g., `(hidden_states, encoder_hidden_states, ...) → (joint_hidden_states, temb, image_rotary_emb)`), the engine stores the **patched** input names. At inference, the qlip adapter reads `block.model.forward` signature to map inputs. If it sees the **original** signature, it won't recognize engine inputs and will try to inject them as extra inputs.

**Fix**: Create a **separate signature patch function** that runs BEFORE `auto_setup()`. This must be a wrapper function on `block.model.forward` (not just `__signature__` on a bound method — that raises `AttributeError`):

```python
import inspect

sig = inspect.Signature([
    inspect.Parameter("joint_hidden_states", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("temb", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("image_rotary_emb", inspect.Parameter.POSITIONAL_OR_KEYWORD),
])

for block in transformer.transformer_blocks:
    target = block.model if hasattr(block, "model") else block
    orig_fwd = target.forward

    def _make_wrapper(orig):
        def wrapper(joint_hidden_states, temb, image_rotary_emb):
            return orig(joint_hidden_states, temb, image_rotary_emb)
        wrapper.__signature__ = sig
        return wrapper

    target.forward = _make_wrapper(orig_fwd)
```

**Important implementation details**:
- Split into TWO functions: signature patch (before `auto_setup`) and caller patch (after `auto_setup`)
- Cannot set `__signature__` on bound methods directly — must create a new function with `__signature__` attribute
- Caller patch function must `import types` locally (don't rely on module-level import if the function was split from another)

### L11: Python values inside compiled blocks are baked as constants

Dynamo bakes **any Python int/float/bool** read inside a compiled block as a constant. This includes values from `self._attribute`, closure variables, and function arguments that resolve to Python scalars.

**How to detect**: Check if the block patch reads **any** Python scalar that could change between runs. Examples (variable names will differ per model):
- A split position between two sequence types (e.g., text vs image boundary)
- A boundary index for reference/conditional tokens
- `getattr(block, "_any_attribute", default)` where the attribute is set by the caller
- Any slice `tensor[:, :python_int]` where the int varies at runtime

**Key insight**: `tensor[:, :n]` where `n` is Python int → baked. `tensor[:, :n]` where `n` is tensor → dynamo tries `item()` → `GuardOnDataDependentSymNode` error. **Neither approach works for truly dynamic values inside compiled blocks.**

**Solutions** (in order of preference):

**1. Pad inputs to fixed size in the caller** (recommended when one input has variable length):
```python
# In caller patch (Python runtime, not compiled):
fixed_len = 1536  # compile-time constant, set via CLI arg
actual = variable_input.shape[1]
if actual < fixed_len:
    pad = torch.zeros(batch, fixed_len - actual, dim, ...)
    variable_input = torch.cat([variable_input, pad], dim=1)
elif actual > fixed_len:
    variable_input = variable_input[:, :fixed_len]

# Now the split position inside the engine is ALWAYS fixed_len
# Engine bakes this value — but it's always correct
```
The caller handles variable-length inputs before the engine sees them. The engine always works with fixed shapes. After the block, caller splits output using the original (unpadded) length.

**2. Move split/concat entirely to caller** (if the block can process joint tensor without splitting):
Only works if the block doesn't need to know the split position internally. If it applies different operations to different parts — this doesn't work.

**3. Accept the baked value as a limitation** and document it:
If the value rarely changes in practice (e.g., txt_len is usually similar), baking the calibration value may be acceptable with a warning.

**How to find these values**: Search the block patch for:
- `getattr(blk, "_` or `self._` — any attribute read
- `[:, :variable]` or `[:, variable:]` — any slice with non-constant
- `.chunk(n, ...)` where `n` could vary
- `if python_bool:` — conditional branches are also baked

**Important**: Before debugging, verify the issue with **correct model weights** in eager mode. A wrong model file produces identical symptoms (noise borders, blur).

### L12: Caller patch must use keyword arguments when calling compiled blocks

### L12: Caller patch must use keyword arguments when calling compiled blocks

At inference time, compiled blocks are wrapped by `QlipLoraModule` which prepends `lora_packed` as first positional arg. If the caller passes other args as positional too, the engine adapter can't match them by name → injects duplicate inputs or fails with "missing argument".

```python
# WRONG — all positional, engine adapter can't identify args
joint_out = block(joint_hidden_states, temb, image_rotary_emb)

# CORRECT — kwargs, engine adapter matches by name
joint_out = block(joint_hidden_states=joint_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
```

This applies to ALL caller patches that call compiled blocks at inference time.

### L11: Text length varies with tokenizer

Don't hardcode `txt_len`. Derive from calibration:
```python
calib_seq_len = cm.modules[0].ioconfig._current_shapes[0]["inputs"]["x"][1]
txt_len = calib_seq_len - calc_img_tokens(calib_w, calib_h)
```

### L11: Audio/video models need ALL modalities in calibration

If model has audio+video streams, calibration MUST include audio latents. Without audio → audio branch not traced → LoRA layers missing from ONNX.

### L12: ALWAYS add a dynamic range profile

Static profiles (min==opt==max) only match exact shapes. At inference, the actual seq_len depends on the input image size, which may not exactly match any compiled resolution. Without a dynamic range profile, inference will fail with "no optimization profile defined for the given input shapes".

**Rule**: Always create static profiles for known resolutions PLUS one dynamic range profile that covers the full min-to-max range:

```python
# Static profiles: exact shapes, best perf for these resolutions
for w, h in target_sizes:
    tokens = calc_tokens(w, h)
    trt_profiles.append({
        "min": {"seq_len": tokens},
        "opt": {"seq_len": tokens},
        "max": {"seq_len": tokens},
    })

# MANDATORY: dynamic range profile — catches everything in between
all_seq = sorted(set(calc_tokens(w, h) for w, h in target_sizes))
trt_profiles.append({
    "min": {"seq_len": all_seq[0]},
    "opt": {"seq_len": all_seq[len(all_seq)//2]},
    "max": {"seq_len": all_seq[-1]},
})
```

**Also consider**: reference images, text tokens, and other factors that increase seq_len beyond what target_sizes alone would produce. If the model supports reference/conditioning images that add tokens, the max profile must account for that:
```python
max_img_tokens = calc_tokens(max_w, max_h)
max_ref_tokens = max_img_tokens * max_ref_images
max_total = max_img_tokens + max_ref_tokens + txt_len
```

### Quick pre-compilation check script

Run this on the model block before writing compilation code:

```python
import inspect, torch

block = transformer.transformer_blocks[0]
sig = inspect.signature(block.forward)
print("=== Block forward signature ===")
for name, param in sig.parameters.items():
    print(f"  {name}: {param.default}")

print("\n=== Ops check ===")
source = inspect.getsource(type(block))
for pattern in ['comfy_kitchen', 'view_as_complex', 'torch.polar', 'view_as_real']:
    if pattern in source:
        print(f"  WARNING: {pattern} found — needs patching")

for pattern in ['reshape', '.view(']:
    count = source.count(pattern)
    if count > 0:
        print(f"  INFO: {count}x {pattern} found — check for batch*seq ambiguity")

if 'return None' in source:
    print("  WARNING: conditional None return found — needs patching")
```

---

## Phase 3: Debugging & Isolated Tests (use when something fails)

### 3.1 Test single block forward pass

Before any compilation, verify that a single block runs correctly with tensor-only inputs:

```python
# Step 1: Run block with original forward (all args including None/dict)
block = transformer.transformer_blocks[0]
block.to("cuda")
# ... run calibration inference once to get real activations

# Step 2: Run block with tensor-only args (as qlip will see it)
# Only pass args that are actual tensors — skip None, dict, dataclass
with torch.no_grad():
    output = block(hidden_states, encoder_hidden_states, temb=temb, image_rotary_emb=pe)
    print(f"Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output shapes: {[o.shape for o in output]}")
    else:
        print(f"Output shape: {output.shape}")
```

If this fails — you have a patching problem. Fix before proceeding.

### 3.2 Test torch.export / dynamo tracing

Test that the block can be traced by dynamo (what qlip uses internally):

```python
import torch

block = transformer.transformer_blocks[0].to("cuda")
# Apply patches + LoRA first (same as compilation)

# Test dynamo trace
try:
    exported = torch.export.export(
        block,
        args=(hidden_states, encoder_hidden_states),
        kwargs={"temb": temb, "image_rotary_emb": pe},
    )
    print("torch.export succeeded!")
    print(f"Input names: {[spec.arg.name for spec in exported.graph_signature.input_specs if spec.kind.name == 'USER_INPUT']}")
except Exception as e:
    print(f"torch.export failed: {e}")
    # This error message tells you exactly what to fix
```

Common failures and what they mean:
- `No ONNX function found for <op>` → custom C++ op needs pure PyTorch replacement
- `view_as_complex` → complex dtype, needs real-valued replacement
- `dynamic_shapes key mismatch` → you have a None/dict arg in dynamic_axes
- `non-default argument follows default` → signature issue from patching

### 3.3 Test compilation of single block

Before compiling all 60 blocks, compile just block 0:

```python
cm = NvidiaCompileManager(transformer, workspace=engines_dir)
cm.setup_modules(modules=["transformer_blocks.0"], builder_config=nvidia_config, dtype=torch.bfloat16)

# Set profiles for just this one module
for module in cm.modules:
    module.ioconfig.set_axes_profiles(dynamic_axes, profiles)

# Compile single block
restore_rope = patch_rope_for_export()
cm.compile(cpu_offload=True, dynamo=True, keep_compiled=False)
restore_rope()
print("Single block compiled successfully!")
```

If single block compiles — all blocks will compile (same architecture). If it fails — fix the error before wasting time on all blocks.

### 3.4 Verify compiled block produces correct output

After compiling block 0, run inference and compare with PyTorch:

```python
# Run compiled block
compiled_output = compiled_block(hidden_states, encoder_hidden_states, temb=temb, image_rotary_emb=pe)

# Compare with PyTorch reference
with torch.no_grad():
    ref_output = original_block(hidden_states, encoder_hidden_states, temb=temb, image_rotary_emb=pe)

diff = (compiled_output - ref_output).abs().max().item()
print(f"Max diff: {diff}")  # Should be < 0.01 for BF16
```

### 3.2 Common dynamo/export issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Custom C++ op | `DispatchError: No ONNX function found` | Patch with pure PyTorch equivalent |
| Complex dtype | `view_as_complex` error | Replace with real-valued ops |
| `dynamic_shapes` key mismatch | `top-level keys must be the arg names [...]` | Only include tensor args that survive after calibration (not None/dict args) |
| `view(batch*seq, hidden)` ambiguity | Two symbolic dims in reshape | Fix batch size at export time (see below) |
| `reshape(b*heads, -1, dim)` in attention | `mismatching dimensions` in MatMul/SDPA | Same fix — batch must be fixed, not dynamic |
| Block returns None | ONNX cannot handle optional outputs | Patch to always return tensor |
| `from module import func` then patching module | Patch doesn't take effect | Also patch the local reference in importing module |
| RoPE `_apply_rope1` type mismatch | `Float * BFloat16` error in TRT | Cast freqs_cis to x.dtype instead of x to freqs_cis.dtype (see RoPE section) |
| `io_dtype_per_tensor` causing type mismatch | `ElementWiseOperation PROD must have same types` | Check if model casts PE to BF16 before blocks — if yes, do NOT set PE to FP32 |
| Mixed FP32/BF16 inputs in STRONGLY_TYPED | `ElementWiseOperation SUM/PROD must have same input types (Half/BFloat16 and Float)` | See **io_dtype debugging** section below |

### 3.3 Dynamic axes — the most critical decision

TRT dynamic axes are the #1 source of compilation failures. Three rules:

1. **Same symbol = same runtime value.** Two inputs with same symbol must always have equal size.
2. **Different symbols = unrelated.** TRT cannot prove `symbol_a + symbol_b == symbol_c`.
3. **reshape(a * b, -1, c)** — if both `a` and `b` are dynamic, TRT can't resolve `-1`.

#### Can batch_size be dynamic?

Search block code for `reshape`/`view` with batch × something:
```python
x.reshape(batch * heads, -1, dim_head)    # Two unknowns → NO
x.view(batch * seq_len, hidden)           # Two unknowns → NO
```
If found → batch CANNOT be dynamic. Compile separate engines per batch size.

#### How many sequence inputs does the block have?

**Single-stream** — one `x` input, one `pe` input, same seq_len:
```
block(x, pe, ...)       # x.shape[1] == pe.shape[2] == seq_len
```
→ One symbol for both. Dynamic seq_len works:
```python
dynamic_axes = {"x": {1: "seq_len"}, "pe": {2: "seq_len"}, "lora_packed": {1: "lora_rank"}}
```
Examples: Z-Image-Turbo (`x` + `freqs_cis`).

**Dual-stream with internal concat** — two sequence inputs concatenated inside block, result used with PE:
```
block(img, txt, pe, ...)   # inside: joint = cat(img, txt); attention(Q, K, PE)
```
→ TRT CANNOT handle this: `img_seq + txt_seq == pe_seq` is unprovable.

**Solution: restructure the block signature.** Move concat to the caller:

1. **Block patch**: Change signature from `(img, txt, pe, ...)` to `(joint, pe, ...)` where `joint = cat(txt, img)` is already concatenated. Inside the patched block: split back to `txt, img` using a stored attribute `_num_txt_tokens`, call original forward, concat output back.

2. **Caller patch**: Before each block call, concat `txt + img` → `joint`. Set `block._num_txt_tokens = txt.shape[1]` and other needed attributes. After block call, split output back.

3. **Dynamic axes**: Now `joint` dim1 == `pe` dim2 == `seq_len` → one symbol works:
```python
dynamic_axes = {
    "joint_hidden_states": {1: "seq_len"},
    "image_rotary_emb": {2: "seq_len"},
    "lora_packed": {1: "lora_rank"},
}
```

Attributes stored on block for the patched forward to read (via `self._attr`): number of text tokens, attention mask, any non-tensor state from the caller.

Example: FLUX Klein uses a similar approach — modulation tensors are stacked into one tensor in the caller before passing to blocks.

**Dual-stream with separate attention** — two sequences processed independently (no internal concat):
```
block(img, txt, ...)   # separate img_attn and txt_attn, no joint attention
```
→ Separate symbols work:
```python
dynamic_axes = {"img": {1: "img_seq"}, "txt": {1: "txt_seq"}}
```
Example: LTX-2 (video + audio processed separately).

#### Quick reference

| Block type | batch dynamic? | seq_len approach | Example |
|---|---|---|---|
| Single-stream, no reshape(b*x) | Yes | One symbol, dynamic | Z-Image-Turbo |
| Single-stream, has reshape(b*heads) | **No** | One symbol, dynamic | — |
| Dual-stream, concat inside block | **No** | **Restructure**: concat in caller, one symbol | Models with joint attention |
| Dual-stream, no concat | Depends | Separate symbols, both dynamic | LTX-2 |
| Dual-stream, modulation outside | **No** | Stack tensors in caller | FLUX Klein |

---

## Phase 4: BF16 Compilation

### 4.1 Pipeline order

**Order is critical. Do not reorder.**

```
1. Load model (ComfyUI API, BF16/FP16 weights — NEVER FP8, even if workflow uses FP8)
2. (Optional) Unfuse QKV / apply block patches
3. Setup LoRA (if needed)
4. Setup compiler (NvidiaCompileManager.setup_modules)
5. (Optional) Caller patches (AFTER setup_modules)
6. Calibration inference (shape collection)
7. Set dynamic axes + shape profiles
8. Patch RoPE for export (temporary)
9. Compile
10. Restore RoPE
11. Save lora_config.json
```

### 4.2 Model loading

**ComfyUI `folder_paths` functions take filenames only, not paths.** For absolute paths, use directly.

```python
import comfy.sd
import comfy.model_management

# UNETLoader:
model_patcher = comfy.sd.load_diffusion_model(unet_path, model_options={})
transformer = model_patcher.model.diffusion_model

# Checkpoint (model+VAE+CLIP):
model_patcher, clip, vae = comfy.sd.load_checkpoint_guess_config(ckpt_path)

# Model stays on CPU after loading. Move to GPU explicitly before calibration:
transformer.to("cuda")
for module in cm.modules:
    if hasattr(module, "model"):
        module.model.to("cuda")
# Move back to CPU before compilation (cpu_offload compiles block-by-block):
transformer.to("cpu")
```

**Do NOT use `transformer.to("cuda", dtype=torch.bfloat16)`** — this casts ALL weights to BF16, but some models have layers that expect FP32 weights (e.g. `patch_embedding` Conv3d where `forward_orig` calls `.float()` on the input).

**Do NOT use `comfy.model_management.load_models_gpu()`** for compilation scripts — it may OOM on large models (14B+) and conflicts with the compile pipeline's memory management. Use explicit `.to("cuda")` / `.to("cpu")` instead.

### 4.3 LoRA setup

`QlipLoraModule.setup()` adds `lora_packed` as the **FIRST positional argument** to block.forward automatically.

**Try auto-detection first:**
```python
from qlip.lora_support import LoRAManager, QlipLoraModule
configs = LoRAManager.infer_config(lora_path)
```

**If `infer_config` returns `[]`** — build config manually. Ask the user to run diagnostic scripts:

**Script 1 — LoRA structure** (`check_lora.py`):
```python
from safetensors import safe_open
import sys
with safe_open(sys.argv[1], framework="pt") as f:
    keys = list(f.keys())
block_keys = [k for k in keys if "." in k]
for prefix in set(".".join(k.split(".")[:1]) for k in block_keys):
    pkeys = [k for k in block_keys if k.startswith(prefix + ".")]
    indices = set(int(k.split(".")[1]) for k in pkeys if k.split(".")[1].isdigit())
    if indices:
        print(f"{prefix}: {max(indices)+1} blocks")
    layers = set()
    for k in pkeys:
        parts = k.split(".")
        layer = ".".join(parts[2:-1]).replace(".lora_down", "").replace(".lora_up", "").replace(".alpha", "")
        if layer: layers.add(layer)
    for l in sorted(layers):
        print(f"  {l}")
```

**Script 2 — Model layer sizes** (`check_model.py`):
```python
import torch, sys
sys.path.insert(0, "./ComfyUI")
import comfy.sd, folder_paths
model_patcher = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", sys.argv[1]))
dm = model_patcher.model.diffusion_model
for attr in ["transformer_blocks", "blocks", "layers", "double_blocks"]:
    blocks = getattr(dm, attr, None)
    if blocks and len(blocks) > 0:
        print(f"Block attr: {attr}, count: {len(blocks)}")
        for name, mod in blocks[0].named_modules():
            if isinstance(mod, torch.nn.Linear):
                print(f"  {name}: in={mod.in_features}, out={mod.out_features}")
        break
```

**Build config manually:**
```python
from qlip.lora_support import LoRAConfig, LayerConfig

lora_config = LoRAConfig(
    name="transformer_blocks",
    block_prefix="transformer_blocks",    # from check_model.py
    num_blocks=60,                        # from check_lora.py
    max_features=12288,                   # max(in, out) across all LoRA layers
    layers=[
        LayerConfig("attn.to_q", out_features=3072, in_features=3072),
        # ... order must match execution order in block.forward()
    ],
)
```

**Then setup:**
```python
for c in configs:
    blocks = getattr(transformer, c.block_prefix)
    manager = LoRAManager(c, device="cuda", dtype=torch.bfloat16)
    manager.load_from_safetensors(lora_path, strength=1.0)
    packed = [manager.pack_block(f"{c.block_prefix}.{i}", max_rank)
              for i in range(len(blocks))]
    for pt in packed:
        pt.uniform_(-0.01, 0.01)  # Random noise, NOT zeros
    QlipLoraModule.setup(transformer, c.block_prefix, c, packed, max_rank=max_rank)
```

### 4.4 Compiler setup

```python
from qlip.compiler.nvidia import NvidiaBuilderConfig, NvidiaCompileManager

nvidia_config = NvidiaBuilderConfig(
    builder_flags={"BF16", "WEIGHT_STREAMING"},
    io_dtype="base",
    # io_dtype_per_tensor — see note below
    creation_flags={"STRONGLY_TYPED"},
    profiling_verbosity="DETAILED",
)

cm = NvidiaCompileManager(transformer, workspace=engines_dir)
# By type:
cm.setup_modules(module_types=(MyBlockClass,), builder_config=nvidia_config, dtype=torch.bfloat16)
# Or by name (when different block types have different signatures):
cm.setup_modules(modules=[f"blocks.{i}" for i in range(n)], builder_config=nvidia_config, dtype=torch.bfloat16)
```

**`io_dtype_per_tensor` for RoPE/PE tensors — best practice:**

PE (positional embeddings / RoPE frequencies) should stay in **FP32 for precision**. Since we write the caller patch anyway, we control how PE is passed to blocks.

**Recommended approach**: In your caller patch, remove `.to(x.dtype)` from PE creation so it stays FP32. Set `io_dtype_per_tensor={"pe_name": "FP32"}` in the builder config. Ensure the RoPE function inside the block casts PE to match data dtype before multiplication:

```python
# In caller patch — keep PE in FP32:
image_rotary_emb = self.pe_embedder(ids).contiguous()  # NO .to(x.dtype)

# In compile script builder config:
io_dtype_per_tensor={"image_rotary_emb": "FP32"}

# In RoPE patch — cast PE to data dtype before multiply:
def _trt_apply_rope1(x, freqs_cis):
    freqs_cis = freqs_cis.to(dtype=x.dtype)  # FP32 → BF16 cast here
    ...
```

| Situation | `io_dtype_per_tensor` | Action |
|---|---|---|
| You write caller patch (recommended) | **Yes** `{"pe": "FP32"}` | Remove `.to(x.dtype)` in caller, PE stays FP32 |
| No caller patch, model keeps PE FP32 | **Yes** `{"pe": "FP32"}` | PE already FP32, just declare it |
| No caller patch, model casts PE to BF16 | **No** | PE is BF16, FP32 override causes mismatch |

Examples:
- **Flux Klein**: `{"pe": "FP32"}` — PE stays FP32
- **Z-Image-Turbo**: `{"freqs_cis": "FP32"}` — freqs_cis stays FP32
- **LTX-2**: none — PE stacked into block input, no separate tensor
- **Wan2.2**: `{"freqs": "FP32"}` — RoPE freqs stays FP32, but `e` (modulation) is NOT FP32 (see below)

### io_dtype debugging: `ElementWiseOperation must have same input types`

**How to diagnose:**

1. Read block.forward() and trace every input tensor through the computation
2. For each input, determine: does it stay in its original dtype, or does the block cast it?
3. The rule: **`io_dtype_per_tensor` must match what the ONNX graph expects, NOT what the caller sends**

**The key question for each input:** Does the block USE it in its original dtype, or does it CAST it first?

| Input arrives as | Block does | io_dtype_per_tensor | Why |
|---|---|---|---|
| FP32 | Uses directly with BF16 data (`e + x`) | Do NOT set FP32 — let engine cast to BF16 | Block mixes it with BF16 → ONNX graph has BF16 ops |
| FP32 | Casts internally (`freqs.to(x.dtype)`) | Set `"FP32"` | ONNX graph has explicit Cast node, TRT handles it |
| FP32 | Uses only with other FP32 tensors | Set `"FP32"` | No dtype conflict |
| BF16 | Uses directly | Default (no override needed) | Already matches |

**Common mistake:** Setting `io_dtype_per_tensor={"e": "FP32"}` because `e` arrives as FP32 from the caller. But if the block does `modulation.to(x.dtype) + e` (Wan2.2 pattern), the ONNX graph contains a BF16 Add — the FP32 `e` gets promoted to BF16 during tracing. Declaring it FP32 in io_dtype forces TRT to keep it FP32 → mismatch with the BF16 Add node.

**Correct approach for Wan2.2 Animate:**
```python
nvidia_config = NvidiaBuilderConfig(
    builder_flags=builder_flags,
    io_dtype="base",                           # all inputs default to model dtype (BF16)
    io_dtype_per_tensor={"freqs": "FP32"},     # only freqs stays FP32 (apply_rope1 casts internally)
    # e is NOT FP32 — block does cast_to(modulation, x.dtype) + e → all BF16 in ONNX
)
```

**Step-by-step debugging process:**

1. First try `io_dtype="base"` with NO `io_dtype_per_tensor` overrides
2. If compilation succeeds → done
3. If `ElementWiseOperation must have same types` error → identify which node fails (e.g. `node_mul_2`)
4. Read block.forward() to find which input is involved in that operation
5. Check: does the block cast that input before using it with other tensors?
   - YES (e.g. `apply_rope1` casts freqs) → add `{"input_name": "FP32"}` to keep it FP32, the Cast node in ONNX handles conversion
   - NO (e.g. `modulation + e` without explicit cast) → do NOT override, let `"base"` cast it to BF16 at engine boundary

### 4.5 Calibration & shape profiles

**Shape collection** runs inference to record tensor shapes for TRT optimization profiles. Run it for **every target resolution** — TRT optimizes kernel selection per profile.

```python
# 1. Enable shape collection
for module in cm.modules:
    module._collect_shapes = True

# 2. Run calibration for ONE size (smallest — saves memory)
#    Shape collection captures tensor names, batch dims, hidden dims etc.
#    The actual seq_len ranges come from set_axes_profiles, NOT from collected shapes.
calib_w, calib_h = min(parsed_sizes, key=lambda s: s[0] * s[1])
run_calibration_inference(model_patcher, clip, prompt, calib_w, calib_h, steps=2)

# 3. CRITICAL: set_axes_profiles and _collect_shapes=False in ONE loop per module.
#    Do NOT disable collection in a separate loop before setting profiles —
#    qlip will use collected shapes and ignore your profiles.
for module in cm.modules:
    module._collect_shapes = False
    module.ioconfig.set_axes_profiles(dynamic_axes, profiles)
```

**Dynamic axes — only include tensor args that survive calibration:**
```python
dynamic_axes = {
    "hidden_states": {0: "batch_size", 1: "seq_len"},
    "lora_packed": {1: "lora_rank"},
    # Do NOT add args that receive None/dict during calibration
}
```

Check the log after calibration: `Prepare export with inputs: [...]` — only those names belong in dynamic_axes.

**Shape profiles — MUST have real variation:**

Always provide **multiple resolutions** (`--sizes`) AND a **dynamic range** (`--dynamic-size-range`). A single resolution produces identical min/opt/max profiles — TRT cannot handle any other seq_len at runtime.

```bash
# BAD — single resolution, no dynamic range:
--sizes 480x832
# Result: static profile seq_len=32760, dynamic profile seq_len=32760 (same!)

# GOOD — multiple resolutions + dynamic range:
--sizes 480x832 480x480 832x480
--dynamic-size-range 320 480 832
# Result: 3 static profiles + 1 dynamic profile covering 320→832 range
```

**Profile construction pattern:**
```python
profiles = []

# Static profiles: one per target resolution (best perf at exact size)
for size in target_sizes:
    seq = compute_seq_len(size)
    profiles.append({"min": {"seq_len": seq}, "opt": {"seq_len": seq}, "max": {"seq_len": seq}})

# MANDATORY: dynamic range profile (handles arbitrary sizes within range)
profiles.append({
    "min": {"seq_len": min_seq},    # smallest supported
    "opt": {"seq_len": opt_seq},    # most common
    "max": {"seq_len": max_seq},    # largest supported
})
```

**Without the dynamic range profile**, the engine can ONLY run at the exact static profile sizes. Any other resolution → TRT error `dimensions not in any optimization profile`.

**FP8 calibration** is separate from shape collection and needs more iterations with diverse prompts:
```python
# FP8 calibration: N iterations, varied prompts, varied resolutions
from datasets import load_dataset
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
for idx in range(calibration_iterations):  # default 10
    w, h = sizes[idx % len(sizes)]
    prompt = dataset[idx]["Prompt"]
    run_calibration_inference(model_patcher, clip, prompt, w, h, steps=2)
```

2 steps with 1 prompt is NOT enough for FP8 — activation ranges vary significantly across prompts. Use at least 10 iterations with different prompts from the dataset.

### 4.6 Calibration inference template

**Use this template for `run_calibration_inference`.** Uses `comfy.sample.sample()` — the same code path as ComfyUI's KSampler node. Do NOT use `CFGGuider` + `KSampler` manually — the API is fragile and changes between versions.

```python
def run_calibration_inference(model_patcher, clip, prompt, width, height,
                              steps=2, cfg=4.0, num_frames=None, seed=42):
    """Run short inference to collect shapes for TRT profiles.

    Works for both image and video models.
    """
    import comfy.sample
    import comfy.model_management

    # --- Conditioning ---
    # encode_from_tokens_scheduled returns a SINGLE value (list), NOT a tuple.
    # Do NOT unpack as (cond, _) — it's one return value.
    tokens_pos = clip.tokenize(prompt)
    conditioning_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize("")
    conditioning_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # --- Latent ---
    # Image model: [batch, 4, height//8, width//8]
    # Video model: [batch, 16, frames//4, height//8, width//8]
    latent_h = height // 8
    latent_w = width // 8
    if num_frames is not None:
        latent_frames = (num_frames + 3) // 4
        latent_image = torch.zeros(1, 16, latent_frames, latent_h, latent_w,
                                   device="cuda")
    else:
        latent_image = torch.zeros(1, 4, latent_h, latent_w,
                                   device="cuda")

    # fix_empty_latent_channels adjusts channels to match model
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_patcher, latent_image,
    ).to("cuda")

    noise = comfy.sample.prepare_noise(latent_image, seed)

    # comfy.sample.sample handles model GPU loading internally.
    # Do NOT call load_models_gpu separately — may OOM on large models (14B+).

    # --- Sample using ComfyUI's standard path ---
    _ = comfy.sample.sample(
        model_patcher,
        noise,
        steps,
        cfg,
        "euler",       # sampler
        "normal",      # scheduler
        conditioning_pos,
        conditioning_neg,
        latent_image,
        denoise=1.0,
        disable_noise=False,
        seed=seed,
    )
```

**Common mistakes to avoid:**
- `clip.encode_from_tokens_scheduled()` returns **one value** (list), NOT a tuple. Do NOT write `cond, _ = clip.encode_from_tokens_scheduled(...)`.
- Do NOT use `comfy.samplers.KSampler("euler", {}, device="cuda")` directly — `KSampler.__init__` expects a model object, not a string. Use `comfy.sample.sample()` instead.
- Do NOT use `comfy.samplers.CFGGuider` + `guider.sample()` — use `comfy.sample.sample()` which handles CFG internally.
- For video models, latent channels are typically 16 (not 4) and shape includes temporal dimension.
- Always call `comfy.sample.fix_empty_latent_channels()` to auto-adjust channels for the model.

### 4.7 Calibration pitfalls

**Conditioning tensor dimensions**: ComfyUI's CLIP encode may return 2D tensors `(seq, dim)` without batch dimension. The model expects 3D `(batch, seq, dim)`. Always check and add batch dim if needed:
```python
cond_tensor = clip_encode_result[0][0][0]
if cond_tensor.ndim == 2:
    cond_tensor = cond_tensor.unsqueeze(0)  # Add batch dim
```

**Positive and negative conditioning must follow same code path**: If positive cond has `reference_latents` (triggers special timestep handling, different tensor shapes), negative cond should too. Otherwise positive pass creates different shapes than negative pass, and TRT shape collection gets inconsistent data.

```python
# WRONG — different paths for pos/neg
positive = [[pos_cond, {"reference_latents": [ref_latent]}]]
negative = [[neg_cond, {}]]  # ← no ref_latents → different shapes in block

# CORRECT — same structure
positive = [[pos_cond, {"reference_latents": [ref_latent]}]]
negative = [[neg_cond, {"reference_latents": [ref_latent]}]]  # ← same path
```

**CFG > 1 doubles batch**: If cfg > 1.0, ComfyUI runs both positive and negative conditioning, doubling the effective batch. If compiling for cfg=1.0, set `cfg=1.0` in calibration to avoid batch=2 shapes being collected.

### 4.7 Compile

```python
restore_rope = patch_rope_for_export()
transformer.to("cpu")
torch.cuda.empty_cache()

cm.compile(cpu_offload=True, dynamo=True, keep_compiled=False)
restore_rope()
```

**`cm.compile()` parameters:**
- `cpu_offload=True` — move model to CPU during compilation to save GPU memory
- `dynamo=True` — use `torch._dynamo` for tracing (required for most models)
- `keep_compiled=False` — don't keep intermediate compiled objects in memory
- `recompile_existing=False` — skip blocks that already have `.engine`/`.qlip` files
- `dump_onnx=False` — set `True` to save intermediate ONNX files for debugging

### 4.8 Save LoRA config

After compilation, save `lora_config.json` in the engines directory. This file is auto-loaded by ComfyUI-Qlip nodes at inference time to configure LoRA injection.

```python
import json

def save_lora_config_json(path, configs):
    """Save LoRA configs to JSON.

    Args:
        path: Output path (e.g., engines_dir / "lora_config.json")
        configs: List of LoRAConfig objects from LoRA setup
    """
    entries = []
    for cfg in configs:
        entries.append({
            "name": cfg.name,               # e.g., "transformer_blocks"
            "block_prefix": cfg.block_prefix, # e.g., "transformer_blocks"
            "num_blocks": cfg.num_blocks,     # e.g., 60
            "max_features": cfg.max_features, # max(in_features, out_features) across all layers
            "layers": [
                {
                    "name": l.name,           # e.g., "attn.to_q"
                    "out_features": l.out_features,  # Linear output dim
                    "in_features": l.in_features,    # Linear input dim
                }
                for l in cfg.layers
            ],
        })
    with open(path, "w") as f:
        json.dump({"configs": entries}, f, indent=2)
    print(f"Saved LoRA config: {path} ({len(entries)} block group(s))")

# Call after compilation
if lora_configs:
    save_lora_config_json(engines_dir / "lora_config.json", lora_configs)
```

**Format details:**
- `name` and `block_prefix` — must match the attribute name on the transformer (e.g., `transformer.transformer_blocks`)
- `num_blocks` — number of blocks in the group
- `max_features` — the largest `in_features` or `out_features` across all LoRA layers. Determines `lora_packed` tensor width
- `layers` — list of Linear layers that have LoRA. **Order must match execution order** in `block.forward()`. Names are relative to the block (e.g., `attn.to_q`, not `transformer_blocks.0.attn.to_q`)

### 4.9 Key API reference

| Class / Function | Package | Purpose |
|---|---|---|
| `NvidiaCompileManager` | `qlip.compiler.nvidia` | Manages compilation of multiple blocks |
| `NvidiaBuilderConfig` | `qlip.compiler.nvidia` | Builder flags, precision, io_dtype |
| `QuantizationManager` | `qlip.quantization` | FP8/INT8 quantization setup |
| `NVIDIA_FLOAT_W8A8` | `qlip.compiler.nvidia` | Static FP8 quantization config |
| `NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC` | `qlip.compiler.nvidia` | Dynamic FP8 quantization config |
| `LoRAManager` | `qlip.lora_support` | Load/pack LoRA weights |
| `LoRAConfig` | `qlip.lora_support` | LoRA structure definition |
| `LayerConfig` | `qlip.lora_support` | Single layer in LoRA config |
| `QlipLoraModule` | `qlip.lora_support` | Patches blocks for LoRA compilation + inference |
| `LoRAManager.infer_config(path)` | `qlip.lora_support` | Auto-detect LoRA config from file (may fail — see manual config) |
| `cm.setup_modules(module_types=...)` | `qlip.compiler.nvidia` | Find blocks by type and wrap in CompiledModule |
| `cm.setup_modules(modules=[...])` | `qlip.compiler.nvidia` | Find blocks by name and wrap in CompiledModule |
| `module.ioconfig.set_axes_profiles(axes, profiles)` | `qlip.compiler` | Set dynamic axes and shape profiles |
| `cm.compile(...)` | `qlip.compiler.nvidia` | Compile all registered modules to engines |

---

## Phase 5: FP8 Quantization + Compilation

### 5.1 Hardware-dependent quantization

| GPU | Quantization | Config |
|-----|-------------|--------|
| H100, B200, L40S, RTX 5090 (Ada+) | FP8 | `NVIDIA_FLOAT_W8A8` or `NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC` |
| A100, RTX 4090 | INT8 | Use INT8 quantization instead of FP8 |

### 5.2 Quantization setup + calibration (must be together)

**Quantization and calibration must happen together, BEFORE any patches.** Static FP8 calibration runs the unpatched model forward to collect accurate activation statistics. If you patch the model first, calibration sees modified activations → wrong FP8 scales → bad quality.

```python
from qlip.quantization import QuantizationManager
from qlip.compiler.nvidia import NVIDIA_FLOAT_W8A8, NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC

modules = get_quantizable_modules(transformer, skip_first=1, skip_last=1)
qmanager = QuantizationManager()

# Dynamic (no calibration needed):
qmanager.setup_modules(modules, calibration_iterations=0, **NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC)

# Static (calibration runs HERE, immediately after setup):
qmanager.setup_modules(modules, calibration_iterations=10, **NVIDIA_FLOAT_W8A8)
# Run calibration NOW — before any block/caller patches:
for i in range(10):
    w, h = sizes[i % len(sizes)]  # varied resolutions
    prompt = prompts[i]            # varied prompts (use dataset)
    run_calibration_inference(model_patcher, clip, prompt, w, h, steps=2)
```

**Pipeline order with FP8**:
```
1. Load model + text encoder
2. Quantization setup + calibration (unpatched model) ← FP8 scales calibrated here
3. Block patches
4. LoRA setup
5. setup_modules
6. Caller patches
7. Shape collection (one inference through patched model)
8. Compile with "FP8" flag
```

Note: shape collection (step 7) is separate from FP8 calibration (step 2). Shape collection runs through the patched model to record tensor shapes for TRT profiles. FP8 calibration runs through the unpatched model to compute activation scales.

### 5.3 FP8 compilation

Same as Phase 4 but add `"FP8"` to builder flags:
```python
builder_flags = {"BF16", "WEIGHT_STREAMING", "FP8"}
```

### 5.4 FP8 quality debugging (only if quality is bad)

If FP8 compiled output looks worse than BF16 — use these isolated tests to diagnose:
1. `--skip-first-blocks 1 --skip-last-blocks 1` — keep first/last blocks in BF16
2. Unfuse QKV (`--unfuse-qkv`) — separate FP8 scales per Q/K/V projection
3. Try dynamic quantization: `NVIDIA_FLOAT_W8A8_PER_TOKEN_DYNAMIC`
4. Increase `--calibration-iterations` for better scale calibration
5. Run FP8 simulation (fake-quantization in PyTorch without compilation) to check if the issue is in quantization or compilation

---

## Phase 6: Benchmarking

After compilation, tell the user to measure performance using **Qlip Timer nodes** in ComfyUI:

1. Add **Qlip Timer Start** node before the sampler
2. Add **Qlip Timer Stop** node after the sampler
3. Add **Qlip Timer Report** node connected after everything
4. Run the workflow **twice** — first run is warmup (engine loading), second run is the real measurement
5. The timer report shows elapsed time per component

Tell the user to compare:
- **Eager PyTorch** (without Qlip engines) — baseline
- **Qlip BF16 + LoRA** — compilation speedup
- **Qlip FP8 + LoRA** — compilation + quantization speedup

Report: time (seconds, second run), speedup vs baseline.

---

## Phase 7: Inference Integration

When a model needs patches at inference time, add to ComfyUI-Qlip. There are **two types** of patches, applied at different times:

### 7.1 Signature patches (BEFORE engine loading)

If the block signature was changed at compile time (e.g., merged inputs into joint tensor), `auto_setup()` needs to see the patched signature to correctly map engine inputs. These patches run **before** `auto_setup()`.

**`utils/helpers.py`:**
```python
def patch_my_model_block_signature(dm):
    import inspect
    sig = inspect.Signature([
        inspect.Parameter("joint_hidden_states", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("temb", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("image_rotary_emb", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    for block in dm.transformer_blocks:
        target = block.model if hasattr(block, "model") else block
        orig_fwd = target.forward
        def _make_wrapper(orig):
            def wrapper(joint_hidden_states, temb, image_rotary_emb):
                return orig(joint_hidden_states, temb, image_rotary_emb)
            wrapper.__signature__ = sig
            return wrapper
        target.forward = _make_wrapper(orig_fwd)
```

**`nodes/engine_loader.py` → `_apply_signature_patches(dm)`:**
```python
from ..utils import is_my_model, patch_my_model_block_signature
if is_my_model(dm):
    patch_my_model_block_signature(dm)
```

### 7.2 Caller patches (AFTER engine loading)

These modify the transformer's forward to prepare inputs for compiled blocks (concat, pad, set attributes). They run **after** `auto_setup()`.

**`utils/helpers.py`:**
```python
def is_my_model(dm):
    return type(dm).__name__ == "MyModelClass"

def patch_my_model_caller(dm, fixed_txt_len=1536):
    # Patch forward_orig or _forward:
    # - Pad txt to fixed length
    # - Concat txt + img before each block
    # - Set block attributes (mask, timestep_zero_index, etc.)
    # - Split output back after each block
    ...
```

**`nodes/engine_loader.py` → `_apply_caller_patches(dm)`:**
```python
from ..utils import is_my_model, patch_my_model_caller
if is_my_model(dm):
    patch_my_model_caller(dm)
```

### 7.3 Execution order in engine_loader.py

```python
# 1. Signature patches — BEFORE auto_setup
self._apply_signature_patches(dm)

# 2. Load engines — auto_setup reads patched signatures
imanager = NvidiaInferenceManager(model=dm, workspace=engines_dir)
imanager.auto_setup()

# 3. Caller patches — AFTER auto_setup
self._apply_caller_patches(dm)

# 4. LoRA wrapper — AFTER caller patches
QlipLoraModule.setup(dm, ...)
```

---

## Model Download Script Pattern

**CRITICAL: Always download the BF16/FP16 version of the model checkpoint, regardless of what the workflow JSON uses.** Many workflows reference FP8 checkpoints for memory savings during eager inference — but for Qlip compilation we MUST start from full-precision weights. The download script should fetch BF16/FP16 as the primary model. FP8 versions can optionally be downloaded for reference/eager inference, but the compile scripts must always point to BF16/FP16.

**Ask the user for exact download URLs.** Do NOT guess HuggingFace URLs — repos have different naming conventions, subdirectories, and organizations. The user knows where the correct BF16/FP16 checkpoint lives. Ask them to provide:
- Diffusion model URL (BF16/FP16)
- Text encoder URL
- CLIP Vision URL (if I2V model)
- VAE URL
- LoRA URLs (if needed)

Example prompt to user: "Please provide the HuggingFace download URLs for each component (model BF16, text encoder, VAE, LoRAs). I'll create the download script."

Create only if models are on HuggingFace. All repos/files as configurable variables:

```bash
#!/bin/bash
set -e
MODEL_REPO="${MODEL_REPO:-org/model-name}"
MODEL_FILE="${MODEL_FILE:-model_bf16.safetensors}"
LORA_REPO="${LORA_REPO:-org/lora-name}"
LORA_FILE="${LORA_FILE:-lora.safetensors}"
COMFYUI_PATH="${COMFYUI_PATH:-./ComfyUI}"

mkdir -p "$COMFYUI_PATH/models/diffusion_models" "$COMFYUI_PATH/models/loras"

download_hf() {
    local repo=$1 file=$2 output_dir=$3
    echo "Downloading $file from $repo..."
    huggingface-cli download "$repo" "$file" --local-dir "$output_dir"
}

download_hf "$MODEL_REPO" "$MODEL_FILE" "$COMFYUI_PATH/models/diffusion_models"
download_hf "$LORA_REPO" "$LORA_FILE" "$COMFYUI_PATH/models/loras"
```

## Bash Compilation Wrapper Pattern

**CRITICAL: The MODEL variable must ALWAYS point to a BF16/FP16 checkpoint, never FP8.** We compile from full-precision and handle quantization ourselves via `--quantize`. Even if the workflow uses an FP8 model, the compile script must reference the BF16/FP16 version.

Compile **all variants** in one script — BF16 first (lossless baseline), then FP8 (quantized by Qlip):

```bash
#!/bin/bash
# Compile all variants: BF16, BF16+LoRA, FP8, FP8+LoRA
# IMPORTANT: MODEL must be BF16/FP16 — we quantize ourselves via --quantize
set -e

COMFYUI_PATH="${COMFYUI_PATH:-./ComfyUI}"
MODEL="model_bf16.safetensors"  # ALWAYS BF16/FP16, never FP8
LORA_PATH="${LORA_PATH:-$COMFYUI_PATH/models/loras/lora.safetensors}"
LORA_RANK="--min-lora-rank 1 --opt-lora-rank 32 --max-lora-rank 128"
SIZES="--sizes 1024x1024 1024x768"
DYNAMIC="--dynamic-size-range 512 768 1024"
COMMON="--comfyui-path $COMFYUI_PATH --model $MODEL $SIZES $DYNAMIC"
FP8="--quantize --static --calibration-iterations 10 --skip-first-blocks 1 --skip-last-blocks 1"

echo "============================================================"
echo "Compiling model variants"
echo "  Model:   $MODEL (BF16 — Qlip handles quantization)"
echo "  ComfyUI: $COMFYUI_PATH"
echo "============================================================"

# --- BF16 (lossless) ---
echo "=== BF16 ==="
python compile_model.py --engines-dir ./engines/bf16 $COMMON

echo "=== BF16 + LoRA ==="
python compile_model.py --engines-dir ./engines/bf16-lora \
  $COMMON --lora "$LORA_PATH" $LORA_RANK

# --- FP8 (quantized by Qlip from BF16 weights) ---
echo "=== FP8 ==="
python compile_model.py --engines-dir ./engines/fp8 $COMMON $FP8

echo "=== FP8 + LoRA ==="
python compile_model.py --engines-dir ./engines/fp8-lora \
  $COMMON $FP8 --lora "$LORA_PATH" $LORA_RANK
```

# Nodes

How to choose an acceleration, the parameters of every node, and how LoRA works with engines.

## Choose an acceleration

The nodes have three roles:

- **Baseline** nodes define your fast model. Leave them in place.
- **Acceleration** nodes are the levers you tune. Each has an `enable` toggle, so one workflow
  serves both the baseline and every variant.
- **Measurement** nodes only report.

Put the acceleration nodes on the `MODEL` line between the loader and the sampler, in this order:
`QlipAutoSparse → QlipCompile → QlipCache → QlipProgressive`.

| Node | Role | What it does | Main knob |
|------|------|--------------|-----------|
| **QlipEnginesLoader** | baseline | Loads a precompiled `.qlip`/`.engine` and swaps in the blocks at runtime | `engines_path` / `hf_repo` |
| **QlipCompile** (+ **QlipQuantConfig**) | baseline | Compiles the model on the fly (no ONNX/engine build), optional FP8/NVFP4 | `quantize` |
| **QlipLoraStack** / **QlipLoraSwitch** | baseline | Chain LoRA files; enable or disable them at runtime without reloading | `lora_path` / `enable` |
| **QlipCache** | acceleration | Skips whole denoising steps (or middle blocks) by predicting slow-moving output | `threshold` (higher = faster) |
| **QlipAutoSparse** | acceleration | Int8 dynamic block-sparse attention on any DiT | `sparsity` (0.5/0.7/0.9) |
| **QlipProgressive** | acceleration | Runs early steps on a downscaled latent | `low_scale` / `switch_mode` |
| **QlipTokenPrune** | acceleration | Drops the least important tokens (video); stacks with sparse attention | `keep_ratio` |
| **QlipDrafter** | acceleration | Self-speculative sparse draft of the step (video) | `split_step` / `sparsity` |
| **QlipTimer Start/Stop/Report**, **QlipCacheReport** | measurement | Times a span of nodes; reports real vs skipped cache steps | — |
| **QlipSpectrumFit** | helper | Fits `speed_A`/`speed_beta` for the `spectral` mode of `QlipProgressive` | `n_samples` |

Which lever helps depends on what dominates the run time:

- **Image:** resolution dominates. `QlipProgressive` gives the most; attention levers are capped.
- **Video:** attention dominates. `QlipAutoSparse`, `QlipTokenPrune` and `QlipDrafter` give the most.
- **Step count:** `QlipCache` scales with steps, not resolution. It is strongest on 20–50-step
  base models and weak on 4–8-step distilled ones.

The best result is usually two orthogonal levers together, for example cache + progressive on
image or sparse + cache on video. To search the combinations automatically, use the
[QLIP Agent](qlip-agent.md).

Parameters of every node are in the [Node reference](#node-reference).

## Node reference

### Qlip Engines Loader

Loads precompiled engines and replaces the transformer blocks at runtime. Engines are cached
across runs: the first load takes a few seconds, later runs are instant.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model from any loader (UNETLoader, CheckpointLoaderSimple, etc.) |
| `engines_path` | STRING | No | `""` | Path to a directory with `.qlip`/`.engine` files |
| `hf_repo` | STRING | No | `""` | Hugging Face repo with engines, e.g. `TheStageAI/Elastic-FLUX-2-Klein:models/H100/klein-9b-fp8_lora` |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack from `Qlip LoRA Stack` node(s) |
| `cuda_graph` | BOOLEAN | No | `False` | Capture a CUDA Graph to reduce kernel launch overhead |
| `shared_memory` | STRING | No | | Group name. Loaders with the same name share one GPU memory pool (size = max, not sum) |

LoRA support turns on automatically when the engines directory has `lora_config.json` or a
`lora_stack` is connected.

If the engines directory contains `qlip_patch.py`, the loader runs it at load time. It can define
`patch_signatures(dm)` and `patch_caller(dm)` hooks for model-specific patches.

**Output:** `MODEL`

### Qlip LoRA Stack

Adds one LoRA file to a chainable list. Chain several nodes through `prev_stack` to stack LoRAs.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `lora_path` | STRING | Yes | `""` | Path to a LoRA `.safetensors` file |
| `strength` | FLOAT | Yes | `1.0` | Strength multiplier (`-10.0` to `10.0`) |
| `prev_stack` | QLIP_LORA_STACK | No | | Previous stack to extend |

**Output:** `QLIP_LORA_STACK`

### Qlip LoRA Switch

Enables or disables LoRA at runtime without reloading engines. Place it after
`Qlip Engines Loader` with LoRA-enabled engines.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model with loaded engines |
| `enable` | BOOLEAN | Yes | `True` | Enable or disable LoRA |
| `lora_stack` | QLIP_LORA_STACK | No | | LoRA stack to load when enabling |

**Output:** `MODEL`

### Qlip Auto Sparse

Block-sparse attention on any DiT, image or video. The node replaces ComfyUI's
`optimized_attention` with a router: long self-attention calls go to qlip's int8 dynamic
block-sparse kernel, and everything else (text/cross attention, masked calls, short sequences)
goes to the original kernel. Blocks are picked every step, with no per-model tuning. It works on
eager models and on engines; turn `enable` off to restore stock attention without a reload.
Needs the TheStage token, like the engines.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Any DiT (LTX, Wan, etc.) |
| `enable` | BOOLEAN | Yes | `True` | Enable sparse attention |
| `sparsity` | FLOAT | Yes | `0.5` | Fraction of attention blocks to drop (0–0.95). 0.5 is safe, 0.7 is the validated fast point with the `diversity` selector, 0.9 is a draft mode |
| `selector` | CHOICE | No | `diversity` | How blocks are picked. `diversity` (recommended) = similarity minus redundancy, best quality at equal speed. `topk` = flat top-k by similarity. `meansim` = cdf mass + self-similarity. `tau` = dynamic budget per query block; ignores `sparsity` and uses `tau` |
| `tau` | FLOAT | No | `1.0` | `selector=tau` only: keep blocks above mean + `tau`·std of each query block's scores. Larger = fewer blocks = faster, lower quality |
| `correction` | CHOICE | No | `none` | `none` = block-sparse only. `sla` = block-sparse kernel + linear-attention compensator. `vspace_block` = block kernel + V-space rank-`d_rank` correction. `vspace_topk` = per-token top-k + V-space: best quality but pure PyTorch, so slow (for quality probes) |
| `d_rank` | INT | No | `24` | `vspace_*` only: rank of the V-space correction subspace |
| `score_rank` | INT | No | `32` | `vspace_topk` only: rank of the SVD key scorer |
| `simthreshd1` | FLOAT | No | `0.1` | `meansim` only: self-similarity threshold; higher = more sparsity |
| `smooth_k` | BOOLEAN | No | `True` | Key smoothing (de-mean); usually improves quality at no cost |

At 70% sparsity the `diversity` selector keeps dense-level quality (checked with DOVER).

**Output:** `MODEL`

### Qlip Cache

Step caching: skips whole denoising steps by predicting the model's output while the trajectory
moves slowly. It wraps the UNet call through ComfyUI's official wrapper hook, so it works over any
engine (FP8 / NVFP4) and over eager models. Each CFG branch has its own cache, so cond and uncond
never share a predictor.

- `step` mode works best on many-step schedules (base models, 20–50 steps).
- `block` mode (DBCache-style) always computes the first and last blocks and skips the middle
  blocks through a cached residual. It also helps few-step distilled models.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Diffusion model |
| `enable` | BOOLEAN | Yes | `True` | Enable caching |
| `threshold` | FLOAT | Yes | `0.15` | `step` mode: accumulated-error budget before a forced recompute. `block` mode: rel-L1 hidden-state difference below which middle blocks are skipped (0.05–0.1 typical). Higher = more skips = faster, riskier |
| `mode` | CHOICE | Yes | `step` | `step` = skip whole denoising steps. `block` = skip the middle transformer blocks per step |
| `method` | CHOICE | No | `easycache` | `easycache` (reuse the last output), `taylor` (TaylorSeer extrapolation), `hermite` (HiCache: damped extrapolation, steadiest) |
| `order` | INT | No | `2` | Extrapolation order for `taylor`/`hermite` (caches `order` history tensors) |
| `warmup_steps` | INT | No | `4` | First steps that are always computed; they set the composition |
| `max_consecutive_skips` | INT | No | `3` | Maximum skips in a row, so the error cannot compound |
| `fn_blocks` | INT | No | `8` | `block` mode: first N blocks that always compute (their output is the skip probe) |
| `bn_blocks` | INT | No | `0` | `block` mode: last N blocks that always compute (refinement tail) |

**Output:** `MODEL`

### Qlip Cache Report

Prints how many steps were computed vs skipped, the compute ratio and the ideal speedup. Connect
`trigger` to a post-sampler output so it runs after generation.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `trigger` | * | No | | Any post-sampler output, to order execution |

**Output:** `STRING` (also printed to the console)

### Qlip Progressive

Progressive resolution: early denoising steps run on a **downscaled latent**, then the trajectory
continues at full resolution. The composition forms early and survives the upscale, so the higher
the target resolution, the more compute it saves. It is a model hook over ComfyUI's official
wrapper, so you keep your own sampler.

The noisy latent is downscaled with nearest sampling (this preserves noise statistics) and clean
conditioning bilinearly. Nested multimodal latents are supported: for MiniMax H3, video is
downscaled and audio passes through.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Any DiT; this is a model hook, not a sampler |
| `enable` | BOOLEAN | Yes | `True` | Enable progressive resolution |
| `low_scale` | FLOAT | Yes | `0.5` | `sigma` mode only: latent side scale for the low-res phase |
| `switch_at` | FLOAT | Yes | `0.5` | `sigma` mode only: fraction of the sigma range spent at low resolution |
| `switch_mode` | CHOICE | No | `auto` | `auto` = adaptive ladder with no knobs (starts at 0.25 and climbs by itself). `sigma` = one fixed switch at `switch_at` with `low_scale` |
| `stab_threshold` | FLOAT | No | `0.08` | Reserved; `auto` mode ignores it |
| `verify_sigma` | FLOAT | No | `0.0` | Run a full-resolution correction step while sigma/first_sigma ≥ this. `0` = off (fastest, clean on most prompts). If a prompt shows a translucent veil along edges, use about `0.9` (costs ~2 extra full-res steps); `1.0` catches only the first low-res step |
| `carry_prev` | BOOLEAN | No | `True` | Use the previous step's full-res prediction as the band-split signal. Turn off if you see a veil or double contour along silhouette edges |
| `up_mode` | CHOICE | No | `bilinear` | How the model output is upscaled back: `bilinear`, `bicubic` (slightly sharper), `nearest-exact` (hardest), `edge` (hard on silhouette edges, bilinear on flat areas; targets the veil at no speed cost) |
| `backbone_mode` | CHOICE | No | `empirical` | `auto` mode only: when the ladder grows the grid. `empirical` = fixed sigma thresholds (0.9→×0.25, 0.75→×0.5). `spectral` = SPEED (arXiv 2605.18736), switches from the model's data power spectrum |
| `speed_delta` | FLOAT | No | `0.01` | `spectral` only: noise tolerance. Smaller = stay low-res longer (faster, more veil risk) |
| `speed_A` | FLOAT | No | `0.0` | `spectral` only: power-spectrum amplitude A for your model. `0` = built-in default |
| `speed_beta` | FLOAT | No | `0.0` | `spectral` only: power-spectrum exponent beta (typically 1.9–2.6). `0` = built-in default |

The built-in `speed_A`/`speed_beta` may not match your model or resolution. Fit them with
[`Qlip Spectrum Fit`](#qlip-spectrum-fit) or [`tools/fit_spectrum.py`](../tools/README.md).

**Output:** `MODEL`

### Qlip Spectrum Fit

Fits the model's data power spectrum `P(ω) = A·|ω|^(-β)` from a few probe generations and outputs
`speed_A` and `speed_beta` for `Qlip Progressive` (`backbone_mode = spectral`). The fit depends on
resolution, so connect the same `EmptyLatentImage` the sampler uses.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Model to fit (passed through) |
| `clip` | CLIP | Yes | | Text encoder for the probe prompts |
| `latent` | LATENT | Yes | | Latent at your generation resolution |
| `n_samples` | INT | No | `3` | Probe generations to pool (1–5) |
| `steps` | INT | No | `8` | Sampling steps per probe |
| `sampler_name` / `scheduler` | CHOICE | No | `euler` / `simple` | Sampler for the probes |
| `seed` | INT | No | `1000` | Probe seed |
| `prompts` | STRING | No | `""` | One probe prompt per line; empty = built-in diverse set |

**Outputs:** `model`, `speed_A` (FLOAT), `speed_beta` (FLOAT), `report` (STRING)

### Qlip Token Prune

Drops the least salient tokens on a window of middle steps; a video lever. It stacks with
`Qlip Auto Sparse`.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Any DiT |
| `enable` | BOOLEAN | Yes | `True` | Enable pruning |
| `keep_ratio` | FLOAT | Yes | `0.75` | Fraction of tokens kept on a pruned step (0.3–1.0). `1.0` = off. Lower = faster, more quality risk |
| `method` | CHOICE | Yes | `l2sq` | Token saliency: `l2sq` (sum of squared features), `l1`, `linf`, `var` |
| `compensation` | CHOICE | Yes | `prev` | How dropped tokens are filled: `prev` = previous step's hidden state (recommended), `zero` = zeros |
| `step_lo` | FLOAT | Yes | `0.2` | Prune only while the step fraction is ≥ this (0 = first step); keeps early structure steps full |
| `step_hi` | FLOAT | Yes | `0.8` | Prune only while the step fraction is ≤ this; keeps late detail steps full |

**Output:** `MODEL`

### Qlip Drafter

Self-speculative draft: early steps run full attention and set the structure, later steps run
sparse attention. A video lever; it is a model hook, so you keep your own sampler.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Any DiT |
| `enable` | BOOLEAN | Yes | `True` | Enable drafting |
| `total_steps` | INT | Yes | `8` | Number of sampler steps you run (needed to place the split) |
| `split_step` | INT | Yes | `4` | Steps before this run full attention; steps from this one on are drafted. Higher = safer, less speedup. 4 of 8 is a good start |
| `sparsity` | FLOAT | No | `0.9` | Draft attention sparsity on the late steps (fraction of blocks dropped) |
| `selector` | CHOICE | No | `topk` | Block selector for the draft kernel: `topk`, `diversity`, `meansim` |

**Output:** `MODEL`

### Qlip Compile

Compiles the diffusion model in the ComfyUI process, with no ONNX export and no engine build. It
finds the transformer block stacks, compiles each block class once through `torch.compile`,
memoizes shape-pure glue (RoPE embedders), and can run every large block Linear through native
FP8/FP4 scaled GEMM.

Compared with whole-model `torch.compile`, the steady-state speed is the same, but the cold start
takes seconds instead of minutes and **resolution changes need no recompile**. Compilation starts
on the first model call, after LoRA patches and device placement are settled. A block that fails
to compile falls back to eager for good, so the run never breaks.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | | Any DiT; composes with the other Qlip nodes |
| `enable` | BOOLEAN | Yes | `True` | Enable compilation |
| `quantize` | CHOICE | Yes | `none` | `fp8` = e4m3 scaled GEMM (H100+). `fp4` = nvfp4, Blackwell `sm_100`+ only (falls back to fp8 elsewhere) |
| `backend` | CHOICE | No | `default` | `torch.compile` mode for the blocks; `max-autotune` compiles much longer for ~1% extra |
| `act_scales` | CHOICE | No | `calibrate-first-run` | fp8 only. The first run records amax, then the scales freeze (fastest). `dynamic` = per-call amax (~6% slower, no warmup) |
| `force_resident` | BOOLEAN | No | `True` | Convert weight-streamed (dynamic-VRAM) models to fully resident before compiling, so compiled steps are not PCIe-bound |
| `attention` | CHOICE | No | `comfy` | Attention kernel inside the compiled blocks. `comfy` = whatever ComfyUI uses (SDPA, or Sage with `--use-sage-attention`). `auto` / `int8_fp8` / `int8_fp16` / `fp4` = qlip's own SageAttention-class kernels, applied to long unmasked self-attention (≥ 2048 tokens) only |
| `weights_policy` | CHOICE | No | `keep` | `release`: free the master weights after fp8/fp4 quantization. Big VRAM saving; irreversible until the checkpoint is reloaded, so apply LoRAs first |
| `engines_dir` | STRING | No | `""` | Directory for ahead-of-time Inductor packages. Empty = pure JIT. Missing or stale = JIT, then the packages are built once at the end of the first compiled run. Ready = blocks load from the packages with no compilation and no calibration run |
| `shapes` | CHOICE | No | `auto` | JIT only. `auto` = compile for the first resolution, recompile once (dynamic) on the first resolution or prompt-length change. `dynamic` = symbolic token dims from the first compile, so resolution changes never recompile. AOT packages are always dynamic in the token dims |
| `aot_autotune` | BOOLEAN | No | `False` | `engines_dir` build only: Inductor max-autotune while building the packages. Paid once at build time; changing it rebuilds |
| `lora_stack` | QLIP_LORA_STACK | No | | Optional LoRA stack, applied according to `lora_mode` |
| `lora_mode` | CHOICE | No | `merge` | `merge` = ComfyUI merges the LoRA into the weights (no per-step cost; a change re-patches weights, ~10 s). `swap` = side path: a change is an in-place swap (~1 s, no recompile, no package rebuild), ~+19% step cost. Do not also load the same LoRA with `LoraLoader` |
| `lora_max_rank` | INT | No | `256` | `swap` only: rank budget per layer; must cover the largest LoRA (sum of ranks when stacking). Changing it reinstalls |
| `prefetch` | BOOLEAN | No | `True` | Read the checkpoint into the page cache in background threads while the prompt is encoded. Matters only on a cold disk |
| `quant_config` | QLIP_QUANT | No | | Optional `Qlip Quant Config` node; overrides `quantize`/`act_scales` |

Packages built into `engines_dir` use the same workspace format as qlip engines, so
`Qlip Engines Loader` can load them too.

**Output:** `MODEL`

### Qlip Quant Config

Full quantization control for `Qlip Compile`. Optional: the `quantize` input covers the common
cases.

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `scheme` | CHOICE | Yes | `fp8_e4m3` | `fp8_e4m3` (H100+) or `nvfp4` (e2m1 + per-16 e4m3 block scales, Blackwell) |
| `weight_granularity` | CHOICE | Yes | `per-tensor` | `per-channel` = rowwise scaled GEMM: more accurate, slightly slower, no calibration |
| `act_scales` | CHOICE | Yes | `calibrate` | `per-tensor` only: `calibrate` (freeze after `calib_runs`) or `dynamic` (per-call amax) |
| `calib_runs` | INT | Yes | `1` | Sampling runs that feed the observer before the scales freeze |
| `observer` | CHOICE | Yes | `max` | `max` = running maximum (safe), `ema` = exponential moving average (ignores rare spikes) |
| `ema_decay` | FLOAT | Yes | `0.8` | EMA observer decay |
| `min_dim` | INT | Yes | `512` | Linears with any side smaller than this stay unquantized |
| `skip_layers` | STRING | Yes | `""` | Comma-separated name substrings kept in high precision, e.g. `qkv` |

**Output:** `QLIP_QUANT`

### Qlip Timer Start, Stop and Report

Time any span of nodes. Put `Qlip Timer Start` before the measured nodes and `Qlip Timer Stop`
with the same `timer_name` after them. Both pass their input through unchanged.

| Input (Start and Stop) | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `passthrough` | * | Yes | | Any data, passed through unchanged |
| `timer_name` | STRING | Yes | `"timer_1"` | Timer name; must match between Start and Stop |
| `cuda_sync` | BOOLEAN | No | `True` | Call `torch.cuda.synchronize()` for accurate GPU timing |
| `measure_gpu` | BOOLEAN | No | `True` | Also report peak GPU memory between Start and Stop (driver view, includes TensorRT engine memory) |

`Qlip Timer Stop` shows the elapsed time in the node UI and prints it to the console.

`Qlip Timer Report` shows a table of all timers. Connect `trigger` to an output that runs after
every Timer Stop.

| Input (Report) | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `trigger` | * | No | | Any output, to order execution |
| `track_cold_start` | BOOLEAN | No | `False` | Compare with the first (cold start) run, as delta % |

Timers reset between workflow runs. Cold-start values persist for comparison.

## LoRA with engines

LoRA weights are **runtime inputs** to the engine, not baked into its weights. The `lora_packed`
tensor, shaped `[num_layers, rank, max_features, 2]`, holds the A and B matrices. The rank
dimension is dynamic, so different LoRA files work without recompiling.

How the loader caches:

- An engine is loaded once per path and reused across runs.
- When the LoRA stack changes, the weights are swapped in place.
- When the LoRA stays the same, the swap is skipped.
- When the LoRA is removed, the packed tensors are zeroed; the engine is not reloaded.

Constraints:

- Engines compiled without `--lora` cannot take a LoRA at runtime.
- Engines compiled with `--lora` work with and without a LoRA (detected through
  `lora_config.json`).
- Stacked LoRAs concatenate their ranks. The total must fit within the `--max-lora-rank` used at
  compilation.
- LyCORIS / LoKR formats are not supported.

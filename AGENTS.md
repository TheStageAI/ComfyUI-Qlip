# AGENTS.md — QLIP acceleration agent

You are an autonomous acceleration agent for diffusion models (QlipEngine). Given a
model and a ComfyUI workflow (plain, or already carrying a compiled qlip engine), you
insert the qlip acceleration nodes, search over their parameter combinations, judge each
candidate against the baseline with **qlip-arena** (LPIPS + preference), build the
speed×quality Pareto frontier, and deliver the best config plus a structured report. You
do the work; the human only provides the model, the workflow, and the machine.

This is agent-native (like NVIDIA Sol-Engine, but the qlip stack): a coding agent
(Claude Code / Codex) reads this file, edits the workflow, and runs the project scripts,
gated by measured quality — no fixed hyperparameter grid, no magic node.

**The qlip runtime is a closed, obfuscated binary — you cannot read its source.** Your
knowledge of the levers comes ONLY from this repo (ComfyUI-Qlip): the nodes' exposed
parameters and tooltips (§2), `tools/*` (fit_spectrum, qlip_report), and this file.
Never assume anything about qlip internals, and never depend on a value baked inside the
runtime (e.g. a built-in spectrum fit) — if a node needs a model-specific quantity,
compute it with the provided tools. Treat the parameter surface + tooltips as the whole
contract.

---

## 1. Environment — verify EVERY prerequisite before starting

These are user-installed and licensed / user-managed — do NOT guess-install them. Run
ALL checks below first; if ANY fails, STOP and report exactly which piece is missing
(name it) — do not proceed with a partial environment, the run will fail mid-way.

Substitute your own paths. `ARENA` is the qlip-arena checkout dir (arena runs from it).

```bash
VENV=/path/to/venv/bin/python        # the venv with torch + qlip + qlip_algorithms
ARENA=/path/to/qlip-arena            # qlip-arena repo (run its CLI from here)

# (a) core runtime — torch + qlip (the licensed engine) + qlip_algorithms (metrics)
$VENV -c "import torch, qlip, qlip_algorithms; print('runtime OK', torch.__version__)"

# (b) qlip-arena importable AND its qlipmetrics judge (needs qlip_algorithms)
( cd "$ARENA" && $VENV -c "import qlip_arena; from qlip_arena.judges.qlipmetrics import QlipMetricsJudge; print('arena + qlipmetrics OK')" )

# (c) TheStage license token configured — licensed nodes (QlipProgressive, QlipCache,
#     QlipAutoSparse, QlipDrafter, QlipTokenPrune) FAIL at inference without it.
test -f ~/.thestage/config.json && echo "thestage token OK" || echo "MISSING thestage token — run: thestage config set --access-token <TOKEN>"
#     (the file is written by `thestage config set --access-token <TOKEN>`; get the
#      token at app.thestage.ai. A missing/invalid token often surfaces later as the
#      misleading error 'Nvidia support is not available' — check this FIRST.)

# (d) matplotlib for the final charts
$VENV -c "import matplotlib; print('plots OK')"

# (e) ComfyUI is up and the qlip nodes are registered
curl -s http://127.0.0.1:8188/object_info | $VENV -c "import sys,json; d=json.load(sys.stdin); q=[k for k in d if k.startswith('Qlip')]; print('ComfyUI up, Qlip nodes:', len(q))"
```

Set the arena store dir: `export QLIP_ARENA_ROOT=/path/to/arena_runs`. Arena CLI is
always `cd "$ARENA" && $VENV -m qlip_arena.cli <cmd>`.

Every check must print its OK line. If (a) fails → qlip / qlip_algorithms not installed
in this venv. (b) → qlip-arena not present or qlip_algorithms missing. (c) → license not
set (licensed nodes will crash). (e) → ComfyUI not running or the node not loaded. STOP
and report the specific failing item; do not work around it.

## 2. Read the live search space (source of truth = the running nodes)

Do NOT trust any hardcoded parameter list (including the summary at the end of this
file — it is a hint, and drifts when nodes/params are added). Read the CURRENT schema
of every qlip node from the running ComfyUI, so newly added nodes and parameters are
picked up automatically:

```bash
curl -s http://127.0.0.1:8188/object_info | $VENV -c '
import sys, json
d = json.load(sys.stdin)
for name in sorted(k for k in d if k.startswith("Qlip")):
    req = d[name].get("input", {}).get("required", {})
    opt = d[name].get("input", {}).get("optional", {})
    print("##", name)
    for grp in (req, opt):
        for pin, spec in grp.items():
            t = spec[0]                                   # type or list of choices
            meta = spec[1] if len(spec) > 1 else {}
            if isinstance(t, list):
                typ, rng = "enum", "choices=" + "|".join(map(str, t))
            else:
                typ = t
                rng = " ".join("%s=%s" % (k, meta[k]) for k in ("default","min","max","step") if isinstance(meta, dict) and k in meta)
            print("   %s: %s  %s" % (pin, typ, rng))
'
```
This enumeration IS your search space — every `pin` is a `--set node_id.pin=value`
target; its range/choices come straight from the node.

**Then READ each param's tooltip and learn how to OPERATE the node — do not just sweep
values blindly.** The tooltip (`meta["tooltip"]`, or `nodes/*.py` INPUT_TYPES, or
`tools/README.md`) is the node's manual; the nodes evolve, so you must derive their
operating rules yourself each run, not from memory. From the tooltips extract, per node:

- **Modality** — does it say image / video / long-sequence / turbo-only? Skip a
  video-only lever (e.g. QlipDrafter, QlipTokenPrune) on an image model. Amdahl guide:
  on IMAGE the resolution lever dominates and attention levers are capped; on VIDEO
  attention/token levers matter more — BUT the resolution lever (QlipProgressive) ALSO
  works on video (it downscales the spatial H×W of the latent while keeping frames T, so
  it cuts tokens across the whole block stack — potentially the strongest video lever,
  and it doesn't touch attention so it survives models whose attention wraps tensors in
  a custom container). So on video, TRY progressive too, not just attention/token.
- **Pre-steps / dependencies** — a param may require an offline step first. Example:
  QlipProgressive `backbone_mode=spectral` needs a power-spectrum fit `(speed_A,
  speed_beta)` for the CURRENT model. **You must compute it yourself** by running
  `tools/fit_spectrum.py` on this model (2–3 samples) and setting speed_A/speed_beta
  from its output — for EVERY model, Krea-2 included. Do NOT leave them 0 relying on a
  built-in default: the qlip runtime is a distributed/obfuscated binary you cannot
  inspect or trust to hold the right fit for the model in front of you, and a wrong
  built-in fit silently degrades quality. Your inputs are the ComfyUI-Qlip nodes,
  `tools/*`, and tooltips — never the qlip internals. If fit_spectrum can't run (e.g.
  the model's loader/CLIP isn't wired for it), fall back to `backbone_mode=empirical`
  and record why. (This is the general rule: whenever a node param depends on a
  model-specific quantity, DERIVE it with the provided tools; never assume a hidden
  default is correct.)
- **Mutual exclusions / mode gating** — some params only apply in a mode (backbone_mode
  only when switch_mode=auto; fn_blocks only when mode=block). Respect these in
  §5's axis coverage — don't sweep a param that its mode disables.
- **Failure modes the tooltip warns about** (e.g. "aggressive = veil risk").

If a node you don't recognize appears, this same reading tells you what it is and how to
use it — treat it first-class. (Fallback when ComfyUI isn't up: parse `INPUT_TYPES` from
`nodes/*.py` and read `tools/README.md`.)

## 2b. Operating manual — what each lever physically does and how to drive it

§2 gives you the parameter surface; this section gives you the physics, so that
every value you try is a hypothesis with an expected speedup and an expected
defect, not a random draw. Read it once per run; cite the relevant rule in each
JOURNAL hypothesis ("H12: cache thr 0.15 → 0.20; expect skip ratio 0.35→0.45,
speedup 1.5→1.7×; risk: texture tail per 2b-B").

### 2b-0. Where the time goes (do this BEFORE choosing a lever)

Wall time of one image ≈ `N_steps × (tokens × cost_per_token_per_block × n_blocks)`
+ VAE + text encoders. Three independent multipliers, three families of levers:

| multiplier | lever family | node | saves |
|---|---|---|---|
| number of steps actually computed | **step cache** | QlipCache (mode=step) | whole model calls |
| tokens per call | **resolution** | QlipProgressive; QlipTokenPrune (video) | tokens ∝ (latent side)² — a ×0.5 rung is 4× fewer tokens |
| cost per token | **attention / kernels** | QlipAutoSparse, QlipDrafter (video); QlipCompile (fp8/fp4, sage-class attention) | attention share only (Amdahl) |
| blocks per call | **block cache** | QlipCache (mode=block) | middle blocks |

Measure, don't guess: (a) `N_steps` from the scheduler; (b) tokens = latent
H×W / patch² (+ text tokens); (c) the attention share ≈ tokens / (tokens + hidden)
— above ~10k tokens (video, 2K images) attention dominates and the sparse levers
pay, below it they are Amdahl-capped at ~1.05–1.1× (measured: pixel-identical
no-ops on a 6k-token image model). Write these three numbers in JOURNAL first;
they predict which family can give 2× and which cannot.

Expected speedups (sampler only; add the fixed VAE/text-encoder overhead from
`--overhead` to get end-to-end):

```
step cache      : 1 / (1 − skipped_steps / N)              (QlipCacheReport prints skipped/N)
progressive     : 1 / (1 − f_low · (1 − s²))               f_low = share of steps at rung s (node log prints the σ ladder)
block cache     : 1 / (1 − skipped_blocks / n_blocks)      (report prints "real / skipped of n")
sparse attention: 1 / (1 − attn_share · sparsity)          (upper bound; real kernels keep some overhead)
combined        : multiply — but each lever raises the next one's error, so stack mild rungs, not aggressive ones
```

### 2b-1. Node insertion order and wiring

On every `MODEL` line, from the loader to the sampler:

```
Loader ─► [QlipEnginesLoader | QlipCompile]  ─► QlipProgressive ─► QlipCache ─► [QlipAutoSparse | QlipTokenPrune | QlipDrafter] ─► sampler
             model definition (baseline)         resolution         steps/blocks     attention / tokens (video)
```

- **Model-definition nodes first** (compile / engines / LoRA): they define the
  baseline and are never toggled per config.
- **Progressive before Cache.** The cache decides skips from the change between
  consecutive model outputs; it must see the outputs the progressive hook
  produces (already upscaled to full res), otherwise its error budget is read on
  the wrong tensor. Progressive → Cache is the validated order.
- **Attention / token levers last**, closest to the sampler; they act inside the
  blocks and are transparent to the two hooks before them.
- `QlipSpectrumFit` (or `tools/fit_spectrum.py`) runs ONCE per model+resolution
  before any spectral progressive config; wire the same `EmptyLatentImage` the
  sampler uses.
- Timers: `QlipTimerStart` on the guider/model just before the sampler,
  `QlipTimerStop` on the sampler output, `QlipCacheReport` after decode — the
  reports are how you read skip ratios and the σ ladder.
- **Several MODEL lines** (dual-model guider, high/low experts): the same chain
  on each line, own node ids (§3). Verify with the node log that the second
  line's hooks actually fire ("… of 96" vs "… of 48" calls): on some samplers
  the framework serves both passes with the first model's wrappers, and a hook
  on the second line is silently inert (measured on ComfyUI's dual-model guider).

### 2b-2. QlipProgressive — resolution lever (image AND video)

**Physics.** Diffusion decides composition at high σ (large structures, low
spatial frequencies) and details at low σ. Early steps therefore do not need the
full grid: run them on a latent of side `low_scale × full` (×0.5 side = 4× fewer
tokens), upsample the prediction, continue at full res. Savings ∝ share of steps
spent low-res. **Defects come from exactly two places:** (1) the low-res steps
also decide the composition — on many-step models a ×0.5 start can pick a
*different* layout (the comparison mode reads "layout moved / scene replaced");
(2) the upsampling of contours: a translucent veil / halo along edges (born on
the ×0.25 rung), resample blur when the rung is not an integer factor (×0.75).

**Parameters and what they do:**
- `low_scale` — the starting rung. 0.25 = maximal token saving, veil risk;
  0.5 = the normal rung; 0.75 = little saving, resample blur (avoid).
- `switch_mode=sigma` + `switch_at` — ONE switch at a fraction of the σ range:
  the simplest, most predictable schedule; speedup ≈ `1/(1 − switch_at·(1 − s²))`.
- `switch_mode=auto` + `backbone_mode` — a ladder ×s → … → full: `empirical` grows
  at fixed σ (0.9 → ×0.25 … 0.75 → ×0.5), `spectral` grows when the data spectrum
  (fitted `speed_A`, `speed_beta`) says fine detail is needed, tuned by
  `speed_delta` (smaller = stay low-res longer). The node log prints the σ at
  which it scaled up — READ IT: if it grows to full above σ ≈ 0.85 the low-res
  phase is empty and the speedup ≈ 1.05× (flat-spectrum models, small β).
- `verify_sigma` — full-res correction steps while σ/σ₀ ≥ value: repairs the
  contour noise/veil born at low res; 0.9 catches the three aggressive structure
  steps, 1.0 only the first; costs ≈ 2 full steps.
- `carry_prev` OFF and `up_mode=edge` — two free veil fixes (band-split signal /
  edge-masked upscale); `bicubic` = slightly sharper, `nearest` = worse.

**Ladder from strong to acceptable** (one config per rung, judge each):
1. *Strong*: `sigma`, `low_scale 0.5`, `switch_at 0.65` → ≈1.7× on 48 steps, ≈2× on
   8-step turbo. Read the comparison mode: if > 25 % of prompts have the scene
   replaced, this rung is a **creative** point, not a faithful one.
2. Bring composition back: `switch_at 0.5 → 0.35`, or `auto` empirical; then
   `verify_sigma 0.9` (composition fixed at step 0, veil repaired) — expect
   the speedup to drop to ≈1.1–1.3× on many-step models, stay ≈1.6–2× on
   turbo samplers where the veil is the only issue.
3. If the veil/halo axis still fires: `carry_prev=false`, `up_mode=edge`
   (free), then `low_scale 0.5` if you were at 0.25.
4. Principled schedule: `auto` + `spectral` with the fitted A/β and a
   `speed_delta` sweep (0.005 / 0.01 / 0.05 / 0.1) — pick the largest delta whose
   ladder still spends ≥ 30 % of the steps low-res (log) and passes the gate.
5. Pair the acceptable rung with a mild cache (2b-3); never pair the strong
   rung with an aggressive cache — the texture tail stacks (measured).

**Acceptable** = passes the two-part gate (§4b) AND ≥ 75 % of prompts aligned.
On few-step (turbo) models progressive is the main lever and reaches 2× within
that; on 40+-step models it reaches 1.5–1.7× only as a creative point — report
it as such, and let the cache carry the faithful frontier.

### 2b-3. QlipCache — step cache (mode=step) and block cache (mode=block)

**Physics.** Consecutive denoising steps produce nearly the same residual on the
plateau of the trajectory; the cache measures the change and, while an error
budget (`threshold`) is not exceeded, reuses the previous residual instead of
calling the model (`easycache`) or extrapolates it (`taylor`, `hermite`). Savings
= skipped calls. **Defects:** skipped steps lose the fine detail those steps
would have added (texture loss, `detail_loss`), skip/compute alternation leaves a
tile grid on flat areas (`grid_db`), extrapolation adds speckle (`texture_gain`)
and, at high order, re-decides the composition; too few warmup steps change the
layout. Few-step distilled models have no plateau — step cache is a no-op there
(use block mode or progressive).

**Parameters:**
- `threshold` — the error budget; the only knob that moves speed. Read
  `QlipCacheReport`: skipped/N gives the speedup directly. 0.05 = nothing
  skipped (no-op ≈1.0×); 0.10 → ~20–25 % skipped; 0.15 → ~35 %; 0.20–0.25 → ~45 %
  but the texture tail fails.
- `method` — `easycache` (reuse; robust; DEFAULT), `hermite` order 1 ≈ easycache,
  `hermite` order ≥ 2 / `taylor` = extrapolation (speckle, composition drift on
  many-step models).
- `warmup_steps` — first steps always computed; they set composition. 4 is the
  floor; raising it costs speed linearly and rarely binds (measured 2/4/8 identical).
- `max_consecutive_skips` — cap on error compounding; 3 is right, 2 costs speed
  without a measurable quality gain.
- `mode=block` — `fn_blocks` first blocks always compute and act as the probe,
  middle blocks are skipped, `bn_blocks` last blocks refine. More `fn_blocks` =
  safer and slower (fn16 ≈ eager, fn8 ≈ 1.27×, fn4 = mesh+haze); it also works on
  few-step models and gives the cleanest tile grid; on multi-model guiders it
  effectively caches the conditional model only.

**Ladder from strong to acceptable:**
1. *Strong*: `easycache`, step, `threshold 0.25` → the maximum skip ratio; expect
   texture p90 to fail the gate.
2. `threshold 0.20 → 0.15 → 0.10` until the gate passes; 0.15 is the usual
   acceptable point (≈1.5× on 48 steps), 0.10 the safe one (≈1.2×).
3. If the tile grid (`grid_db` > 8 dB) or the texture tail is the blocker at the
   speed you want, switch to `mode=block`, `threshold 0.08`, `fn_blocks 12`
   (1.2×, cleanest) and walk `fn_blocks 12 → 8` / `threshold 0.08 → 0.10`.
4. Probe `hermite` order 1 at the same threshold (usually identical) and
   order 2 / `taylor` once for coverage — do not refine them.

### 2b-4. QlipAutoSparse, QlipDrafter, QlipTokenPrune — video / long-sequence levers

**Physics.** Attention cost ∝ tokens²; when tokens ≥ ~10k (video, 2K+ images)
attention is > 50 % of the step and dropping attention blocks (`sparsity`) or
tokens (`keep_ratio`) pays; below that the levers are Amdahl-capped (measured
0.96–0.98× on a 6k-token image model: pixel-identical output, no speedup).
**Defects:** dropped blocks/tokens lose long-range consistency (flicker, drift,
texture pumping on video); the "diversity" selector and the `sla` compensator
exist to keep the distinct blocks.

**Ladders:**
- AutoSparse: `selector=diversity`, `sparsity 0.5 → 0.7 → 0.9` (0.7 is the
  validated fast point); then `correction=sla` at the chosen sparsity (the
  fast+quality path); `selector=tau` with `tau 1.0 → 2.0` for an adaptive budget.
  `smooth_k` stays on.
- Drafter: `split_step` = the step from which late steps are sparse-drafted;
  start at N/2 with `sparsity 0.9`, move the split later (N·0.6, N·0.75) until the
  temporal axes pass; earlier split = faster + drift.
- TokenPrune: `keep_ratio 0.75 → 0.6 → 0.5` inside the `step_lo 0.2 – step_hi 0.8`
  window (early structure and late detail steps stay full); `compensation=prev`.
  If it crashes on a joint text+image token stream, log the signature and exclude.
- Only on video: combine the best sparse/prune point with a mild step cache.

### 2b-5. QlipCompile — the baseline lever

Compilation (per-block torch.compile + fp8/fp4 GEMM + a sage-class attention
kernel) raises the whole baseline: `quantize=fp8` on Hopper+, `fp4` on Blackwell,
`attention=auto` for long sequences, `act_scales=calibrate-first-run` (then
`--warmup 2` in `arena gen` on EVERY config, or the timing is wrong). It is not a
per-config toggle: when present, it defines `<model>-engine` as the baseline
(§3) and every lever above is measured relative to it. If it is absent, note in
REPORT.md that it is available as the offline step that raises everything.

### 2b-6. How to read "acceptable" and how to walk the ladder

Acceptable = the §4b two-part gate (typical prompt at most *slight*, worst 10 %
not *strong*) on `texture, texture_gain, noise, haze, halo, mesh` AND the
comparison mode faithful (≤ 25 % scene replaced) AND adherence not worse. The
procedure per lever is always the same three moves:

1. **Start at the strong rung** of the ladder above (one config): it tells you
   the ceiling of that lever on this model and which axis breaks first.
2. **Walk down one knob at a time** — the knob that owns the failing axis (2b-2/3
   name it: veil → verify/carry/edge; texture tail → threshold; grid → block mode;
   composition → switch_at/warmup) — until the gate passes. That config is the
   lever's acceptable point; the previous rung is its speed ceiling.
3. **Combine** the acceptable points of two levers (resolution × steps), never
   the strong ones; then refine the two corners of the frontier by one
   neighbouring value each (§5).

Log every step with the expected number from the formulas in 2b-0 and the
measured one; a lever whose measured speedup is within 0.05 of 1.0 or whose
images are byte-identical to eager is a no-op — reject it, do not refine it.

## 3. Prepare the workflow (you do this, not the human)

The human hands you a workflow. It may be UI- or API-format, and it may already contain
qlip nodes (fine — see baseline cases). You:

1. **UI → API.** If UI-format, convert to API-format `{node_id:{class_type,inputs}}` (or
   start ComfyUI and capture the API-JSON it POSTs to `/prompt`). Must contain a
   `SaveImage`/`SaveVideo`, a discoverable sampler, and a `CLIPTextEncode`, or arena
   `gen` cannot find the prompt/seed nodes.

2. **Establish the baseline** — the quality reference and speed denominator. TWO cases:
   - **Plain model (no engine):** baseline = the model with all *acceleration* nodes
     disabled → eager. Name it `<model>-eager`.
   - **Already-compiled model (workflow ships `QlipCompile` / `QlipEnginesLoader`):**
     that compiled engine IS the intended baseline — do NOT strip it. The user shipped a
     TRT/fp8/int8 engine on purpose; keep it and measure acceleration nodes *relative to
     the compiled model*. Name it `<model>-engine`. (Compilation is a real speed lever —
     it just isn't a per-config toggle; it defines the baseline the search improves on.)
   - Either way: **baseline = same graph, acceleration hooks off.** Acceleration hooks =
     the per-config toggles (`QlipProgressive`, `QlipCache`, `QlipAutoSparse`,
     `QlipDrafter`). `QlipCompile`/`QlipEnginesLoader`/`QlipLora*`
     are model-definition nodes — left AS GIVEN in both baseline and candidates.

3. **Insert acceleration nodes** into the right points (types/ranges from §2; per-node
   meaning in the repo README):
   - Model-hooks: splice into the `MODEL` line between the model source (loader OR the
     `QlipEnginesLoader`/`QlipLoraSwitch` output) and `KSampler.model`, chained e.g.
     `… → QlipProgressive → QlipCache → KSampler`.
   - Keep an `enable` input on every acceleration node. **Baseline = all acceleration
     `enable=false`** so one workflow serves the reference AND every candidate via `--set`.
   - **Several MODEL lines = several hook targets.** A graph may run more than one
     model per step: a dual-model guider (conditional + unconditional UNet, e.g.
     Ideogram-4's asymmetric CFG), a high/low-noise expert pair (Wan 2.2), a
     refiner. Splice a full hook chain into EVERY line, with its own node ids, and
     treat the split as an axis (§5): the lines contribute differently to the image
     (the unconditional / negative branch only enters through the CFG difference,
     the low-noise expert only touches the last steps), so the best config is often
     ASYMMETRIC — e.g. cache the unconditional UNet at a high threshold while the
     conditional one stays exact, or run only the negative branch at low resolution.

4. Save as `<work>/<model>_qlip_api.json`. Record which node_id is which qlip node —
   those are your `--set node_id.input=value` targets.

5. **Check the baseline for blocked outputs.** Some models return a placeholder
   instead of a picture — a flat grey "Image blocked by safety filter" card, a black
   frame — for prompts their safety filter refuses. The arena detects these
   (`media.is_blocked`, luma std of a thumbnail < 0.035; `store.blocked_keys(run)`)
   and **excludes those prompts from every battle automatically** (a grey card vs a
   grey card is not a comparison). Your job: after the baseline `gen`, print the
   count (`python -c "from qlip_arena import store; print(store.blocked_keys('<model>-eager'))"`)
   and record it in JOURNAL. If more than ~25 % of the suite is blocked, the effective
   suite is too small for `--n 16` — raise `--n` so that ≥ 16 *unblocked* prompts
   remain. Never replace the suite or hand-pick prompts: the exclusion is the
   baseline's refusal, reported as such in the report ("Excluded prompts" line).

## 4. The metric — arena is the judge; LPIPS gates, Elo ranks preference

Everything is measured through **qlip-arena** — never eyeball quality.

- **Two speeds of judging — a deliberate limitation.** In the SEARCH LOOP every
  config is judged by the preference judge only (`arena verdict --light`: PickScore
  win-rate → Δelo, plus the speedup from the run meta). It is cheap in GPU and, more
  importantly, in tokens: one line per config. It is also **measurably blind** to
  veil, halo, mesh and speckle — so the loop's frontier is provisional. At CLOSURE
  the frontier set (plus a margin of runners-up) gets the full degradation judge
  (`arena verdict` / `arena dashboard`: every axis, the two-part gate, the comparison
  mode, the rank with CI), and the final choice is made from that table. Never
  recommend from the light readout.
- Objective: **max speedup subject to passing the arena's degradation gate** (§4b:
  `arena gate` on the defect axes + a dashboard quality rank whose CI overlaps the
  leader's or is within 1.0 of it). `mean_lpips ≤ tol` (default tol=0.35 image) is
  reported for every config and used as the *fidelity* readout, but it is NOT the
  decision: LPIPS cannot tell "the scene was damaged" from "the scene was re-decided
  at the same quality" (progressive/distilled levers do the latter), and the
  preference judges cannot see veil, halo or mesh at all (measured). The
  defect-axis verdict from the dashboard summary decides; LPIPS and Elo explain. Report the whole
  Pareto front (speed × quality rank) too.
- **Suite (the prompts).** Use the arena's built-in FIXED suite — `suite_v1.jsonl` for
  IMAGE (120 prompts across people/scene/composition/dense/text/style/short),
  `suite_video_v1.jsonl` for VIDEO (24). It is the SAME across all models of that
  modality on purpose: a fixed common suite is what makes speedups comparable and
  honest. Do NOT hand-pick per-model prompts (that would let you cherry-pick easy
  cases). The only split is by modality (image vs video). Start with `--n 16`; a fixed
  seed (1000), same seed for the baseline.
- **Three judges, different jobs.** `fingerprint` (+ the dashboard summary over the set) =
  the DECIDER: one number per defect in physical units vs the eager baseline, the
  learned halo/mesh/ringing axes, the comparison mode (aligned / layout moved / scene
  replaced) and a quality ranking with bootstrap CI and `p_best`. `qlipmetrics` =
  reference-fidelity readout: LPIPS + DISTS + SSIM (video: frame- + temporal-LPIPS) of
  each config vs eager; LPIPS is the honest "how far from eager" distance (PSNR is too
  pixel-literal; SSIM forgives blur — measured) but it is a *distance*, not a verdict.
  `ensemble` = human-preference: PickScore + HPSv2 → a Bradley-Terry **Δelo**; run it
  too, because it catches "looks better to a person" — but it prefers veiled/smoothed
  images (measured: PickScore picks the veiled image 63 % of the time), so a positive
  Δelo NEVER overrides a fired defect axis. So: the defect verdict / gate decides IN/OUT and the
  order; LPIPS tells how far from eager and, together with the comparison mode, whether
  the config re-decides the composition; Elo flags user-preferred points among the
  survivors.
- Speedup from `wall` in run meta (arena auto-derives on `export`); `--overhead` gives
  the sampler-only column (VAE+text-encode subtracted; a rough estimate is fine, it does
  not change the reported total speedup). Median over suite, model-load excluded.
- **COMPILED models — you MUST warm up, or every speedup is wrong.** If the model is
  `torch.compile`-d / calibrated (a `QlipCompile` node is enabled, or the checkpoint is
  already compiled — e.g. an `int8-convrot` / TensorRT engine), the first few clips of
  EACH config pay compilation/recompile time (dynamo recompiles on the first real shape,
  and again when a spliced node changes the graph). arena's trimmed-mean can't remove
  that, so a lever looks 2× slower when it isn't. Pass **`--warmup 2`** (or 3) to `arena
  gen` on every config INCLUDING the baseline — it runs that many prompts first and
  discards them, so all measured clips run on the settled graph. Symptom to check for: if
  a config's per-clip walls jump (e.g. 35s,35s,35s → 61s,61s…), warmup wasn't enough —
  raise it. Skip warmup only for pure-eager (uncompiled) baselines where it's wasted time.
- **Remote or local ComfyUI** — arena drives whatever `--base http://HOST:PORT` points
  at. Local (same box): `--base http://127.0.0.1:8188`. Remote server: use its address.
  Same workflow either way; just make `--comfy-output` point at that ComfyUI's real
  output dir. Never claim a speedup that fails the quality gate.

### 4b. Explain WHY a config lost — the degradation report (mandatory for the WHY column)

LPIPS and Elo say *how far* and *whether preferred*; they do not say *what broke*.
The arena's degradation judge does: every config vs the baseline becomes one number
per defect, in physical units (same thresholds for any model/resolution, no
calibration): `blur_px` (gaussian σ, px), `haze` (veil fraction), `noise` (grain
std), `texture` (fine detail lost), `ghost`, `banding`, `wobble` (contour
displacement, px), plus two learned axes `halo` (translucent smear along object
contours — the signature of aggressive progressive schedules) and `ringing`
(over-sharpening overshoot). `color` / `structure` / `drift` are fidelity-only
(reported, never ranked).

Use it in two places:

1. **In the loop — per BATCH, the light readout only.** After a batch of configs has
   generated (§5, 8–12 per round), run ONE command over the whole batch:
   ```
   arena verdict --light --b <baseline> --a <cfg1> <cfg2> … --out-json <work>/verdicts/batchN.json
   ```
   One line per config: `speedup · win-rate · Δelo [CI] · W/T/L · n`. That is all you
   read in the loop — no fingerprint, no inspect, no dashboard, no HTML, no raw logs.
   Frontier law in the loop: RETAIN a config if it improves speed at a non-worse Δelo
   (CI-aware: Δelo intervals that overlap are the same) or improves Δelo at the same
   speed; DISCARD dominated; REJECT no-ops (speedup within 0.05 of 1.0) and crashes.
   Write the JOURNAL "WHY" as `1.49x, elo −41 [−90,+5], 2/13/2` — one line per config.
   Known blind spots of this readout (measured): a veiled, haloed, meshed or speckled
   image can WIN the preference vote. Do not try to compensate in the loop; the
   closure step catches it. Chain hypotheses from the Δelo-vs-speed shape and the
   physics of §2b, not from axes you did not measure.

2. **At closure, for the frontier + a margin.** Take the loop's Pareto front on
   (speedup, Δelo) AND the runners-up that the blind readout might have misplaced:
   every retained config within 0.05× speed or within the Δelo CI of a frontier
   point, up to ~10 configs. Run the FULL judge over that set once:
   ```
   arena verdict --b <baseline> --a <frontier+margin> --out-json <work>/report/verdict_final.json
   ```
   (fingerprint + pickscore auto-judged; prints per config: speedup, **Q** — the
   single quality number, 100 = identical, 50 = every axis at "just noticeable" —,
   **V = speedup·Q/100** — the effective speedup, V < 1 is worse than doing nothing —,
   the two-part gate with the failing axes, faithful / creative / rejected, comparison
   mode, quality rank with CI, LPIPS if qlipmetrics was run). The rows come out in
   **release order**: faithful + gate first, then creative + gate, then rejected,
   inside a tier by V. **Make the final choice from this table**: the first row is the
   recommendation unless its rank CI overlaps a faster faithful row (then the faster
   one, the §4 rule); if the loop's favourite fails the gate, take the next survivor —
   do NOT reopen the search, report it as the cost of the light loop. Q sorts, it never
   replaces the gate. Then, over the same set (ordered mild → aggressive), the
   client-facing page — ONE report, the dashboard (no `arena fpcharts`, no
   `degradation.html`: the dashboard contains the radars, the matrix and the
   ranking, and writes the same `fpcharts_data.json` for `gate` / `qlip_report`):
   ```
   arena dashboard --b <baseline> --a <cfg1> <cfg2> <cfg3> --out <work>/report/dashboard --device cuda --single-file
   ```
   `--single-file` also writes `index_single.html` (~20 MB, images inlined) — the
   file to attach to the ticket; `index.html` needs its `img/` and `maps/` folders.
   Arena v4 (2026-09-09): the gate is `pass` / `fail` / **`insufficient`** — a
   required axis (haze, noise, ghost, texture, texture_gain, halo, mesh) that is
   not measured on enough prompts can never pass; the class can be `unknown`
   (structure model unavailable). Treat INSUFFICIENT / unknown as "not ready",
   never as a pass. Q and V are HEURISTIC summaries; the gate is the decision.
   Every verdict carries `estimator_version`; numbers from different versions
   are never mixed (stale battles are re-judged automatically).
   `arena gate` exits 3 on insufficient data and 2 on a bundle without raw
   observations.
   The dashboard is one interactive page (release table, frontier, defect matrix in
   threshold units, problems in words, PickScore pairs, image explorer with ×4 diff
   and defect heat-maps). **Use it as your eyes before the final choice**: for the
   recommended config and every frontier point open `<work>/report/dashboard/img/`
   and `maps/` — Read the candidate thumbnail, the baseline thumbnail and the
   `<prompt>_<head>.png` maps of the 3 worst prompts (from `data.json →
   per_cfg[cfg].worst_by_axis`), and say in REPORT.md what you SAW (where the halo /
   speckle / smoothing sits, on which objects), not only what the numbers say. A
   number without a looked-at image is not a closure.
   Do NOT put every searched config on the client page — 30+ columns of matrix, radar
   and charts are unreadable and the reader cannot act on them. The all-config
   dashboard you use for the decision goes to `<work>/report/all/` (for the journal and
   for `qlip_report.py --fpcharts <work>/report/all/fpcharts_data.json`, which needs
   every config's summary to classify them); the client sees the frontier page. `qlip_report.py` does the same by default
   (`--report-set frontier`): charts, collage and `workflows/` cover the frontier set,
   the rest is listed by name with the reason (dominated / no-op / gate) and kept in
   `report.json`.
   It auto-judges what is missing (fingerprint + preference) and renders: the quality
   ranking, a summary table, the **defect × config matrix** ("which defect appears where",
   incl. "noticeable on 2/16 prompts" for intermittent ones), a radar (outward = better
   on every petal), per-axis charts, the per-prompt readout, and evidence images — the
   worst pair per config with a ×4 diff, and for the learned axes the model's own defect
   map (bright = where the halo is). **Attach this html next to `frontier.png`**; the
   REPORT.md "rejected hypotheses" must cite its axes.

   Also at closure only, over the frontier set: `arena judge --judge qlipmetrics`
   (LPIPS for the report), `arena judge --judge ensemble` (Δelo with HPSv2 for the
   registry export), `arena adherence` for every creative point, `arena export` for
   every config that goes into `report/`.

No baseline at hand (e.g. checking a single output folder)? `arena qdm --images <dir>`
scores each image alone and prints a one-line defect passport in words.

Two more checks that belong in the closure gate:
- **Statistical tie.** The verdict prints a 95 % CI and `p_best` per config;
  two frontier points whose rank CIs overlap are NOT separated by the data — say so in
  REPORT.md instead of declaring a winner by 0.1 rank.
- **Machine gate.** The two-part `arena gate` from the loop above (median ≤ threshold
  AND p90 ≤ 2×threshold on texture / texture_gain / noise / haze / halo / mesh) returns exit 1 when a config's
  typical prompt is noticeably defective or its bad tail is strong; a config that fails
  the gate cannot be the recommended best, whatever its LPIPS or Elo. For distillation / step-pruning levers also run
  `arena adherence --a <cfg> --b <baseline>` (prompt adherence + seed diversity) and
  reject a config whose adherence `share worse` exceeds 0.2 or whose diversity collapsed.

Reading guide and units: `<qlip-arena>/docs/EVALUATE_YOUR_MODEL.md`, section "Picking
the best accelerated config".

## 5. Search — axes are prescribed, operating points are discovered

This is NOT a fixed grid and NOT a short hand-picked list. Each lever (node) has named
AXES you must cover; the exact VALUES you pick per model from what §2 exposes and from
what the previous results tell you. The completeness guarantee is a **family/axis floor
+ a structured-negative gate**, not a Cartesian product. This is what makes the answer
trustworthy — the user can only confidently pick the workflow if you can show you
covered each axis and explicitly dismissed what you didn't try.

Use **16 prompts per config** (`--n 16`), one fixed seed. Baseline first (§3).

### Axes you must cover per lever (breadth is mandatory; values are yours)

Read each lever's live params from §2. For each lever whose modality applies, you must
probe EVERY axis below with at least 2 distinct values across its range, AND at least one
COMBINATION with another enabled lever, before you may declare that lever explored.
(Amdahl guide only ranks which levers matter MOST — image leans resolution, video leans
attention/token — but QlipProgressive works on BOTH: cover it on video too.)

- **QlipProgressive** (image AND video — spatial downscale of the latent): switch_mode
  (auto AND sigma) · low_scale (aggressive AND
  mild, e.g. 0.25 vs 0.5) · backbone_mode (empirical AND spectral) · verify_sigma (off
  AND ≥1 on-value) · carry_prev (both) · up_mode (≥2 incl. `edge`). Then ≥1 combination
  that pairs a mild rung with a veil fix (e.g. low_scale 0.5 + verify + carry_prev
  false), because single-axis prog configs all pin to the veil (~0.58 LPIPS) and only
  combinations escape it.
- **QlipCache**: threshold (gentle AND aggressive) · method (≥2 of easycache/taylor/
  hermite) · mode (step; block with fn_blocks if the model has ≥2 block regions).
- **QlipAutoSparse** (video): sparsity sweep (≥3 values) OR selector=tau with ≥2 tau ·
  selector (diversity AND ≥1 other incl. tau) · correction (none AND ≥1 vspace).
- **QlipDrafter** (video): split_step (≥2) · sparsity (≥2).
- **QlipTokenPrune** (video): keep_ratio (≥2) · step window (≥1 non-default).
- **Per-line asymmetry** (mandatory whenever the graph has ≥2 MODEL lines, §3): for
  each lever, ≥1 config where only the secondary line (unconditional / negative /
  low-noise expert) is accelerated, and ≥1 where the two lines get different
  strengths (secondary aggressive, primary mild). Compare against the symmetric
  config of the same lever; the asymmetric one usually buys speed at near-zero
  defect cost because the secondary line is weighted by the guidance scale only.
- **Cross-lever combinations** (mandatory, ≥2 total): the best resolution config + cache;
  and (video) best sparse + cache, or sparse + token-prune (orthogonal token vs
  attention cuts).
- **Any node found live in §2 that isn't listed here**: read its tooltip/`nodes/*.py`,
  treat each of its params as an axis, and cover it the same way. The list above is a
  FLOOR, not a ceiling — new nodes are first-class.

Discretisation warning: on an 8-step turbo sampler, two nearby sigma-thresholds can gate
the SAME steps and produce a byte-identical image (seen: verify_sigma 0.85 == 0.90).
When two values give identical output, that axis is saturated there — record it and
move the value further apart, don't count it as two points.

### Practitioner priors — where to START each lever (measured, not guessed)

The axes above are the coverage floor; the ORDER you try values in decides how
fast the frontier fills. These starting points come from measured runs (Krea-2,
MiniMax-H3, Ideogram-4) and beat the node tooltips' old defaults:

- **QlipCache: `method=easycache`, `mode=step` first.** It has been the best
  quality/speed cache on every model so far. Sweep `threshold` 0.10 → 0.15 → 0.20
  (0.05 is usually a no-op ≈ 1.0×; ≥ 0.25 tends to re-decide the scene). `warmup_steps`
  and `max_consecutive_skips` rarely bind — check ONE alternative value, and if the
  images are byte-identical mark the axis saturated. `hermite` order 1 ≈ easycache;
  order ≥ 2 and `taylor` extrapolate and, on many-step models, replace the composition
  and add mesh — probe them once for coverage, do not refine them. `mode=block` is a
  secondary axis (cleaner tile-grid, but see the multi-line caveat: its controller is
  process-global, so with several MODEL lines only one line gets block caching).
- **QlipProgressive: `backbone_mode=spectral` WITH the fitted `speed_A/speed_beta`
  first** (`tools/fit_spectrum.py`, §7) — it is the principled schedule and, when the
  fit is right (R² ≥ 0.95), the best progressive point; `empirical` is the fallback
  when the fit cannot be produced, not the starting point. `low_scale` 0.5 (0.25 is
  the veil/haze rung), `carry_prev` on, `up_mode` bilinear; add `verify_sigma` 0.9
  only when the veil / halo axis fires. **Sweep `speed_delta`** (0.005 / 0.01 / 0.05)
  with spectral: the SPEED rule grows the grid when the data spectrum says detail is
  needed, so on a model with a FLAT spectrum (small β, e.g. Ideogram-4 β ≈ 1.1) the
  default delta 0.01 grows to full res after ~5 % of the steps and the speedup
  collapses to ~1.05× — the node log prints the σ at which it scaled up; if that is
  above 0.85, raise delta or use `switch_mode=sigma` with `switch_at` 0.5–0.65.
  Progressive IS fast on many-step models too (1.5–1.7× measured on Ideogram-4 at
  sigma 0.5–0.65) — but there the composition is decided in the low-res steps, so
  those points land in the **creative** class (scene re-decided on ~30 % of prompts);
  the faithful class keeps only the variant with a full-res verify at step 0
  (~1.1×). Report both classes; which one the client wants is their call.
- **Combination**: best faithful cache + best faithful progressive is the first
  combo to run; stacking an aggressive rung with an aggressive threshold stacks
  defects (texture tail) — measured, do not expect it to add up.
- **Multi-line graphs** (§3/§5): the asymmetric cache (unconditional line at a
  higher threshold) is usually the cheapest extra speed with near-zero defect cost —
  run it right after the symmetric sweep.

### Token economy — how to run this without reading yourself to death

The expensive resource is not the GPU, it is your context. Rules:
- **Batches, not single configs.** Plan 8–12 configs per round from the §2b ladders,
  queue them as ONE detached job, arm ONE wait for `BATCH DONE`. One wake-up per
  batch, never per config.
- **One table per batch.** `arena verdict --light` over the batch is the only thing
  you read. Never open the dashboard HTML in the loop, never `cat` generation logs or ComfyUI logs —
  `tail -3` of the batch log on failure only, `grep -c` for progress.
- **Journal in lines, not prose.** One line per config (name · --set delta · speedup ·
  Δelo [CI] · W/T/L · verdict); a hypothesis is one line with the 2b-0 expectation.
- **No re-derivation.** Read `/object_info` once, write the parameter table to the
  journal, never re-print it. Fit the spectrum once.
- **Heavy artifacts once.** fingerprint, qlipmetrics, ensemble, adherence, inspect,
  dashboard, qlip_report, collage — all at closure, all over the frontier set + margin.
- **Sync less.** JOURNAL/STATUS to the local work dir once per batch, the report at
  the end.

### The loop (one hypothesis per iteration, chained from the last result)

```
1. Observe    read JOURNAL (frontier + rejected signatures) + current best + which axes
              of which levers are still uncovered.
2. Propose    ONE hypothesis = a specific config that either covers an untouched axis or
              is expected to beat the frontier, with a one-line rationale grounded in the
              PREVIOUS results (not blind) AND in the physics of §2b: which multiplier it
              attacks, the expected speedup from the 2b-0 formula, the defect it risks and
              the knob that owns that defect. Chain from the last root-cause (e.g. "prog
              pinned at 0.58 = veil → add verify" → "verify killed speed → pair with
              cache to buy it back").
3. Preflight  validate --set against §2 ranges + mode conditions (backbone_mode only if
              switch_mode=auto; fn_blocks only if mode=block; restart replaces KSampler).
4. Run        arena gen --name <cfg> --set ... --n 16        (queue the whole batch, one
              detached job, ONE wake-up when the batch log says BATCH DONE)
              arena verdict --light --b <baseline> --a <batch…>   # the only judge in the loop
5. Node fails a config crashes (dtype/kernel/license)? Do NOT stop. Retry the SAME lever
              with different params (e.g. quantize=none instead of fp8); if still broken,
              EXCLUDE it and log JOURNAL-rejected with the error signature. One dead node
              never blocks the search.
6. Export     (at CLOSURE only, for the frontier set) arena export --a <cfg> --b <baseline>
              --judges qlipmetrics,ensemble --out results/<cfg>__vs__<baseline>.json --overhead <s>
7. Log --set  append to <work>/configs.jsonl: {"name":"<cfg>","set":"--set ..."}
              (arena does NOT store overrides — this file feeds best_workflow.json and
              the frontier's --set column. REQUIRED per config.)
8. Record     append to JOURNAL (one line per config from the light table), update
              STATUS once per batch. Frontier law in the loop: quality = Δelo (CI-aware);
              RETAIN if speed OR Δelo improved; DISCARD if dominated; REJECT if invalid
              (broken OFF-identity, no-op ~1.0×, crash). The defect gate and the
              faithful / creative split are applied at closure (§4b-2), not here.
9. Loop.      A single failure does NOT end the loop — log the signature, propose a
              MEANINGFULLY DIFFERENT next hypothesis.
```

### Run to frontier SATURATION — there is no config-count target

Do NOT stop at a fixed number of configs. Run in rounds until the frontier stops
improving:

1. **Coverage round.** Run every axis of every applicable lever (the list above),
   baseline first. After each config, regenerate the report (`qlip_report.py`) so the
   current Pareto frontier is always known.
2. **Refine rounds.** Look at the frontier's 2–3 corner points. For each, sweep the ONE
   neighbouring value of its dominant param (best is verify_sigma 0.9 → try 0.8 and
   0.95; best low_scale 0.5 → try 0.4, 0.6; well apart, per the discretisation warning),
   plus any promising combination the last round suggested. Regenerate the report.
3. **Saturation test.** A round is "dry" if it added NO new point to the Pareto frontier
   (every new config was dominated). Keep running refine rounds until **2 consecutive
   dry rounds** — then the frontier is saturated. (This is the loop-until-dry rule; a
   single dry round is not enough — the tail matters.)
4. Only then apply the structured-negative gate below.

Cost is bounded by the model and the frontier shape, not by a budget you set — a simple
image model may saturate in ~15 configs, a rich one in 40+. Fewer than ~12 almost
certainly means axes were skipped, not that the frontier saturated — check the gate.

### Structured-negative gate — you may only STOP when ALL hold

1. **Saturation reached**: 2 consecutive dry refine rounds (no new frontier point).
2. **OFF-identity proven**: baseline (all acceleration off) ≈ eager (mean_lpips ≈ 0).
3. Every applicable lever's axes above covered, OR the lever excluded with a logged
   reason (crash signature / no-op / Amdahl-irrelevant for this modality).
4. The full judge (`arena verdict` / dashboard with every axis, plus qlipmetrics and
   ensemble) on the frontier set + margin, and `arena gate` exit 0 on the recommended
   best. The loop itself used the light readout only — say so in REPORT.md.
5. The frontier names, explicitly: the **best-speed point and its quality cost**
   (which axes fired, from the defect matrix), and the **best-quality point and its
   speed cost**; re-deciding configs (scene replaced) are listed separately as
   "creative" points with their adherence, never mixed into the faithful frontier.
6. For every axis/lever you did NOT push further, a one-line reason WHY it can't beat the
   frontier (saturated / dominated / Amdahl-bounded / crashes). "I ran out of ideas" is
   not a reason; "sparse is Amdahl-capped at 1.09× on this image model, measured" is.
7. Record closure in STATUS.json: `state="search_closed"`, `search_closed_reason="..."`
   naming each dismissed lever and the dry-round count.

## 6. Final report (call the script)

When the search is done, generate the structured deliverable in ONE call:

```bash
$VENV <ComfyUI-Qlip>/tools/qlip_report.py \
    --results <qlip-arena>/results \
    --baseline <model>-eager \
    --tol 0.35 \
    --out <work>/report \
    --model <model> --gpu "<your GPU>" \
    --fpcharts <work>/report/all/fpcharts_data.json \  # REQUIRED: the summary the all-config dashboard writes (§4b-2) — the verdict decides
    --report-set frontier \                      # default: charts / collage / workflows cover the frontier set only
    --workflow <work>/<model>_qlip_api.json \    # API graph → best_workflow.json (queue-ready)
    --ui-workflow <original UI-format workflow.json> \  # UI graph → *_ui.json (canvas-openable)
    --object-info http://127.0.0.1:8188/object_info \   # maps node.pin → the right widget slot
    --configs <work>/configs.jsonl               # the --set map (arena omits overrides)
```

**ALWAYS pass BOTH `--workflow` (API) AND `--ui-workflow` (UI) + `--object-info`.** The API
graph is what you POST to `/prompt` but it does NOT open in the ComfyUI canvas; the UI graph
(the original `{nodes,links}` workflow the human handed you, e.g. `h3.json`) is what a person
drags into the editor to inspect/tweak. The tool bakes the winning `--set` into BOTH and writes
`best_workflow.json` + `best_workflow_ui.json` (and the same pair per frontier point). Without
`--ui-workflow` the user gets only the un-openable API json — always provide it. `--object-info`
lets the tool place each override in the correct `widgets_values` slot by name (fetch it live
from the running ComfyUI); without it, UI overrides that can't be mapped are left untouched
rather than guessed.

> **Note on the UI graph containing your Qlip hooks.** The `--set` targets are your inserted
> acceleration nodes (QlipProgressive/Cache/AutoSparse/…). For the `*_ui.json` to carry those
> overrides, the UI workflow you pass must ALSO contain those Qlip nodes (same node ids as the
> API graph). If your UI→API step (§3) added the hooks only to the API graph, add them to a UI
> copy too (or keep the human's UI graph with the hooks placed) so both formats stay in sync.
It reads the exported arena JSONs and writes:
- `report.json` — every config with (speedup, mean_lpips, Δelo), the Pareto frontier,
  and the single recommended best config for the LPIPS tolerance. **This LPIPS-based
  pick is provisional**: the recommended best in REPORT.md and `best_workflow*.json`
  MUST be the config chosen by the §4b rule (gate exit 0 + quality rank, faithful
  mode). If qlip_report's pick differs, re-run it with `--point <verdict-best>` and
  state in REPORT.md which axis/mode disqualified the LPIPS pick;
- `frontier.png` — speed × quality scatter, each point coloured by the arena judge's
  verdict vs eager (green=win, grey=tie, red=lose) so you see where pixels differ but the
  preference judge still calls it OK;
- `verdicts.png` — per-config win/tie/lose bars from the ensemble judge vs eager;
- `params.png` — which node parameters/combinations the frontier configs use;
- `REPORT.md` — human summary (best config, reproduction `--set`, rejected hypotheses);
- **`best_workflow.json`** (API) + **`best_workflow_ui.json`** (UI/canvas) — the winning
  config baked into the source workflow, in BOTH formats: the API graph queues directly via
  `/prompt`, the UI graph opens in the ComfyUI editor to inspect/tweak;
- **`workflows/<config>.json`** + **`workflows/<config>_ui.json`** — the same API+UI pair for
  EVERY frontier point, so the user can pick any operating point (max-speed, max-quality,
  balanced) and open it in either form. ALWAYS pass `--workflow` AND `--ui-workflow` so both
  are produced.

To hand the user the workflow for a SPECIFIC point on request (any config, not just the
frontier), re-run with `--point <config-name>` → `report/point_<name>.json`. (You can
also just point them at the pre-written `workflows/<name>.json`.)

**Visual collage (REQUIRED — numbers aren't enough, the user must SEE the quality).**
Make a single shareable PNG comparing eager against key Pareto points. Use the bundled
tool (it reads `report.json`, picks eager + best + max-quality + max-speed automatically,
highlights the recommended best with a box + "★ BEST", and labels each column with
speedup / LPIPS / judge-verdict):
```bash
$VENV <ComfyUI-Qlip>/tools/qlip_collage.py \
    --store "$QLIP_ARENA_ROOT" \
    --report <work>/report/report.json \
    --out <work>/report/collage.png \
    --rows 3 --seed 1000
```
`--store` is your `QLIP_ARENA_ROOT` (it has `runs/<config>/images/`). It auto-selects
columns, but you can force them with `--configs "<cfg1>,<cfg2>,..."` (eager is prepended).
This replaces the older `arena compare` HTML path — the PNG is the deliverable.

Hand the user: the recommended `best_workflow.json` (+ `best_workflow_ui.json` to open in the
canvas), the `workflows/` for other points (API + `_ui.json` each), `frontier.png` +
`verdicts.png`, the `collage.png`, and `report/` — and point them to
`tools/READING_THE_AGENT_REPORT.md` (this folder) and the arena's `docs/READING_THE_REPORT.md`
(every metric, in plain words) so they can read it without you.

## 7. Spectrum fit — REQUIRED before using QlipProgressive spectral backbone

If you use `backbone_mode=spectral`, you MUST fit the power spectrum of the CURRENT
model yourself and pass `speed_A`/`speed_beta` — for every model, no exceptions. Do not
rely on any built-in default inside the qlip runtime (it is obfuscated and may not match
this model; leaving speed_A/speed_beta at 0 is a silent quality bug).

```bash
$VENV <ComfyUI-Qlip>/tools/fit_spectrum.py --model <name> --unet <ckpt> \
    --clip <te> --vae <vae> ...       # see tools/README.md for the exact flags
# → prints speed_A, speed_beta, R². Set those on the QlipProgressive node.
```
Read `tools/README.md` for the invocation. If the fit can't be produced (loader/CLIP
not available to the script), do NOT fake it — use `backbone_mode=empirical` and record
that spectral was unavailable.

## Journal / status formats

`JOURNAL.md` (append-only): a `## Frontier` table (id | --set | speedup | mean_lpips |
status), a `## Rejected` table (hypothesis | signature), and `## Iterations` — per
config a **hypothesis** entry (Hypothesis / Config `--set` / Expect) then a **result**
entry (speedup, mean_lpips, authenticity, WHY, RETAIN|discard|reject).

`STATUS.json`: `{model, iter, state, objective, baseline, frontier:[{id,config,speedup,
mean_lpips,status}], best_config:{...,reproduce}, rejected:[{hypothesis,signature}]}`.

## Node roles (HINT ONLY — the live `/object_info` from §2 is authoritative)

This list orients you; it does NOT bound the search. Always enumerate params live (§2),
so params/nodes added after this file was written are still searched.

Acceleration hooks (the per-config toggles you search over):
- **QlipProgressive** — image resolution lever (downscaled early steps, grows grid).
- **QlipCache** — temporal reuse across steps/blocks.
- **QlipAutoSparse** — video attention sparsity.
- **QlipDrafter** — video self-speculative sparse-draft.

Model-definition nodes (NOT searched; define the baseline, left as given):
- **QlipCompile** / **QlipQuantConfig** — compile the model to a TRT/fp8/int8 engine.
  This is the biggest single speed lever, but it's a build step, not a per-config knob:
  when present it defines the `<model>-engine` baseline the search improves upon.
- **QlipEnginesLoader** — load a prebuilt engine. **QlipLoraStack/Switch** — LoRA.

Amdahl guide: image (attn small) → progressive/restart + cache; video (attn large) →
sparse/drafter + cache. Always try the best resolution config + cache together. If the
model is NOT pre-compiled, note that `QlipCompile` is available as an offline lever the
user can apply to raise the whole baseline (out of the per-config search, but worth
recommending in the report when eager is the baseline).

## Done criteria

A run is complete only when the §5 structured-negative gate is satisfied — every
applicable lever's axes covered or explicitly dismissed with a reason — AND you have a
non-empty Pareto frontier, a recommended best-config with speedup > 1.0 at mean_lpips
under the tolerance, best_workflow.json, and populated JOURNAL / STATUS / REPORT. A thin
frontier from few configs is NOT done: it means axes were skipped.

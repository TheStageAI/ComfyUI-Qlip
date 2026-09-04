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
  only when switch_mode=auto; fn_blocks only when mode=block; a SAMPLER-emitting node
  like QlipRestartSampler needs a SamplerCustom branch, not KSampler). Respect these in
  §5's axis coverage — don't sweep a param that its mode disables.
- **Failure modes the tooltip warns about** (e.g. "aggressive = veil risk").

If a node you don't recognize appears, this same reading tells you what it is and how to
use it — treat it first-class. (Fallback when ComfyUI isn't up: parse `INPUT_TYPES` from
`nodes/*.py` and read `tools/README.md`.)

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
     `QlipDrafter`, `QlipRestartSampler`). `QlipCompile`/`QlipEnginesLoader`/`QlipLora*`
     are model-definition nodes — left AS GIVEN in both baseline and candidates.

3. **Insert acceleration nodes** into the right points (types/ranges from §2; per-node
   meaning in the repo README):
   - Model-hooks: splice into the `MODEL` line between the model source (loader OR the
     `QlipEnginesLoader`/`QlipLoraSwitch` output) and `KSampler.model`, chained e.g.
     `… → QlipProgressive → QlipCache → KSampler`.
   - `QlipRestartSampler`: emits a `SAMPLER`, wire into `SamplerCustom` (mutually
     exclusive with the progressive hook in one run).
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

## 4. The metric — arena is the judge; LPIPS gates, Elo ranks preference

Everything is measured through **qlip-arena** — never eyeball quality.

- Objective: **max speedup subject to passing the arena's degradation gate** (§4b:
  `arena gate` on the fpcharts axes + a fpcharts quality rank whose CI overlaps the
  leader's or is within 1.0 of it). `mean_lpips ≤ tol` (default tol=0.35 image) is
  reported for every config and used as the *fidelity* readout, but it is NOT the
  decision: LPIPS cannot tell "the scene was damaged" from "the scene was re-decided
  at the same quality" (progressive/distilled levers do the latter), and the
  preference judges cannot see veil, halo or mesh at all (measured). The
  defect-axis verdict from `fpcharts` decides; LPIPS and Elo explain. Report the whole
  Pareto front (speed × fpcharts quality rank) too.
- **Suite (the prompts).** Use the arena's built-in FIXED suite — `suite_v1.jsonl` for
  IMAGE (120 prompts across people/scene/composition/dense/text/style/short),
  `suite_video_v1.jsonl` for VIDEO (24). It is the SAME across all models of that
  modality on purpose: a fixed common suite is what makes speedups comparable and
  honest. Do NOT hand-pick per-model prompts (that would let you cherry-pick easy
  cases). The only split is by modality (image vs video). Start with `--n 16`; a fixed
  seed (1000), same seed for the baseline.
- **Three judges, different jobs.** `fingerprint` (+ `fpcharts` over the set) =
  the DECIDER: one number per defect in physical units vs the eager baseline, the
  learned halo/mesh/ringing axes, the comparison mode (aligned / layout moved / scene
  replaced) and a quality ranking with bootstrap CI and `p_best`. `qlipmetrics` =
  reference-fidelity readout: LPIPS + DISTS + SSIM (video: frame- + temporal-LPIPS) of
  each config vs eager; LPIPS is the honest "how far from eager" distance (PSNR is too
  pixel-literal; SSIM forgives blur — measured) but it is a *distance*, not a verdict.
  `ensemble` = human-preference: PickScore + HPSv2 → a Bradley-Terry **Δelo**; run it
  too, because it catches "looks better to a person" — but it prefers veiled/smoothed
  images (measured: PickScore picks the veiled image 63 % of the time), so a positive
  Δelo NEVER overrides a fired defect axis. So: fpcharts/gate decides IN/OUT and the
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

1. **Per config, in the loop** (cheap, CPU ≈ 0.5 s/pair; the learned axes use the GPU
   if present): after step 4 also run
   ```
   arena judge --a <cfg> --b <baseline> --judge fingerprint
   arena inspect --a <cfg> --b <baseline>          # one plain-words line per prompt
   arena fpcharts --b <baseline> --a <all retained cfgs> <cfg> --out <work>/report/degradation.html
   # the defect gate = two checks (typical prompt + bad tail), both must pass:
   arena gate --data <work>/report/fpcharts_data.json --config <cfg> \
              --max texture --max haze --max halo --max mesh --stat median      # typical prompt ≤ threshold ("slight" at most)
   arena gate --data <work>/report/fpcharts_data.json --config <cfg> \
              --max texture=0.30 --max haze=0.20 --max halo=0.10 --max mesh=0.20 --stat p90   # worst 10 % ≤ 2×threshold (no "strong")
   ```
   (2×threshold is the arena's "strong" band; the limits above are 2× the
   thresholds in `fpcharts_data.json` → `summary.thresholds`. `tools/qlip_report.py
   --fpcharts` applies exactly this two-part gate itself.)
   `fpcharts` is regenerated over the whole retained set every iteration (it only
   judges what is missing, so it is cheap) — its verdict block is the current state
   of the search: the quality ranking with CI/p_best, the defect × config matrix and
   the **comparison mode** per config. Read three things from it for the new config:
   - **gate**: exit 0/1 from `arena gate` (p90 tails). Exit 1 → the config is
     REJECTED for the recommendation whatever its LPIPS/Elo; it may stay on the
     frontier only as a labelled "fails gate: <axis>" point.
   - **rank**: its position in the fpcharts quality ranking; a rank CI overlapping
     the leader's is a tie, not a loss.
   - **mode**: `aligned` / `layout moved` / `scene replaced` counts. A config that
     replaces the scene on > 25 % of prompts (or moves the layout on > 50 %) is a
     **re-deciding** config: it is judged on structure + the learned axes only, its
     LPIPS is meaningless as quality, and it can only be recommended as a separate
     "creative / non-faithful" operating point — never as the default best. Run
     `arena adherence` on it (share worse ≤ 0.2 required).
   Write the JOURNAL "WHY" from the axis that fired — `texture +0.23 (strong on
   4/16), haze p90 +0.18, mode: 9/16 scene replaced` beats "looks worse" or "lpips
   0.41". A config whose LPIPS is fine but whose `halo`/`texture`/`mesh` axis crosses
   the threshold is a config the user WILL see — log it and reject it. Chain
   hypotheses from the axis, not from LPIPS: veil → `verify_sigma`/milder
   `low_scale`; texture loss → earlier switch / `carry_prev`; halo → the schedule is
   too aggressive at the switch, not a cache problem; scene replaced → the low-res
   rung decides the composition, so start higher (`low_scale`) or verify earlier.
2. **At closure, for the frontier** — one command over the frontier configs, ordered
   mild → aggressive, gives the client-facing page:
   ```
   arena fpcharts --b <baseline> --a <cfg1> <cfg2> <cfg3> --out <work>/report/degradation.html
   ```
   It auto-judges what is missing (fingerprint + preference) and renders: the quality
   ranking, a summary table, the **defect × config matrix** ("which defect appears where",
   incl. "noticeable on 2/16 prompts" for intermittent ones), a radar (outward = better
   on every petal), per-axis charts, the per-prompt readout, and evidence images — the
   worst pair per config with a ×4 diff, and for the learned axes the model's own defect
   map (bright = where the halo is). **Attach this html next to `frontier.png`**; the
   REPORT.md "rejected hypotheses" must cite its axes.

No baseline at hand (e.g. checking a single output folder)? `arena qdm --images <dir>`
scores each image alone and prints a one-line defect passport in words.

Two more checks that belong in the closure gate:
- **Statistical tie.** The fpcharts verdict prints a 95 % CI and `p_best` per config;
  two frontier points whose rank CIs overlap are NOT separated by the data — say so in
  REPORT.md instead of declaring a winner by 0.1 rank.
- **Machine gate.** The two-part `arena gate` from the loop above (median ≤ threshold
  AND p90 ≤ 2×threshold on texture / haze / halo / mesh) returns exit 1 when a config's
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
- **QlipRestartSampler** (image): switch_step (≥2) · low_scale (≥2) · up_mode (≥2).
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

### The loop (one hypothesis per iteration, chained from the last result)

```
1. Observe    read JOURNAL (frontier + rejected signatures) + current best + which axes
              of which levers are still uncovered.
2. Propose    ONE hypothesis = a specific config that either covers an untouched axis or
              is expected to beat the frontier, with a one-line rationale grounded in the
              PREVIOUS results (not blind). Chain from the last root-cause (e.g. "prog
              pinned at 0.58 = veil → add verify" → "verify killed speed → pair with
              cache to buy it back").
3. Preflight  validate --set against §2 ranges + mode conditions (backbone_mode only if
              switch_mode=auto; fn_blocks only if mode=block; restart replaces KSampler).
4. Run        arena gen --name <cfg> --set ... --n 16
              arena judge --a <cfg> --b <baseline> --judge qlipmetrics
              arena judge --a <cfg> --b <baseline> --judge ensemble    # BOTH — Elo matters
5. Node fails a config crashes (dtype/kernel/license)? Do NOT stop. Retry the SAME lever
              with different params (e.g. quantize=none instead of fp8); if still broken,
              EXCLUDE it and log JOURNAL-rejected with the error signature. One dead node
              never blocks the search.
6. Export     arena export --a <cfg> --b <baseline> --judges qlipmetrics,ensemble
              --out results/<cfg>__vs__<baseline>.json --overhead <s>
7. Log --set  append to <work>/configs.jsonl: {"name":"<cfg>","set":"--set ..."}
              (arena does NOT store overrides — this file feeds best_workflow.json and
              the frontier's --set column. REQUIRED per config.)
8. Record     append to JOURNAL, update STATUS. Frontier law (quality = the fpcharts
              quality rank, NOT LPIPS): RETAIN if quality OR speed improved (it's on
              the frontier); DISCARD only if neither; REJECT if invalid (broken
              OFF-identity, no-op ~1.0× claiming a gain, crash, `arena gate` exit 1,
              or a re-deciding config outside the separate "creative" list).
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
4. All three judges (fingerprint via fpcharts, qlipmetrics, ensemble) on every
   retained config, and `arena gate` exit 0 on the recommended best.
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
    --fpcharts <work>/report/fpcharts_data.json \  # REQUIRED: the arena verdict decides (gate, mode, quality rank)
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
  MUST be the config chosen by the §4b rule (gate exit 0 + fpcharts rank, faithful
  mode). If qlip_report's pick differs, re-run it with `--point <fpcharts-best>` and
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
`verdicts.png`, the `collage.png`, and `report/`.

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
- **QlipRestartSampler** — image, a `SAMPLER` (grid-grow restart).

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

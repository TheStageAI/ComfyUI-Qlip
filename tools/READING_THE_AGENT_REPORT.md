# Reading the agent's `report/`

The QLIP agent's deliverable is a folder produced by `tools/qlip_report.py`
and `tools/qlip_collage.py` on top of qlip-arena's judges. The arena's own
pages and every metric in them are decoded in the arena repo:
`docs/READING_THE_REPORT.md` (cheat sheet) and `docs/METRICS.md` (formulas).
This note covers only the files the agent adds on top.

## `REPORT.md`

- **Header line** — the decision rule and the ranking weights the run used.
- **Recommended config** — name, speedup, arena quality rank with its 95 %
  interval, LPIPS, Δelo; "Why this one" (the fastest config statistically
  tied with the best-quality one, or *COMPROMISE* when none is); the axes it
  is best / worst on; its comparison mode ("17/17 aligned" = keeps the
  composition on every prompt); "Gate: passed". If an LPIPS-only rule would
  have picked another config, it says which and what disqualified it. Then
  the exact `--set` to reproduce.
- **Frontier table** — the faithful Pareto front, slowest to fastest:
  speedup, rank [interval] and p_best, "best on / worst on" (on how many axes
  the config is the best / worst of the set), mode, LPIPS, Δelo, settings.
  ⭐ = the recommendation. Rows further down are faster and worse; the arena's
  defect matrix says what got worse.
- **Re-deciding configs (creative points)** — configs that change the
  composition on too many prompts: a different render, not a degraded one.
  Offer them only when the composition need not be preserved.
- **Rejected by the defect gate** — the speeds are real, the quality is not
  acceptable; the failing axis and value are named.
- **"N other configs …"** — dominated / no-op configs by name; data in
  `report.json`.

## Charts

- **`frontier.png`** — x = speedup, y = arena quality rank (lower = better);
  blue line = the faithful Pareto front; ★ = recommended; × = failed the
  gate; hollow circle = creative; grey bar = the rank interval; dot colour =
  the preference judge (grey tie, green win, red lose).
- **`verdicts.png`** — per config: how often the preference judge preferred
  it / tied / preferred the baseline.
- **`params.png`** — which node settings the frontier configs use.
- **`collage.png`** — eager | recommended | quality leader | max-speed
  point, same prompts and seeds, three rows; each column header repeats
  speedup, LPIPS and the judge's verdict. Blocked (safety-filter) prompts are
  skipped.

## Workflows and data

- `best_workflow.json` (API, queue it) and `best_workflow_ui.json` (open in
  the ComfyUI canvas) — the recommended config baked into the source graph.
- `workflows/<config>.json` + `_ui.json` — the same pair for every frontier
  and creative point, so any operating point can be picked.
- `report.json` — every config with all fields, the frontier, the creative
  and rejected lists, the decision rule; `degradation.html` +
  `fpcharts_data.json` — the arena page over the frontier set;
  `all/fpcharts_data.json` — over every config (what the decision used).
- `JOURNAL.md`, `STATUS.json`, `configs.jsonl` — the search history: every
  hypothesis, its expected and measured numbers, why it was retained or
  rejected, and the `--set` of every config.

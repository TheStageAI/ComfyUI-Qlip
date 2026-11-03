# QLIP Agent: automatic tuning

Let a coding agent find the fastest node configuration that keeps your model's quality.

ComfyUI-Qlip has many levers, and finding the combination that makes *your* model fast without
visibly hurting quality is a long sweep. The QLIP Agent runs that sweep for you.

Point a coding agent (Claude Code, Codex, Cursor) at **[`AGENTS.md`](../AGENTS.md)**. It:

- reads your ComfyUI nodes and their parameters live, so it picks up new nodes automatically;
- combines the levers and keeps refining until the speed×quality frontier stops improving;
- judges every candidate with qlip-arena on a fixed prompt suite: LPIPS vs your eager baseline
  and a human-preference Elo;
- returns `best_workflow.json`, a workflow for every point on the frontier, a speed×quality chart
  colored by the judge's verdict (win/tie/lose vs eager) and a side-by-side collage;
- names the defect in each rejected config in physical units (blur in px, veil fraction, grain,
  texture loss, ghosting, contour wobble, contour halo, edge ringing), on how many prompts it
  appears, with the image and defect map.

It works with a local or remote ComfyUI and uses only the exposed node parameters.

Example goal for your agent:

> Read `custom_nodes/ComfyUI-Qlip/AGENTS.md` and follow it to find the best QLIP
> acceleration config for my model. My ComfyUI is at `http://127.0.0.1:8188`, the
> workflow is `my_workflow.json`, and I want max speedup while keeping quality within
> LPIPS 0.3 of the original. Give me the recommended workflow, the full Pareto frontier
> with the arena win/tie/lose verdicts, and a visual collage.

The agent returns a `report/` folder; [`tools/READING_THE_AGENT_REPORT.md`](../tools/READING_THE_AGENT_REPORT.md)
explains each file. The scripts it drives are in [`tools/`](../tools/).

### Compare configs without the agent

If you already generated a baseline and some candidate configs on the same prompts and seeds
(with `arena gen` or any other way; the arena only reads output folders):

```bash
cd /path/to/qlip-arena
python -m qlip_arena.cli collect  --name base --src out_base/        # once per run
python -m qlip_arena.cli collect  --name fast --src out_fast/
python -m qlip_arena.cli fpcharts --b base --a fast fast-2 --out report.html
```

`report.html` is self-contained: the quality ranking, a defect × config matrix in words
(slight / noticeable / strong), a radar (outward = better), per-axis charts, one line per prompt
and the evidence images. With only a folder of outputs and no baseline,
`python -m qlip_arena.cli qdm --images out_fast/` prints a defect passport per image. See
qlip-arena's `docs/EVALUATE_YOUR_MODEL.md`, section "Picking the best accelerated config".

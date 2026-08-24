"""QLIP acceleration search — final report generator.

The QLIP agent (see AGENTS.md) runs a bounded search: it generates each candidate
config through ComfyUI, judges it against the eager baseline with qlip-arena
(qlipmetrics judge → LPIPS; ensemble judge → preference Elo), and exports one
`results/<cfg>__vs__<baseline>.json` per config (arena `export`).

This script consumes that `results/` folder and emits, in one call, the structured
answer the user wants:

  1. report.json  — machine-readable: every config with (speedup, mean_lpips, elo),
                    the Pareto frontier, and the single recommended best config for a
                    given quality tolerance.
  2. frontier.png — speed × quality scatter, Pareto front highlighted, best starred.
  3. params.png   — which node parameters / combinations the frontier configs use
                    (so the user sees WHICH knobs matter, not just the winner).
  4. REPORT.md    — human summary (bottleneck, best config, reproduction, rejected).

Pareto = a config is dominated if another is both faster AND lower LPIPS. The
recommended config = fastest config with mean_lpips <= --tol.

Usage (agent calls this at the end of the search):
    python tools/qlip_report.py \
        --results ~/krea2_engine/qlip-arena/results \
        --baseline krea2-eager \
        --tol 0.35 \
        --out ~/krea2_engine/report \
        --gpu H100 --model krea2

Reads only the exported JSONs — never regenerates or re-judges. Matplotlib is
optional: if unavailable, JSON + REPORT.md are still written (PNGs skipped).
"""
import argparse
import glob
import json
import os


def _as_str(s):
    """Coerce a config/--set value to a string (some runners log it as a list)."""
    if isinstance(s, (list, tuple)):
        return " ".join(str(x) for x in s)
    return s if isinstance(s, str) else ""


def load_config_map(configs_path):
    """Read a configs.jsonl written by the search runner: one {name, set} per
    line, where `set` is the arena --set override string for that config. arena
    itself does NOT store overrides in its export, so the runner records them
    here and we join by name → this is how best_workflow.json knows what to apply."""
    m = {}
    if not configs_path or not os.path.exists(configs_path):
        return m
    for line in open(configs_path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("name"):
                s = d.get("set", "") or d.get("config", "")
                # normalise to a string — some runners log `set` as a list of tokens
                if isinstance(s, (list, tuple)):
                    s = " ".join(str(x) for x in s)
                m[d["name"]] = s if isinstance(s, str) else ""
        except Exception:
            continue
    return m


def load_entries(results_dir, baseline, config_map=None):
    """Read arena export JSONs → list of flat config records."""
    config_map = config_map or {}
    entries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        # arena export records carry name/opponent/speedup/judges{...}
        if baseline and d.get("opponent") not in (baseline, None):
            continue
        judges = d.get("judges", {}) or {}
        qm = judges.get("qlipmetrics", {}) or {}
        ens = judges.get("ensemble", {}) or {}
        speedup = d.get("speedup")
        if speedup is None:
            continue
        lpips = qm.get("mean_lpips")
        if lpips is None:
            lpips = qm.get("median_lpips")
        name = d.get("name", os.path.basename(path))
        entries.append({
            "name": name,
            "method": d.get("method", ""),
            "speedup": float(speedup),
            "sampler_speedup": d.get("sampler_speedup"),
            "mean_lpips": float(lpips) if lpips is not None else None,
            "elo_delta": ens.get("elo_delta"),
            # arena preference verdict vs the eager baseline (ensemble judge):
            # how the judge scored the accelerated output against eager, per prompt.
            "wins": ens.get("wins"),
            "losses": ens.get("losses"),
            "ties": ens.get("ties"),
            "winrate_a": ens.get("winrate_a"),
            "config": _as_str(config_map.get(name)
                              or d.get("config") or d.get("overrides") or ""),
            "wall_s": d.get("wall_s"),
            "source": os.path.basename(path),
        })
    return entries


def _verdict(e):
    """Judge's overall preference vs eager: 'win' / 'tie' / 'lose' from winrate
    (ties=0.5). None if the ensemble judge wasn't run for this config."""
    w = e.get("winrate_a")
    if w is None:
        return None
    if w > 0.55:
        return "win"
    if w < 0.45:
        return "lose"
    return "tie"


def pareto_front(entries):
    """Non-dominated by (speedup ↑, mean_lpips ↓). Configs without LPIPS are
    treated as worst quality (kept only if uniquely fastest)."""
    scored = [e for e in entries if e["mean_lpips"] is not None]
    front = []
    for e in scored:
        dominated = any(
            o is not e
            and o["speedup"] >= e["speedup"]
            and o["mean_lpips"] <= e["mean_lpips"]
            and (o["speedup"] > e["speedup"] or o["mean_lpips"] < e["mean_lpips"])
            for o in scored)
        if not dominated:
            front.append(e)
    front.sort(key=lambda e: e["speedup"])
    return front


def recommend(front, tol):
    """Fastest config on the frontier within the quality tolerance."""
    ok = [e for e in front if e["mean_lpips"] is not None and e["mean_lpips"] <= tol]
    if not ok:
        return None
    return max(ok, key=lambda e: e["speedup"])


def parse_overrides(cfg_str):
    """Turn an arena --set string into {node.input: value} for the params chart.
    Accepts a string ('--set a=1 --set b=2') or a list of such tokens/pairs."""
    out = {}
    if not cfg_str:
        return out
    if isinstance(cfg_str, (list, tuple)):
        cfg_str = " ".join(str(x) for x in cfg_str)
    if not isinstance(cfg_str, str):
        return out
    toks = cfg_str.replace("--set", " ").split()
    for t in toks:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def write_json(entries, front, best, tol, out_dir, meta):
    front_names = {e["name"] for e in front}
    report = {
        "model": meta.get("model"),
        "gpu": meta.get("gpu"),
        "baseline": meta.get("baseline"),
        "quality_tolerance_lpips": tol,
        "n_configs": len(entries),
        "recommended": best,
        "frontier": front,
        "all_configs": sorted(entries, key=lambda e: -e["speedup"]),
        "dominated": [e["name"] for e in entries if e["name"] not in front_names
                      and e["mean_lpips"] is not None],
    }
    path = os.path.join(out_dir, "report.json")
    json.dump(report, open(path, "w"), indent=2)
    return path, report


def plot_frontier(entries, front, best, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    scored = [e for e in entries if e["mean_lpips"] is not None]
    if not scored:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6))
    # Colour each point by the ARENA JUDGE's preference verdict vs eager (ensemble):
    # green = judge prefers the accelerated output, grey = tie (indistinguishable to
    # the judge even though pixels differ), red = judge prefers eager. This shows the
    # LPIPS distance AND whether a human-preference judge still calls it OK.
    vcol = {"win": "#188038", "tie": "#9aa0a6", "lose": "#d93025", None: "#c9c9c9"}
    for v in ("win", "tie", "lose", None):
        pts = [e for e in scored if _verdict(e) == v]
        if not pts:
            continue
        lbl = {"win": "judge: win vs eager", "tie": "judge: tie",
               "lose": "judge: lose", None: "no judge"}[v]
        ax.scatter([e["speedup"] for e in pts], [e["mean_lpips"] for e in pts],
                   c=vcol[v], s=54, label=lbl, zorder=2, edgecolor="white", lw=0.5)
    fx = [e["speedup"] for e in front]
    fy = [e["mean_lpips"] for e in front]
    ax.plot(fx, fy, "-", color="#1a73e8", lw=2, label="Pareto frontier", zorder=3)
    if best:
        ax.scatter([best["speedup"]], [best["mean_lpips"]], marker="*",
                   s=440, color="#f9ab00", edgecolor="#3c4043", lw=1.2,
                   label="recommended", zorder=4)
    for e in front:
        ax.annotate(e["name"], (e["speedup"], e["mean_lpips"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("speedup ×  (higher = faster)")
    ax.set_ylabel("mean LPIPS vs eager  (lower = better)")
    ax.set_title("QLIP: speed × quality — point colour = arena judge vs eager")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(out_dir, "frontier.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_verdicts(entries, out_dir):
    """Per-config win/tie/lose bar chart from the ensemble judge (vs eager).
    Makes the 'pixels differ but judge says OK' story explicit."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = [e for e in entries
            if e.get("wins") is not None and e["name"] != "baseline"]
    if not rows:
        return None
    rows.sort(key=lambda e: (e.get("winrate_a") or 0))
    names = [e["name"] for e in rows]
    wins = [e.get("wins") or 0 for e in rows]
    ties = [e.get("ties") or 0 for e in rows]
    losses = [e.get("losses") or 0 for e in rows]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.32)))
    ax.barh(list(y), wins, color="#188038", label="win vs eager")
    ax.barh(list(y), ties, left=wins, color="#9aa0a6", label="tie")
    ax.barh(list(y), losses, left=[w + t for w, t in zip(wins, ties)],
            color="#d93025", label="lose")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("prompts (arena ensemble judge vs eager)")
    ax.set_title("Per-config verdict — judge preference over the eager baseline")
    ax.legend(fontsize=8, loc="lower right")
    path = os.path.join(out_dir, "verdicts.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_params(front, out_dir):
    """Show which node params the frontier configs actually use."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = []
    for e in front:
        ov = parse_overrides(e["config"])
        rows.append((e["name"], e["speedup"], ov))
    if not rows:
        return None
    keys = sorted({k for _, _, ov in rows for k in ov
                   if not k.endswith(".enable")})
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.1),
                                    max(3, len(rows) * 0.6)))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n}  ({s:.2f}×)" for n, s, _ in rows], fontsize=8)
    for yi, (_, _, ov) in enumerate(rows):
        for xi, k in enumerate(keys):
            v = ov.get(k, "")
            if v != "":
                ax.text(xi, yi, v, ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe",
                                  ec="#1a73e8", lw=0.8))
    ax.set_title("Frontier configs — which node parameters they use")
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    path = os.path.join(out_dir, "params.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_md(report, best, out_dir, meta):
    lines = [f"# QLIP REPORT — {meta.get('model')} on {meta.get('gpu')}", ""]
    lines.append(f"Baseline: `{meta.get('baseline')}` · quality tolerance "
                 f"LPIPS ≤ {report['quality_tolerance_lpips']} · "
                 f"{report['n_configs']} configs searched.\n")
    if best:
        lines += ["## Recommended config", "",
                  f"**{best['name']}** — **{best['speedup']:.2f}×** at "
                  f"mean LPIPS **{best['mean_lpips']:.3f}**"
                  + (f", ensemble Δelo {best['elo_delta']}"
                     if best.get("elo_delta") is not None else "") + ".", "",
                  "Reproduce (arena `--set`):", "", "```",
                  best.get("config", "") or "(baseline config)", "```", ""]
    else:
        lines += ["## Recommended config", "",
                  "No config met the quality tolerance — loosen `--tol` or search "
                  "more conservative settings.", ""]
    lines += ["## Frontier (speed × quality)", "",
              "| config | speedup | mean_lpips | Δelo | --set |",
              "|---|---|---|---|---|"]
    for e in report["frontier"]:
        star = " ⭐" if best and e["name"] == best["name"] else ""
        lines.append(f"| {e['name']}{star} | {e['speedup']:.2f}× | "
                     f"{e['mean_lpips']:.3f} | "
                     f"{e['elo_delta'] if e.get('elo_delta') is not None else '—'} | "
                     f"`{(e['config'] or '')[:80]}` |")
    lines += ["", "![frontier](frontier.png)", "", "![params](params.png)", "",
              "## Files", "- `report.json` — machine-readable full result",
              "- `frontier.png`, `params.png` — charts",
              "- `best_workflow.json` — the winning config as a ready-to-run "
              "ComfyUI API workflow (import & queue directly)", ""]
    path = os.path.join(out_dir, "REPORT.md")
    open(path, "w").write("\n".join(lines))
    return path


def _apply_overrides(wf, config_str):
    """Bake a config's --set overrides onto a workflow dict (in place)."""
    for key, val in parse_overrides(config_str).items():
        if "." not in key:
            continue
        nid, pin = key.split(".", 1)
        node = wf.get(nid)
        if not isinstance(node, dict) or "inputs" not in node:
            continue
        try:
            parsed = json.loads(val)      # true/0.5/"str", like arena's --set
        except Exception:
            parsed = val
        node["inputs"][pin] = parsed
    return wf


def write_workflow_for(entry, workflow_path, out_path):
    """Apply one config's --set onto the source API-workflow → a ready-to-run graph."""
    if not entry or not workflow_path or not os.path.exists(workflow_path):
        return None
    try:
        wf = json.load(open(workflow_path))
    except Exception:
        return None
    _apply_overrides(wf, entry.get("config", ""))
    json.dump(wf, open(out_path, "w"), indent=2)
    return out_path


def write_frontier_workflows(front, best, workflow_path, out_dir):
    """Write best_workflow.json AND one workflow per frontier point (workflows/),
    so the user can pick ANY operating point, not just the recommended best."""
    written = {}
    if best:
        p = write_workflow_for(best, workflow_path,
                               os.path.join(out_dir, "best_workflow.json"))
        if p:
            written["best"] = p
    wf_dir = os.path.join(out_dir, "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    for e in front:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in e["name"])
        p = write_workflow_for(e, workflow_path,
                               os.path.join(wf_dir, safe + ".json"))
        if p:
            written[e["name"]] = p
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="dir of arena export JSONs")
    ap.add_argument("--baseline", required=True, help="baseline run name (opponent)")
    ap.add_argument("--tol", type=float, default=0.35, help="max mean_lpips to accept")
    ap.add_argument("--out", required=True, help="output dir for report.json/png/md")
    ap.add_argument("--model", default="model")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--workflow", default=None,
                    help="source API-workflow; best config's --set is applied to it "
                         "and written as best_workflow.json (ready to queue)")
    ap.add_argument("--configs", default=None,
                    help="configs.jsonl written by the search runner: {name, set} per "
                         "line. Supplies the --set overrides (arena export omits them) "
                         "so best_workflow.json and the frontier's --set column work.")
    ap.add_argument("--point", default=None,
                    help="also write a ready-to-run workflow for THIS config name "
                         "(any point, not just best) → out/point_<name>.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    config_map = load_config_map(args.configs)
    entries = load_entries(args.results, args.baseline, config_map)
    if not entries:
        raise SystemExit(f"no arena export JSONs found in {args.results} "
                         f"(opponent={args.baseline}) — run judge+export first")
    front = pareto_front(entries)
    best = recommend(front, args.tol)
    meta = {"model": args.model, "gpu": args.gpu, "baseline": args.baseline}

    jpath, report = write_json(entries, front, best, args.tol, args.out, meta)
    fpath = plot_frontier(entries, front, best, args.out)
    ppath = plot_params(front, args.out)
    vpath = plot_verdicts(entries, args.out)
    mpath = write_md(report, best, args.out, meta)
    # best_workflow.json + one workflow per frontier point (workflows/)
    wf_written = write_frontier_workflows(front, best, args.workflow, args.out)
    wpath = wf_written.get("best")
    # optional: a workflow for a user-requested point (any config, not just frontier)
    ptpath = None
    if args.point:
        pe = next((e for e in entries if e["name"] == args.point), None)
        if pe:
            ptpath = write_workflow_for(
                pe, args.workflow,
                os.path.join(args.out, "point_%s.json" % args.point))

    print(f"configs: {len(entries)} | frontier: {len(front)}")
    if best:
        print(f"BEST: {best['name']}  {best['speedup']:.2f}x  "
              f"LPIPS {best['mean_lpips']:.3f}")
    else:
        print(f"BEST: none within tol={args.tol}")
    print(f"wrote: {jpath}")
    print(f"       {mpath}")
    if fpath:
        print(f"       {fpath}")
    if ppath:
        print(f"       {ppath}")
    if vpath:
        print(f"       {vpath}  <- win/tie/lose vs eager (arena judge)")
    if wpath:
        print(f"       {wpath}  <- best workflow (ready to queue)")
    n_front_wf = len([k for k in wf_written if k != "best"])
    if n_front_wf:
        print(f"       {os.path.join(args.out, 'workflows')}/  "
              f"<- {n_front_wf} frontier-point workflows (pick any point)")
    if ptpath:
        print(f"       {ptpath}  <- requested point '{args.point}'")
    print("QLIP_REPORT_DONE")


if __name__ == "__main__":
    main()

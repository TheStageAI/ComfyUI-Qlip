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
recommended config: with --fpcharts <fpcharts_data.json> (the normal case, see
AGENTS.md §4) = the fastest FAITHFUL config that passes the arena's defect gate
and is statistically tied on the fpcharts quality rank with the best-quality one;
configs that re-decide the composition are listed apart as "creative" points and
gate failures as rejected. Without --fpcharts (legacy) = fastest config with
mean_lpips <= --tol.

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


def pareto_front(entries, qkey="mean_lpips"):
    """Non-dominated by (speedup ↑, quality ↓ where lower is better).
    ``qkey`` is the quality coordinate: ``mean_lpips`` (legacy) or
    ``quality_rank`` (fpcharts). Configs without it are left out."""
    scored = [e for e in entries if e.get(qkey) is not None]
    front = []
    for e in scored:
        dominated = any(
            o is not e
            and o["speedup"] >= e["speedup"]
            and o[qkey] <= e[qkey]
            and (o["speedup"] > e["speedup"] or o[qkey] < e[qkey])
            for o in scored)
        if not dominated:
            front.append(e)
    front.sort(key=lambda e: e["speedup"])
    return front


def recommend(front, tol):
    """Legacy rule: fastest config on the frontier within the LPIPS tolerance."""
    ok = [e for e in front if e["mean_lpips"] is not None and e["mean_lpips"] <= tol]
    if not ok:
        return None
    return max(ok, key=lambda e: e["speedup"])


# ---------------------------------------------------------------------------
# fpcharts-based decision (the arena's degradation judge decides; LPIPS and
# Elo explain). See AGENTS.md §4 / §4b.
# ---------------------------------------------------------------------------

DEFAULT_GATE = ["texture=0.15", "texture_gain", "noise", "halo", "mesh", "haze"]


def load_fpcharts(path):
    """fpcharts_data.json -> (doc, summary). ``summary`` is written by
    `arena fpcharts` (qlip-arena >= the summary export); None if absent."""
    doc = json.load(open(path))
    return doc, doc.get("summary")


def _p_stat(vals, stat):
    import numpy as np
    a = np.asarray(vals, float)
    if stat == "median":
        return float(np.median(a))
    if stat == "p90":
        return float(np.quantile(a, 0.9))
    return float(np.mean(a))


def evaluate_gate(doc, summary, cfg, limits, stat="median", tail_mult=2.0):
    """Re-implements `arena gate` on the fpcharts data (so this tool does not
    need qlip_arena importable). Two checks per axis in ``limits``:
      typical prompt : ``stat`` (median) of the per-prompt values <= limit
                       (limit None = the arena's noticeability threshold, i.e.
                       the typical prompt is at most "slight");
      bad tail       : p90 <= tail_mult x limit (the arena's "strong" band is
                       2 x threshold: no strong defect on the worst 10 %).
    tail_mult <= 0 disables the tail check.
    -> (passed, [failing 'axis stat value > limit' strings])"""
    data = doc.get("data", {})
    thr = (summary or {}).get("thresholds", {})
    fails = []
    for spec in limits:
        ax, lim = (spec.split("=", 1) + [None])[:2]
        ax = ax.strip()
        lim = float(lim) if lim is not None else thr.get(ax)
        if lim is None:
            continue
        vals = [v for cat in data.get(ax, {}).get(cfg, {}).values() for v in cat]
        if not vals:
            continue
        v = _p_stat(vals, stat)
        if v > lim:
            fails.append(f"{ax} {stat} {v:+.3f} > {lim:.3f}")
        if tail_mult and tail_mult > 0:
            t = _p_stat(vals, "p90")
            if t > tail_mult * lim:
                fails.append(f"{ax} p90 {t:+.3f} > {tail_mult * lim:.3f} (strong)")
    return (not fails), fails


def attach_fpcharts(entries, doc, summary, gate_limits, gate_stat="median",
                    replaced_max=0.25, moved_max=0.5, tail_mult=2.0):
    """Add the fpcharts verdict to each entry: quality_rank, rank_ci, p_best,
    per-axis wins/losses, comparison mode, gate result and the resulting
    class: 'faithful' (eligible), 'creative' (re-decides the composition —
    judged separately), or 'rejected' (fails the gate)."""
    cfgs = (summary or {}).get("configs", {})
    for e in entries:
        s = cfgs.get(e["name"])
        if not s:
            e["fp_class"] = None
            continue
        e["quality_rank"] = s.get("quality_rank")
        e["rank_ci"] = s.get("ci")
        e["p_best"] = s.get("p_best")
        e["pickscore_wr"] = s.get("pickscore_wr")
        ranks = s.get("axis_ranks", {}) or {}
        n_cfg = len(cfgs)
        e["axes_won"] = sorted(ax for ax, rk in ranks.items() if rk is not None and rk <= 1.0)
        e["axes_lost"] = sorted(ax for ax, rk in ranks.items()
                                if rk is not None and n_cfg > 1 and rk >= n_cfg)
        e["axes_ranked"] = len(ranks)
        e["mode"] = s.get("mode", {})
        m = e["mode"]
        n = max(m.get("n", 0), 1)
        e["re_deciding"] = (m.get("replaced", 0) / n > replaced_max
                            or m.get("moved", 0) / n > moved_max)
        e["gate_passed"], e["gate_fails"] = evaluate_gate(doc, summary, e["name"],
                                                          gate_limits, gate_stat,
                                                          tail_mult)
        if not e["gate_passed"]:
            e["fp_class"] = "rejected"
        elif e["re_deciding"]:
            e["fp_class"] = "creative"
        else:
            e["fp_class"] = "faithful"
    return entries


def recommend_fp(front, slack=1.0):
    """Fastest faithful frontier point that is statistically tied with the
    best-quality one (rank CI overlaps the leader's CI); if none overlaps,
    the fastest within ``slack`` rank of the leader, flagged as a compromise.
    -> (entry or None, reason string)"""
    elig = [e for e in front if e.get("quality_rank") is not None]
    if not elig:
        return None, "no faithful config passed the gate"
    leader = min(elig, key=lambda e: e["quality_rank"])
    lo_l, hi_l = leader.get("rank_ci") or (leader["quality_rank"],
                                           leader["quality_rank"])
    tied = [e for e in elig
            if (e.get("rank_ci") or (e["quality_rank"], e["quality_rank"]))[0] <= hi_l]
    if tied:
        best = max(tied, key=lambda e: e["speedup"])
        return best, (f"fastest config whose quality-rank CI overlaps the "
                      f"leader's ({leader['name']} {leader['quality_rank']:.2f} "
                      f"[{lo_l:.1f}–{hi_l:.1f}]) — a statistical tie on quality")
    near = [e for e in elig if e["quality_rank"] <= leader["quality_rank"] + slack]
    best = max(near, key=lambda e: e["speedup"])
    return best, (f"COMPROMISE: no faster config is statistically tied with the "
                  f"leader ({leader['name']}); fastest within {slack:.1f} rank "
                  f"of it")


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
        "decision": meta.get("decision"),          # fpcharts rule details or None
        "n_configs": len(entries),
        "recommended": best,
        "frontier": front,
        "creative": [e for e in entries if e.get("fp_class") == "creative"],
        "rejected_by_gate": [e for e in entries if e.get("fp_class") == "rejected"],
        "all_configs": sorted(entries, key=lambda e: -e["speedup"]),
        "dominated": [e["name"] for e in entries if e["name"] not in front_names
                      and e.get(meta.get("qkey", "mean_lpips")) is not None
                      and e.get("fp_class") in (None, "faithful")],
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
    qkey = "quality_rank" if any(e.get("quality_rank") is not None
                                 for e in entries) else "mean_lpips"
    scored = [e for e in entries if e.get(qkey) is not None]
    if not scored:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6))
    # Colour each point by the ARENA JUDGE's preference verdict vs eager (ensemble):
    # green = judge prefers the accelerated output, grey = tie (indistinguishable to
    # the judge even though pixels differ), red = judge prefers eager. This shows the
    # quality coordinate AND whether a human-preference judge still calls it OK.
    vcol = {"win": "#188038", "tie": "#9aa0a6", "lose": "#d93025", None: "#c9c9c9"}
    faithful = [e for e in scored if e.get("fp_class") in (None, "faithful")]
    for v in ("win", "tie", "lose", None):
        pts = [e for e in faithful if _verdict(e) == v]
        if not pts:
            continue
        lbl = {"win": "judge: win vs eager", "tie": "judge: tie",
               "lose": "judge: lose", None: "no judge"}[v]
        ax.scatter([e["speedup"] for e in pts], [e[qkey] for e in pts],
                   c=vcol[v], s=54, label=lbl, zorder=2, edgecolor="white", lw=0.5)
    if qkey == "quality_rank":
        # bootstrap CI of the rank as a thin vertical bar
        for e in faithful:
            ci = e.get("rank_ci")
            if ci:
                ax.plot([e["speedup"]] * 2, ci, color="#9aa0a6", lw=0.8, zorder=1)
        creative = [e for e in scored if e.get("fp_class") == "creative"]
        if creative:
            ax.scatter([e["speedup"] for e in creative],
                       [e[qkey] for e in creative], facecolors="none",
                       edgecolors="#7b1fa2", s=70, lw=1.4, zorder=2,
                       label="re-decides the scene (creative point)")
        rejected = [e for e in scored if e.get("fp_class") == "rejected"]
        if rejected:
            ax.scatter([e["speedup"] for e in rejected],
                       [e[qkey] for e in rejected], marker="x", c="#d93025",
                       s=60, lw=1.4, zorder=2, label="fails the defect gate")
        for e in creative + rejected:
            ax.annotate(e["name"], (e["speedup"], e[qkey]), fontsize=6.5,
                        color="#5f6368", xytext=(4, -9), textcoords="offset points")
    fx = [e["speedup"] for e in front]
    fy = [e[qkey] for e in front]
    ax.plot(fx, fy, "-", color="#1a73e8", lw=2, label="Pareto frontier", zorder=3)
    if best:
        ax.scatter([best["speedup"]], [best[qkey]], marker="*",
                   s=440, color="#f9ab00", edgecolor="#3c4043", lw=1.2,
                   label="recommended", zorder=4)
    for e in front:
        ax.annotate(e["name"], (e["speedup"], e[qkey]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("speedup ×  (higher = faster)")
    if qkey == "quality_rank":
        ax.set_ylabel("arena quality rank, fpcharts  (1 = best; bar = 95 % CI)")
        ax.set_title("QLIP: speed × quality — defect-axis rank; colour = preference judge")
    else:
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


def _fmt_lpips(e):
    return f"{e['mean_lpips']:.3f}" if e.get("mean_lpips") is not None else "—"


def _fmt_rank(e):
    if e.get("quality_rank") is None:
        return "—"
    ci = e.get("rank_ci")
    s = f"{e['quality_rank']:.2f}"
    if ci:
        s += f" [{ci[0]:.1f}–{ci[1]:.1f}]"
    if e.get("p_best") is not None:
        s += f", p_best {e['p_best']:.2f}"
    return s


def _fmt_mode(e):
    m = e.get("mode") or {}
    if not m.get("n"):
        return "—"
    return (f"{m.get('aligned', 0)}/{m['n']} aligned, {m.get('moved', 0)} moved, "
            f"{m.get('replaced', 0)} replaced")


def write_md(report, best, out_dir, meta):
    dec = report.get("decision")
    lines = [f"# QLIP REPORT — {meta.get('model')} on {meta.get('gpu')}", ""]
    if dec:
        lines.append(
            f"Baseline: `{meta.get('baseline')}` · {report['n_configs']} configs "
            f"searched · **decision = arena degradation judge (fpcharts)**: a config "
            f"must pass the defect gate ({', '.join(dec['gate'])}: {dec['gate_stat']} "
            f"≤ threshold"
            + (f" and p90 ≤ {dec['tail_mult']:g}× threshold, i.e. no strong defect "
               f"on the worst 10 % of prompts" if dec.get("tail_mult") else "") + ") "
            f"and keep the composition (scene replaced ≤ {dec['replaced_max']:.0%} of "
            f"prompts, layout moved ≤ {dec['moved_max']:.0%}); among those the Pareto "
            f"front is speed × fpcharts quality rank (weighted rank over the defect "
            f"axes + preference, 1 = best) and the recommendation is the fastest point "
            f"statistically tied with the best-quality one. LPIPS (≤ "
            f"{report['quality_tolerance_lpips']} was the legacy gate) and Δelo are "
            f"reported for reference. Ranking weights: "
            + ", ".join(f"{k} {v:g}" for k, v in sorted(dec["weights"].items(),
                                                        key=lambda kv: -kv[1])
                        if v != 1.0) + ", others 1.\n")
    else:
        lines.append(f"Baseline: `{meta.get('baseline')}` · quality tolerance "
                     f"LPIPS ≤ {report['quality_tolerance_lpips']} · "
                     f"{report['n_configs']} configs searched.\n")
    if best:
        head = (f"**{best['name']}** — **{best['speedup']:.2f}×**")
        if best.get("quality_rank") is not None:
            head += f" at quality rank **{_fmt_rank(best)}**"
        head += f", mean LPIPS {_fmt_lpips(best)}"
        if best.get("elo_delta") is not None:
            head += f", ensemble Δelo {best['elo_delta']}"
        lines += ["## Recommended config", "", head + ".", ""]
        if dec:
            lines.append(f"Why this one: {dec['reason']}.")
            won, lost = best.get("axes_won") or [], best.get("axes_lost") or []
            lines.append(f"Axes ranked: {best.get('axes_ranked', 0)}; best on "
                         f"{len(won)} ({', '.join(won) or '—'}); worst on "
                         f"{len(lost)} ({', '.join(lost) or '—'}). Comparison mode: "
                         f"{_fmt_mode(best)}. Gate: passed.")
            lines.append("")
        lines += ["Reproduce (arena `--set`):", "", "```",
                  best.get("config", "") or "(baseline config)", "```", ""]
    else:
        lines += ["## Recommended config", "",
                  ("No faithful config passed the defect gate — search more "
                   "conservative settings." if dec else
                   "No config met the quality tolerance — loosen `--tol` or search "
                   "more conservative settings."), ""]
    if dec:
        lines += ["## Frontier (speed × arena quality rank), faithful configs", "",
                  "| config | speedup | quality rank [95 % CI] | best on / worst on "
                  "(axes) | mode | mean_lpips | Δelo | --set |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in report["frontier"]:
            star = " ⭐" if best and e["name"] == best["name"] else ""
            lines.append(
                f"| {e['name']}{star} | {e['speedup']:.2f}× | {_fmt_rank(e)} | "
                f"{len(e.get('axes_won') or [])} / {len(e.get('axes_lost') or [])} "
                f"of {e.get('axes_ranked', 0)} | {_fmt_mode(e)} | {_fmt_lpips(e)} | "
                f"{e['elo_delta'] if e.get('elo_delta') is not None else '—'} | "
                f"`{(e['config'] or '')[:80]}` |")
        if report.get("creative"):
            lines += ["", "## Re-deciding configs (creative points, judged separately)",
                      "", "These change the composition on too many prompts to be "
                      "compared as a degradation of the baseline; their LPIPS is not a "
                      "quality measure. Only structure and the learned axes apply; "
                      "check `arena adherence` before offering them.", "",
                      "| config | speedup | quality rank | mode | Δelo | --set |",
                      "|---|---|---|---|---|---|"]
            for e in sorted(report["creative"], key=lambda e: -e["speedup"]):
                lines.append(
                    f"| {e['name']} | {e['speedup']:.2f}× | {_fmt_rank(e)} | "
                    f"{_fmt_mode(e)} | "
                    f"{e['elo_delta'] if e.get('elo_delta') is not None else '—'} | "
                    f"`{(e['config'] or '')[:80]}` |")
        if report.get("rejected_by_gate"):
            lines += ["", "## Rejected by the defect gate", "",
                      "| config | speedup | failing axes | mean_lpips | Δelo |",
                      "|---|---|---|---|---|"]
            for e in sorted(report["rejected_by_gate"], key=lambda e: -e["speedup"]):
                lines.append(
                    f"| {e['name']} | {e['speedup']:.2f}× | "
                    f"{'; '.join(e.get('gate_fails') or [])} | {_fmt_lpips(e)} | "
                    f"{e['elo_delta'] if e.get('elo_delta') is not None else '—'} |")
    else:
        lines += ["## Frontier (speed × quality)", "",
                  "| config | speedup | mean_lpips | Δelo | --set |",
                  "|---|---|---|---|---|"]
        for e in report["frontier"]:
            star = " ⭐" if best and e["name"] == best["name"] else ""
            lines.append(f"| {e['name']}{star} | {e['speedup']:.2f}× | "
                         f"{_fmt_lpips(e)} | "
                         f"{e['elo_delta'] if e.get('elo_delta') is not None else '—'} | "
                         f"`{(e['config'] or '')[:80]}` |")
    if report.get("not_shown"):
        ns = report["not_shown"]
        lines += ["", f"*{len(ns)} other configs were measured and are dominated, no-ops "
                  f"or gate failures; they are kept out of the charts and workflows/ to keep "
                  f"the report readable (all data in `report.json`): "
                  + ", ".join(ns) + ".*"]
    lines += ["", "![frontier](frontier.png)", "", "![params](params.png)", "",
              "## Files", "- How to read this folder: `tools/READING_THE_AGENT_REPORT.md` (ComfyUI-Qlip); the arena's "
              "pages and every metric: qlip-arena `docs/READING_THE_REPORT.md`, formulas `docs/METRICS.md`",
              "- `report.json` — machine-readable full result",
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


# ---- UI-format sibling output --------------------------------------------
# The API graph above is what you POST to /prompt, but it does NOT open in the
# ComfyUI canvas (the UI wants {nodes:[...], links:[...]} with widgets_values).
# So we ALSO emit a UI-format copy per point, with the same --set baked in, so
# the user can drag it into the UI to inspect/tweak. Mapping node.pin -> the
# right widgets_values slot needs the node's widget order, which UI json does
# not store by name; we take it from a live /object_info dump when available,
# and fall back to leaving the widget untouched (never corrupt the graph).

def _load_object_info(oinfo):
    """Return the /object_info dict from a path or URL, or None."""
    if not oinfo:
        return None
    try:
        if oinfo.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(oinfo, timeout=15) as r:
                return json.load(r)
        return json.load(open(oinfo))
    except Exception:
        return None


def _widget_input_order(object_info, class_type):
    """Ordered list of the WIDGET input names for a node class (the ones that
    become widgets_values, i.e. non-link inputs), from /object_info. The UI
    lays widgets_values in required-then-optional declaration order, skipping
    inputs that are wired as links (model/latent/etc. — type is a known socket,
    not a widget)."""
    spec = (object_info or {}).get(class_type, {})
    inp = spec.get("input", {})
    order = []
    LINKY = {"MODEL", "LATENT", "CONDITIONING", "VAE", "CLIP", "IMAGE",
             "SAMPLER", "SIGMAS", "GUIDER", "NOISE", "AUDIO", "VIDEO"}
    for grp in ("required", "optional"):
        for name, spec_v in inp.get(grp, {}).items():
            t = spec_v[0] if isinstance(spec_v, (list, tuple)) and spec_v else spec_v
            # a link socket is a single known type string (not a list of choices,
            # not a primitive widget type). Everything else renders as a widget.
            if isinstance(t, str) and t in LINKY:
                continue
            order.append(name)
    return order


def _apply_overrides_ui(ui_wf, config_str, object_info):
    """Bake a config's --set onto a UI-format graph (in place). node.pin -> the
    widgets_values slot via /object_info widget order; skip if we can't map it
    safely (never corrupt the graph)."""
    overrides = parse_overrides(config_str)
    if not overrides:
        return ui_wf
    by_id = {str(n.get("id")): n for n in ui_wf.get("nodes", [])}
    for key, val in overrides.items():
        if "." not in key:
            continue
        nid, pin = key.split(".", 1)
        node = by_id.get(str(nid))
        if not node or "widgets_values" not in node:
            continue
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = val
        order = _widget_input_order(object_info, node.get("type"))
        wv = node["widgets_values"]
        if pin in order and isinstance(wv, list) and order.index(pin) < len(wv):
            wv[order.index(pin)] = parsed          # name-mapped slot (robust)
        # else: unknown widget order -> leave untouched rather than guess
    return ui_wf


def write_ui_workflow_for(entry, ui_workflow_path, object_info, out_path):
    """Apply one config's --set onto the source UI-workflow → a canvas-openable graph."""
    if not entry or not ui_workflow_path or not os.path.exists(ui_workflow_path):
        return None
    try:
        ui = json.load(open(ui_workflow_path))
    except Exception:
        return None
    if "nodes" not in ui:            # not a UI graph — skip quietly
        return None
    _apply_overrides_ui(ui, entry.get("config", ""), object_info)
    json.dump(ui, open(out_path, "w"), indent=2)
    return out_path


def write_frontier_workflows(front, best, workflow_path, out_dir,
                             ui_workflow_path=None, object_info=None):
    """Write best_workflow.json AND one workflow per frontier point (workflows/),
    so the user can pick ANY operating point, not just the recommended best.
    When a UI-format source workflow is given, ALSO write *_ui.json siblings
    (best_workflow_ui.json, workflows/<name>_ui.json) that open in the ComfyUI
    canvas — same --set baked in."""
    written = {}

    def _pair(entry, api_out, ui_out):
        p = write_workflow_for(entry, workflow_path, api_out)
        if ui_workflow_path:
            write_ui_workflow_for(entry, ui_workflow_path, object_info, ui_out)
        return p

    if best:
        p = _pair(best, os.path.join(out_dir, "best_workflow.json"),
                  os.path.join(out_dir, "best_workflow_ui.json"))
        if p:
            written["best"] = p
    wf_dir = os.path.join(out_dir, "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    for e in front:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in e["name"])
        p = _pair(e, os.path.join(wf_dir, safe + ".json"),
                  os.path.join(wf_dir, safe + "_ui.json"))
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
    ap.add_argument("--ui-workflow", default=None,
                    help="source UI-format workflow (the {nodes,links} graph). When "
                         "given, ALSO writes *_ui.json siblings that open in the ComfyUI "
                         "canvas, with the same --set baked in.")
    ap.add_argument("--object-info", default=None,
                    help="path to a /object_info dump OR its URL (e.g. "
                         "http://127.0.0.1:8188/object_info). Used to map node.pin -> the "
                         "correct widgets_values slot when writing the UI workflow.")
    ap.add_argument("--configs", default=None,
                    help="configs.jsonl written by the search runner: {name, set} per "
                         "line. Supplies the --set overrides (arena export omits them) "
                         "so best_workflow.json and the frontier's --set column work.")
    ap.add_argument("--point", default=None,
                    help="also write a ready-to-run workflow for THIS config name "
                         "(any point, not just best) → out/point_<name>.json")
    ap.add_argument("--fpcharts", default=None,
                    help="fpcharts_data.json from `arena fpcharts` over the searched "
                         "configs. When given, the arena's degradation judge DECIDES: "
                         "gate on the defect tails, faithful vs re-deciding mode, "
                         "Pareto on speed x quality rank, recommendation = fastest "
                         "point statistically tied with the best-quality one. LPIPS "
                         "and Elo become reference columns.")
    ap.add_argument("--gate-max", action="append", default=None,
                    help="defect-gate limit 'axis=value' or 'axis' (= the arena's "
                         "noticeability threshold); repeatable. Default: "
                         + " ".join(DEFAULT_GATE))
    ap.add_argument("--gate-stat", default="median", choices=("median", "mean", "p90"),
                    help="statistic of the per-prompt values checked against the "
                         "limit (typical prompt); default median")
    ap.add_argument("--gate-tail", type=float, default=2.0,
                    help="ALSO require p90 <= this x limit (the arena's 'strong' band "
                         "is 2 x threshold: no strong defect on the worst 10 %% of "
                         "prompts). 0 disables the tail check.")
    ap.add_argument("--replaced-max", type=float, default=0.25,
                    help="share of prompts with the scene replaced above which a "
                         "config is a re-deciding (creative) point")
    ap.add_argument("--moved-max", type=float, default=0.5,
                    help="share of prompts with the layout moved above which a "
                         "config is a re-deciding (creative) point")
    ap.add_argument("--report-set", default="frontier", choices=("frontier", "all"),
                    help="which configs the charts, collage list and workflows/ cover: "
                         "'frontier' (default) = the faithful Pareto front + the creative "
                         "points + the recommended one; dominated / no-op / gate-failed "
                         "configs are counted and listed by name only (full data stays in "
                         "report.json). 'all' = every config on every chart (the old "
                         "behaviour; unreadable beyond ~10 configs).")
    ap.add_argument("--rank-slack", type=float, default=1.0,
                    help="if no faster config is tied with the quality leader, "
                         "accept the fastest within this many rank units (compromise)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    config_map = load_config_map(args.configs)
    entries = load_entries(args.results, args.baseline, config_map)
    if not entries:
        raise SystemExit(f"no arena export JSONs found in {args.results} "
                         f"(opponent={args.baseline}) — run judge+export first")
    meta = {"model": args.model, "gpu": args.gpu, "baseline": args.baseline,
            "qkey": "mean_lpips", "decision": None}
    if args.fpcharts:
        doc, summary = load_fpcharts(args.fpcharts)
        if not summary:
            raise SystemExit(f"{args.fpcharts} has no 'summary' block — regenerate "
                             f"it with a qlip-arena that exports the summary "
                             f"(arena fpcharts ...)")
        gate = args.gate_max or DEFAULT_GATE
        attach_fpcharts(entries, doc, summary, gate, args.gate_stat,
                        args.replaced_max, args.moved_max, args.gate_tail)
        unscored = [e["name"] for e in entries if e.get("fp_class") is None]
        if unscored:
            print(f"WARNING: not in fpcharts (left out of the decision): "
                  f"{', '.join(unscored)}")
        faithful = [e for e in entries if e.get("fp_class") == "faithful"]
        front = pareto_front(faithful, qkey="quality_rank")
        best, reason = recommend_fp(front, args.rank_slack)
        legacy = recommend(pareto_front(entries), args.tol)
        if legacy and (not best or legacy["name"] != best["name"]):
            why = (legacy.get("gate_fails") or
                   (["re-decides the composition: " + _fmt_mode(legacy)]
                    if legacy.get("re_deciding") else ["dominated on quality rank"]))
            reason += (f". The LPIPS-only rule would have picked {legacy['name']} "
                       f"({legacy['speedup']:.2f}x, LPIPS {_fmt_lpips(legacy)}); "
                       f"disqualified by: {'; '.join(why)}")
        meta.update(qkey="quality_rank",
                    decision={"gate": gate, "gate_stat": args.gate_stat,
                              "replaced_max": args.replaced_max,
                              "moved_max": args.moved_max,
                              "rank_slack": args.rank_slack,
                              "tail_mult": args.gate_tail,
                              # only the axes measured on this modality
                              "weights": {k: v for k, v in
                                          summary.get("weights", {}).items()
                                          if k == "pickscore_wr"
                                          or k in doc.get("data", {})},
                              "n_prompts": summary.get("n_prompts"),
                              "reason": reason})
    else:
        front = pareto_front(entries)
        best = recommend(front, args.tol)

    jpath, report = write_json(entries, front, best, args.tol, args.out, meta)
    fpath = plot_frontier(entries if args.report_set == "all" else
                          [e for e in entries if e.get("fp_class") != "faithful"
                           or e in front], front, best, args.out)
    ppath = plot_params(front, args.out)
    if args.report_set == "frontier":
        keep = {e["name"] for e in front} | {e["name"] for e in report.get("creative", [])}
        if best:
            keep.add(best["name"])
        shown = [e for e in entries if e["name"] in keep]
        report["report_set"] = sorted(keep)
        report["not_shown"] = sorted(e["name"] for e in entries if e["name"] not in keep)
        json.dump(report, open(jpath, "w"), indent=2)
    else:
        shown = entries
    vpath = plot_verdicts(shown, args.out)
    mpath = write_md(report, best, args.out, meta)
    # best_workflow.json + one workflow per frontier point (workflows/);
    # + *_ui.json siblings when a UI-format source workflow is provided.
    object_info = _load_object_info(args.object_info)
    wf_written = write_frontier_workflows(front + report.get("creative", []), best,
                                          args.workflow, args.out,
                                          ui_workflow_path=args.ui_workflow,
                                          object_info=object_info)
    wpath = wf_written.get("best")
    # optional: a workflow for a user-requested point (any config, not just frontier)
    ptpath = None
    if args.point:
        pe = next((e for e in entries if e["name"] == args.point), None)
        if pe:
            ptpath = write_workflow_for(
                pe, args.workflow,
                os.path.join(args.out, "point_%s.json" % args.point))
            if args.ui_workflow:
                write_ui_workflow_for(
                    pe, args.ui_workflow, object_info,
                    os.path.join(args.out, "point_%s_ui.json" % args.point))

    print(f"configs: {len(entries)} | frontier: {len(front)}"
          + (f" | creative: {len(report['creative'])} | rejected by gate: "
             f"{len(report['rejected_by_gate'])}" if meta.get("decision") else ""))
    if best:
        print(f"BEST: {best['name']}  {best['speedup']:.2f}x  "
              f"LPIPS {_fmt_lpips(best)}"
              + (f"  quality rank {_fmt_rank(best)}"
                 if best.get("quality_rank") is not None else ""))
        if meta.get("decision"):
            print(f"      {meta['decision']['reason']}")
    else:
        print("BEST: none" + ("" if meta.get("decision") else f" within tol={args.tol}"))
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

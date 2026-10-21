"""QLIP frontier collage — a single PNG comparing eager vs key Pareto points.

Builds a grid: rows = prompts, columns = configs (eager first as reference, then a
few frontier points INCLUDING the recommended best, which is highlighted). Each column
header shows config / speedup / LPIPS / judge-verdict. This is the "see the quality,
not just the number" deliverable, as a shareable PNG (no HTML to open).

Which columns: eager, the recommended best (always, boxed + "★ BEST"), the max-quality
frontier point, and the max-speed frontier point under the gate — deduped. Override with
--configs "a,b,c". Which rows: the first --rows prompts that all chosen configs share.

Usage (agent calls after the search):
    python tools/qlip_collage.py \
        --store $QLIP_ARENA_ROOT --report <out>/report.json \
        --out <out>/collage.png --rows 3 --seed 1000
"""
import argparse
import glob
import json
import os


def _img_path(store, config, prompt_id, seed):
    base = os.path.join(store, "runs", config, "images")
    for ext in ("png", "webp", "jpg", "jpeg"):
        p = os.path.join(base, "%s_s%s.%s" % (prompt_id, seed, ext))
        if os.path.exists(p):
            return p
    # fallback: any file starting with the prompt id
    hits = glob.glob(os.path.join(base, "%s_s%s.*" % (prompt_id, seed)))
    return hits[0] if hits else None


_VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".avi")


def _load_frame(path):
    """Return a PIL RGB image for an image file, OR a representative frame for a
    video file (middle frame). Video (arena stores .mp4 for video models) can't be
    opened by PIL, so pull one frame via imageio."""
    from PIL import Image

    if path.lower().endswith(_VIDEO_EXT):
        import imageio

        reader = imageio.get_reader(path)
        try:
            try:
                n = reader.get_length()
            except Exception:
                n = None
            frames = []
            it = reader.iter_data()
            # grab up to the middle frame (cheap; avoids reading the whole clip)
            target = (n // 2) if isinstance(n, int) and n > 0 and n < 1e6 else 8
            for i, fr in enumerate(it):
                frames.append(fr)
                if i >= target:
                    break
            arr = frames[len(frames) // 2] if frames else None
            return Image.fromarray(arr).convert("RGB") if arr is not None else None
        finally:
            reader.close()
    return Image.open(path).convert("RGB")


def _prompt_ids(store, config, seed, limit):
    base = os.path.join(store, "runs", config, "images")
    ids = []
    for f in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if ("_s%s." % seed) in f:
            ids.append(f.split("_s%s." % seed)[0])
    return ids[:limit]


def pick_columns(report, baseline, explicit):
    """Return ordered list of (config_name, label) columns."""
    entries = {e["name"]: e for e in report.get("all_configs", [])}
    front = report.get("frontier", [])
    best = report.get("recommended")
    cols = [(baseline, "eager (reference)")]
    if explicit:
        for n in explicit:
            n = n.strip()
            if n and n != baseline:
                cols.append((n, n))
        return cols, (best["name"] if best else None)
    tol = report.get("quality_tolerance_lpips", 0.35)
    chosen = []
    if best:
        chosen.append(best["name"])
    # candidates that actually pass the quality gate (usable configs, speedup>1)
    fq = [
        e
        for e in front
        if e.get("mean_lpips") is not None
        and e["speedup"] > 1.01
        and e["mean_lpips"] <= tol
    ]
    # max-quality under gate (lowest lpips)
    if fq:
        mq = min(fq, key=lambda e: e["mean_lpips"])
        if mq["name"] not in chosen:
            chosen.append(mq["name"])
    # max-speed UNDER GATE (a usable fast config, not a degraded one)
    if fq:
        ms = max(fq, key=lambda e: e["speedup"])
        if ms["name"] not in chosen:
            chosen.append(ms["name"])
    # one over-gate extreme-speed point for contrast (clearly labelled by verdict),
    # so the user sees WHY the gate exists — but never in place of a usable one.
    over = [
        e
        for e in front
        if e.get("mean_lpips") is not None
        and e["mean_lpips"] > tol
        and e["speedup"] > 1.01
    ]
    if over and len(chosen) < 4:
        ms2 = max(over, key=lambda e: e["speedup"])
        if ms2["name"] not in chosen:
            chosen.append(ms2["name"])
    for n in chosen:
        e = entries.get(n, {})
        v = _verdict_str(e)
        lab = "%s\n%.2fx · LPIPS %.2f%s" % (
            n,
            e.get("speedup", 0),
            e.get("mean_lpips", 0),
            (" · " + v) if v else "",
        )
        cols.append((n, lab))
    return cols, (best["name"] if best else None)


def _verdict_str(e):
    w = e.get("winrate_a")
    if w is None:
        return ""
    if w > 0.55:
        return "judge:win"
    if w < 0.45:
        return "judge:lose"
    return "judge:tie"


def build(store, report, out_path, rows, seed, thumb, explicit_cols):
    from PIL import Image, ImageDraw, ImageFont

    baseline = report.get("baseline") or "eager"
    cols, best_name = pick_columns(report, baseline, explicit_cols)

    # rows = prompt ids shared by all chosen configs
    pid_sets = []
    for cfg, _ in cols:
        pid_sets.append(set(_prompt_ids(store, cfg, seed, 999)))
    shared = set.intersection(*pid_sets) if pid_sets else set()

    # skip prompts whose reference output is a blank / safety-filter card —
    # a row of five grey placeholders shows nothing
    def _blank(path):
        try:
            import numpy as np
            from PIL import Image

            a = np.asarray(
                Image.open(path).convert("L").resize((128, 128)), dtype=np.float32
            )
            return float(a.std() / 255.0) < 0.035
        except Exception:
            return True

    live = [
        pid
        for pid in sorted(shared)
        if not _blank(_img_path(store, cols[0][0], pid, seed))
    ]
    prompt_ids = live[:rows]
    if not prompt_ids:
        raise SystemExit(
            "no shared prompts across the chosen configs — check the store"
        )

    pad, header_h, label_h = 8, 54, 22
    ncol, nrow = len(cols), len(prompt_ids)
    # cell width follows the media aspect (portrait outputs would otherwise
    # leave an empty strip inside the square cell)
    cell_w = thumb
    try:
        im0 = _load_frame(_img_path(store, cols[0][0], prompt_ids[0], seed))
        if im0 is not None and im0.height > 0 and im0.width < im0.height:
            cell_w = max(64, int(round(thumb * im0.width / im0.height)))
    except Exception:
        pass
    W = ncol * (cell_w + pad) + pad
    H = header_h + nrow * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
        fontb = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = fontb = ImageFont.load_default()

    for ci, (cfg, label) in enumerate(cols):
        x = pad + ci * (cell_w + pad)
        is_best = cfg == best_name
        head = ("★ BEST\n" + label) if is_best else label
        # header background box for best
        if is_best:
            draw.rectangle(
                [x - 2, 0, x + cell_w + 2, header_h - 2],
                fill="#fff3cd",
                outline="#f9ab00",
                width=2,
            )
        draw.multiline_text(
            (x + 2, 3),
            head,
            fill="#202124",
            font=(fontb if is_best else font),
            spacing=1,
        )
        for ri, pid in enumerate(prompt_ids):
            y = header_h + ri * (thumb + pad)
            p = _img_path(store, cfg, pid, seed)
            if p:
                try:
                    im = _load_frame(p)
                    if im is None:
                        raise ValueError("no frame")
                    im.thumbnail((thumb, thumb))
                    canvas.paste(im, (x, y))
                except Exception:
                    draw.rectangle([x, y, x + cell_w, y + thumb], outline="#ccc")
            else:
                draw.rectangle([x, y, x + cell_w, y + thumb], outline="#ccc")
            if is_best:
                draw.rectangle(
                    [x - 1, y - 1, x + cell_w + 1, y + thumb + 1],
                    outline="#f9ab00",
                    width=3,
                )
    canvas.save(out_path)
    return out_path, [c[0] for c in cols], prompt_ids


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--store", required=True, help="QLIP_ARENA_ROOT (has runs/<cfg>/images)"
    )
    ap.add_argument("--report", required=True, help="report.json from qlip_report.py")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--rows", type=int, default=3, help="number of prompts (rows)")
    ap.add_argument("--seed", default="1000")
    ap.add_argument("--thumb", type=int, default=320, help="thumbnail size px")
    ap.add_argument(
        "--configs",
        default=None,
        help="comma list of config names to use as columns (overrides the "
        "auto pick of best+max-quality+max-speed); eager is prepended",
    )
    args = ap.parse_args()

    report = json.load(open(args.report))
    explicit = args.configs.split(",") if args.configs else None
    path, cols, pids = build(
        args.store, report, args.out, args.rows, args.seed, args.thumb, explicit
    )
    print("collage: %s" % path)
    print("columns: %s" % ", ".join(cols))
    print("rows (prompts): %s" % ", ".join(pids))
    print("QLIP_COLLAGE_DONE")


if __name__ == "__main__":
    main()

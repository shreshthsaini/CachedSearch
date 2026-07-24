"""Strategy-level multi-metric evaluation (paper tab:multimetric + fig:multimetric).

Review ask (wave-4, lane M): on ~150 official-suite prompts, compare the DELIVERED
video of four strategies -- single (seed-0 full), best-of-8 full (ImageReward
winner), CachedSearch-keep (cached winner video), CachedSearch-commit (full video
of the cached pick; exists on disk since all full rollouts were saved) -- under
the standard video-gen metric suite, plus measured wall-clock and analytic FLOPs.
Scoring only: every video already exists in results/b1_gate_vbench/videos/.

Prompt population: validated hash groups per paper_figs/e3_analysis.py (CLIP
prompt-index assignment confirmed against the gate's in-memory ImageReward,
median group |IR_e3 - IR_gate| < 0.10), restricted to groups with complete
8 full + 8 cached coverage (seeds 0-7) under BOTH verifiers and all 16 mp4s on
disk; a seeded random sample (rng seed 0) of --n prompts from that population.

Subcommands (in run order):
  select     [login CPU]  build selection.json + a symlink dir of the unique
                          delivered videos for the external scorers.
  lpips      [GPU task]   LPIPS(keep video, same-seed full video) per prompt --
                          distance of the delivered keep draft to its own
                          full-compute reference (0 by construction for the
                          other three strategies, which deliver full videos).
  aggregate  [login CPU]  join gate jsonl (ImageReward, latency, skip stats) +
                          E3 VideoScore + official VBench custom_input results +
                          RAFT flow fallback + LPIPS -> per-strategy means with
                          95% prompt-bootstrap CIs (B=10^4, seed 0; project
                          convention), the LaTeX table body, and
                          paper/figs/f12_multimetric.pdf.

External scoring tasks (enqueued via bin/enqueue.sh after `select`):
  scripts/run_vbench.sh $MM/videos $MM/vbench_official <dims>   (vbench-eval env)
  experiments/vbench_temporal_fallback.py --videos-dir $MM/videos \
      --tag multimetric --metrics dynamic_degree_flow             (videogen1 env)
  python paper_figs/multimetric.py lpips                          (videogen1 env)

FLOPs accounting (stated in the paper): the DiT transformer dominates rollout
compute, so we count transformer forward passes from the recorded per-branch
skip statistics; a full rollout is 50 steps x 2 CFG branches = 100 forwards
(ratio 1.0), a cached rollout computes the recorded fraction. Text encoder and
VAE decode are excluded from the ratio (identical across strategies per video);
wall-clock costs, which include them, are reported alongside.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
GATE_DIR = os.path.join(RESULTS, "b1_gate_vbench")
E3_DIR = os.path.join(RESULTS, "b1_verifiers_vbench")
MM_DIR = os.path.join(RESULTS, "multimetric")
VIDEO_SRC = os.path.join(GATE_DIR, "videos")
VIDEOS_OUT = os.path.join(MM_DIR, "videos")
VBENCH_OUT = os.path.join(MM_DIR, "vbench_official")
FALLBACK_DIR = os.path.join(RESULTS, "vbench_temporal_fallback_multimetric")
LPIPS_LOG = os.path.join(MM_DIR, "lpips.jsonl")
SELECTION = os.path.join(MM_DIR, "selection.json")
NUMBERS_OUT = os.path.join(MM_DIR, "multimetric_numbers.json")
FIG_DIR = os.environ.get("CACHEDSEARCH_FIGURES", "./assets")
FIG_OUT = os.path.join(FIG_DIR, "f12_multimetric.pdf")
FIG_OUT_COMPACT = os.path.join(FIG_DIR, "f12_multimetric_compact.pdf")

GROUP_DIFF_TOL = 0.10          # e3_analysis.py validation tolerance
SEEDS = list(range(8))
STRATS = ("single", "bon_full", "cached_keep", "cached_commit")
VS_DIMS = ["videoscore_visual", "videoscore_temporal", "videoscore_dynamic",
           "videoscore_alignment", "videoscore_factual", "videoscore_avg"]
VBENCH_DIMS = ["subject_consistency", "background_consistency", "motion_smoothness",
               "temporal_flickering", "aesthetic_quality", "imaging_quality"]
B_BOOT = int(os.environ.get("CACHEDSEARCH_BOOT_B", "10000"))


def read_jsonl(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn tail line of a live shard
    return out


def load_gate():
    """(prompt, seed, variant) -> full gate record (score, latency, stats);
    cross-partition duplicates share scores -- last record wins (make_figs.py)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(GATE_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            if r.get("seed") is not None and r["seed"] < 8:
                rows[(r["prompt"], r["seed"], r["variant"])] = r
    return rows


def load_e3():
    """file -> E3 record with both verifiers (last line per file wins)."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(E3_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            recs[r["file"]] = r
    return {f: r for f, r in recs.items()
            if "seed" in r and "imagereward" in r and "videoscore_avg" in r}


def validated_groups(gate, e3):
    """hash -> [records], keeping only hash groups whose CLIP prompt assignment is
    confirmed by the gate ImageReward join (e3_analysis.py section [0])."""
    by_hash = defaultdict(list)
    for r in e3.values():
        by_hash[r["hash"]].append(r)
    kept = {}
    for h, rs in by_hash.items():
        diffs = [abs(r["imagereward"] - gate[(r["prompt"], r["seed"], r["variant"])]["score"])
                 for r in rs if (r["prompt"], r["seed"], r["variant"]) in gate]
        if diffs and float(np.median(diffs)) < GROUP_DIFF_TOL:
            kept[h] = rs
    return kept


def flops_frac(rec):
    """Fraction of transformer forwards computed (all CFG branches pooled)."""
    st = rec.get("stats") or {}
    comp = sum(b.get("computes", 0) for b in st.values())
    skip = sum(b.get("skips", 0) for b in st.values())
    return comp / (comp + skip) if comp + skip else 1.0


# --------------------------------------------------------------------- select

def cmd_select(args):
    gate = load_gate()
    e3 = load_e3()
    groups = validated_groups(gate, e3)
    print(f"gate records {len(gate)}  e3 both-verifier records {len(e3)}  "
          f"validated hash groups {len(groups)}")

    # prompt -> candidate hash groups with complete 8f+8c coverage + files on disk
    complete = defaultdict(list)   # prompt -> [(hash, {(seed, variant): rec})]
    for h, rs in groups.items():
        d = {(r["seed"], r["variant"]): r for r in rs if r["seed"] < 8}
        if not all((s, v) in d for s in SEEDS for v in ("full", "cached")):
            continue
        prompt = rs[0]["prompt"]
        if not all((prompt, s, v) in gate for s in SEEDS for v in ("full", "cached")):
            continue
        if not all(os.path.exists(os.path.join(VIDEO_SRC, d[(s, v)]["file"]))
                   for s in SEEDS for v in ("full", "cached")):
            continue
        complete[prompt].append((h, d))
    print(f"prompts with a complete validated 8f+8c group: {len(complete)}")

    # one group per prompt: highest median index margin (most confident assignment)
    chosen = {}
    for p, cands in complete.items():
        cands.sort(key=lambda hd: -float(np.median([r["index_margin"]
                                                    for r in hd[1].values()])))
        chosen[p] = cands[0]

    prompts = sorted(chosen)
    rng = np.random.default_rng(0)
    if len(prompts) > args.n:
        prompts = sorted(rng.choice(prompts, size=args.n, replace=False))
    print(f"selected prompts: {len(prompts)} (deterministic rng seed 0)")

    rows = []
    for p in prompts:
        h, d = chosen[p]
        fu = np.array([gate[(p, s, "full")]["score"] for s in SEEDS])
        ca = np.array([gate[(p, s, "cached")]["score"] for s in SEEDS])
        pick, bon = int(np.argmax(ca)), int(np.argmax(fu))
        strat = {
            "single": (0, "full"),
            "bon_full": (bon, "full"),
            "cached_keep": (pick, "cached"),
            "cached_commit": (pick, "full"),
        }
        rows.append({
            "prompt": p, "hash": h, "bon_seed": bon, "pick_seed": pick,
            "strategies": {k: {"seed": s, "variant": v, "file": d[(s, v)]["file"]}
                           for k, (s, v) in strat.items()},
            "files": {f"{s}_{v}": d[(s, v)]["file"]
                      for s in SEEDS for v in ("full", "cached")},
        })

    os.makedirs(VIDEOS_OUT, exist_ok=True)
    linked = set()
    for row in rows:
        for spec in row["strategies"].values():
            f = spec["file"]
            if f in linked:
                continue
            dst = os.path.join(VIDEOS_OUT, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(VIDEO_SRC, f), dst)
            linked.add(f)
    with open(SELECTION, "w") as fh:
        json.dump({"n": len(rows), "rng_seed": 0, "tol": GROUP_DIFF_TOL,
                   "rows": rows}, fh, indent=1)
    print(f"unique delivered videos symlinked: {len(linked)} -> {VIDEOS_OUT}")
    print(f"selection written: {SELECTION}")
    # quick sanity: strategy composition
    same_pick = sum(1 for r in rows if r["bon_seed"] == r["pick_seed"])
    print(f"sanity: cached pick == full argmax on {same_pick}/{len(rows)} prompts "
          f"({100 * same_pick / len(rows):.0f}% top-1, expect ~72% official-suite)")


# ---------------------------------------------------------------------- lpips

def cmd_lpips(args):
    sel = json.load(open(SELECTION))["rows"]
    done = {r["prompt"] for r in read_jsonl(LPIPS_LOG)} if os.path.exists(LPIPS_LOG) else set()
    from videogen1.gen import video_lpips, append_jsonl
    from videogen1.video_io import read_video_np
    for row in sel:
        if row["prompt"] in done:
            continue
        keep = os.path.join(VIDEO_SRC, row["strategies"]["cached_keep"]["file"])
        ref = os.path.join(VIDEO_SRC, row["strategies"]["cached_commit"]["file"])
        lp = video_lpips(read_video_np(keep), read_video_np(ref), k=16, device=args.device)
        append_jsonl(LPIPS_LOG, {"prompt": row["prompt"], "hash": row["hash"],
                                 "seed": row["pick_seed"], "lpips_keep_vs_ref": lp})
        print(f"lpips={lp:.4f} :: {row['prompt'][:60]}", flush=True)
    print("lpips done:", LPIPS_LOG)


# ------------------------------------------------------------------ aggregate

def load_vbench_scores():
    """basename -> {dim: score} from custom_<dim>_eval_results.json files."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(VBENCH_OUT, "custom_*_eval_results.json"))):
        blob = json.load(open(path))
        for dim, (mean_score, vids) in blob.items():
            for v in vids:
                out[os.path.basename(v["video_path"])][dim] = float(v["video_results"])
    return out


def load_flow():
    out = {}
    for f in sorted(glob.glob(os.path.join(FALLBACK_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            if "dynamic_degree_flow" in r:
                out[r["file"]] = r["dynamic_degree_flow"]
    return out


def boot_ci(vals, seed=0):
    v = np.asarray(vals, float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(B_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_aggregate(args):
    sel = json.load(open(SELECTION))["rows"]
    gate = load_gate()
    e3 = load_e3()
    vb = load_vbench_scores()
    flow = load_flow()
    lp = {r["prompt"]: r["lpips_keep_vs_ref"] for r in read_jsonl(LPIPS_LOG)} \
        if os.path.exists(LPIPS_LOG) else {}

    # coverage report ------------------------------------------------------
    need = sorted({spec["file"] for row in sel for spec in row["strategies"].values()})
    cov_vb = {d: sum(1 for f in need if d in vb.get(f, {})) for d in VBENCH_DIMS}
    cov_fl = sum(1 for f in need if f in flow)
    cov_vs = sum(1 for f in need if f in e3)
    print(f"videos needed {len(need)} | videoscore {cov_vs} | flow {cov_fl} | "
          f"vbench per dim {cov_vb} | lpips {len(lp)}/{len(sel)}")

    # per (strategy, metric) per-prompt value matrix ------------------------
    metrics = {}   # name -> {strat: np.array over prompts (nan = missing)}

    def put(name, strat, vals):
        metrics.setdefault(name, {})[strat] = np.array(vals, float)

    for strat in STRATS:
        files = [row["strategies"][strat]["file"] for row in sel]
        sv = [(row["strategies"][strat]["seed"], row["strategies"][strat]["variant"])
              for row in sel]
        put("imagereward", strat,
            [gate[(row["prompt"], s, v)]["score"] for row, (s, v) in zip(sel, sv)])
        for dim in VS_DIMS:
            put(dim, strat, [e3[f].get(dim, np.nan) if f in e3 else np.nan for f in files])
        for dim in VBENCH_DIMS:
            put(dim, strat, [vb.get(f, {}).get(dim, np.nan) for f in files])
        put("flow_mag", strat, [flow.get(f, np.nan) for f in files])

        # efficiency: measured wall-clock + analytic transformer-forward FLOPs
        wall, fl = [], []
        for row in sel:
            p = row["prompt"]
            lat = {(s, v): gate[(p, s, v)]["latency"] for s in SEEDS for v in ("full", "cached")}
            fr = {(s, v): flops_frac(gate[(p, s, v)]) for s in SEEDS for v in ("full", "cached")}
            if strat == "single":
                wall.append(lat[(0, "full")]); fl.append(fr[(0, "full")])
            elif strat == "bon_full":
                wall.append(sum(lat[(s, "full")] for s in SEEDS))
                fl.append(sum(fr[(s, "full")] for s in SEEDS))
            elif strat == "cached_keep":
                wall.append(sum(lat[(s, "cached")] for s in SEEDS))
                fl.append(sum(fr[(s, "cached")] for s in SEEDS))
            else:  # cached_commit
                wall.append(sum(lat[(s, "cached")] for s in SEEDS)
                            + lat[(row["pick_seed"], "full")])
                fl.append(sum(fr[(s, "cached")] for s in SEEDS)
                          + fr[(row["pick_seed"], "full")])
        put("wall_clock_s", strat, wall)
        put("flops_full_rollouts", strat, fl)

    put("lpips_keep_vs_ref", "cached_keep", [lp.get(row["prompt"], np.nan) for row in sel])

    # summarize -------------------------------------------------------------
    summary = {}
    for name, per in metrics.items():
        summary[name] = {}
        for strat, vals in per.items():
            ok = vals[np.isfinite(vals)]
            if not len(ok):
                continue
            lo, hi = boot_ci(ok)
            summary[name][strat] = {"mean": float(ok.mean()), "lo": lo, "hi": hi,
                                    "n": int(len(ok))}
    # paired deltas vs best-of-N (same prompts), the "faster AND better" stat
    deltas = {}
    for name, per in metrics.items():
        if "bon_full" not in per:
            continue
        deltas[name] = {}
        for strat in ("cached_keep", "cached_commit", "single"):
            if strat not in per:
                continue
            d = per[strat] - per["bon_full"]
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            lo, hi = boot_ci(d)
            deltas[name][strat] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                                   "n": int(len(d))}

    # paired keep-vs-commit contrast (same pick seed; the dampening probe of
    # sec:abl-temporal, here on official-suite winner picks)
    kc = {}
    for name, per in metrics.items():
        if "cached_keep" not in per or "cached_commit" not in per:
            continue
        d = per["cached_keep"] - per["cached_commit"]
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        lo, hi = boot_ci(d)
        kc[name] = {"mean": float(d.mean()), "lo": lo, "hi": hi, "n": int(len(d))}

    os.makedirs(MM_DIR, exist_ok=True)
    with open(NUMBERS_OUT, "w") as fh:
        json.dump({"n_prompts": len(sel), "summary": summary, "deltas": deltas,
                   "keep_vs_commit": kc}, fh, indent=1)
    print("numbers written:", NUMBERS_OUT)
    print_table(summary, deltas, len(sel))
    print("% ---- paired keep-vs-commit (same pick seed; mean [95% CI]) ----")
    for _label, key, _fmt, _ in ROWS:
        if key in kc:
            d = kc[key]
            print(f"%   {key:24s} keep-commit {d['mean']:+.4f} "
                  f"[{d['lo']:+.4f},{d['hi']:+.4f}] n={d['n']}")
    if args.fig:
        make_figure(metrics, summary, len(sel))


ROWS = [
    ("ImageReward $\\uparrow$", "imagereward", "{:+.3f}", 1),
    ("VideoScore avg $\\uparrow$", "videoscore_avg", "{:.3f}", 1),
    ("VideoScore visual $\\uparrow$", "videoscore_visual", "{:.3f}", 1),
    ("VideoScore dynamic $\\uparrow$", "videoscore_dynamic", "{:.3f}", 1),
    ("VBench subject cons.\\ $\\uparrow$", "subject_consistency", "{:.3f}", 100),
    ("VBench background cons.\\ $\\uparrow$", "background_consistency", "{:.3f}", 100),
    ("VBench motion smooth.\\ $\\uparrow$", "motion_smoothness", "{:.3f}", 100),
    ("VBench temporal flicker $\\uparrow$", "temporal_flickering", "{:.3f}", 100),
    ("VBench aesthetic qual.\\ $\\uparrow$", "aesthetic_quality", "{:.3f}", 100),
    ("VBench imaging qual.\\ $\\uparrow$", "imaging_quality", "{:.3f}", 100),
    ("mean flow magnitude (px)", "flow_mag", "{:.2f}", 1),
    ("LPIPS to full reference $\\downarrow$", "lpips_keep_vs_ref", "{:.3f}", 1),
    ("wall-clock (s) $\\downarrow$", "wall_clock_s", "{:.0f}", 1),
    ("FLOPs (full rollouts) $\\downarrow$", "flops_full_rollouts", "{:.2f}", 1),
]


def print_table(summary, deltas, n):
    print(f"\n% ---- tab:multimetric body (n={n} prompts) ----")
    hdr = " & ".join(["metric", "single", "best-of-8 full",
                      "\\method{}-keep", "\\method{}-commit"])
    print(hdr + r" \\")
    for label, key, fmt, _scale in ROWS:
        cells = []
        for strat in STRATS:
            s = summary.get(key, {}).get(strat)
            if s is None:
                cells.append("--")
            else:
                cells.append(f"${fmt.format(s['mean'])}"
                             f"\\ci{{{fmt.format(s['lo']).lstrip('+')},"
                             f"{fmt.format(s['hi']).lstrip('+')}}}$")
        print(f"{label} & " + " & ".join(cells) + r" \\")
    print("% ---- paired deltas vs best-of-8 (mean [95% CI]) ----")
    for label, key, fmt, _ in ROWS:
        for strat in ("cached_keep", "cached_commit"):
            d = deltas.get(key, {}).get(strat)
            if d:
                print(f"%   {key:24s} {strat:14s} {d['mean']:+.4f} "
                      f"[{d['lo']:+.4f},{d['hi']:+.4f}] n={d['n']}")


# --------------------------------------------------------------------- figure

def make_figure(metrics, summary, n):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # canonical rcParams + palette: paper_figs/style.py (design-overhaul Lane 0);
    # 8pt everywhere -- this figure historically drifted to 7pt (fixed).
    from style import C_STRAT, INK, apply_style
    import matplotlib.pyplot as plt
    import style; style.SHOW_UNCERTAINTY = True  # figs 9/10 are uncrowded: show CI bars

    apply_style()
    LBL = {"single": "single (no search)", "bon_full": "best-of-8 full",
           "cached_keep": "CachedSearch-keep", "cached_commit": "CachedSearch-commit"}

    panels = [
        ("ImageReward", "imagereward", ""),
        ("VideoScore avg", "videoscore_avg", ""),
        ("subject consist.", "subject_consistency", ""),
        ("background cons.", "background_consistency", ""),
        ("motion smooth.", "motion_smoothness", ""),
        ("temporal flicker", "temporal_flickering", ""),
        ("aesthetic qual.", "aesthetic_quality", ""),
        ("imaging (MUSIQ)", "imaging_quality", ""),
        ("flow magn. (px)", "flow_mag", ""),
    ]
    panels = [(t, k, pre) for (t, k, pre) in panels
              if summary.get(k, {}).get("bon_full") or summary.get(k, {}).get("single")]
    ncol = 6
    fig = plt.figure(figsize=(6.6, 2.55))
    gs = fig.add_gridspec(2, ncol, hspace=0.78, wspace=0.62,
                          left=0.05, right=0.99, top=0.86, bottom=0.10)
    xs = np.arange(len(STRATS))

    for i, (title, key, prefix) in enumerate(panels):
        ax = fig.add_subplot(gs[i // (ncol - 1), i % (ncol - 1)])
        for j, strat in enumerate(STRATS):
            s = summary.get(key, {}).get(strat)
            if s is None:
                continue
            ax.errorbar([j], [s["mean"]],
                        yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                        fmt="o", ms=3.4, color=C_STRAT[strat], mfc=C_STRAT[strat],
                        elinewidth=0.7, capsize=1.5, capthick=0.6, zorder=3)
        b = summary.get(key, {}).get("bon_full")
        if b:
            ax.axhline(b["mean"], color=C_STRAT["bon_full"], lw=0.6, ls=":",
                       alpha=0.65, zorder=1)
        ax.set_title(prefix + title, fontsize=6.2, pad=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(STRATS) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)

    # cost panels (last column, one per row): measured wall-clock + analytic FLOPs
    for r, (key, lab, fmt) in enumerate([
            ("wall_clock_s", "wall-clock (s)", "{:.0f}"),
            ("flops_full_rollouts", "FLOPs (full rollouts)", "{:.2f}")]):
        ax = fig.add_subplot(gs[r, ncol - 1])
        for j, strat in enumerate(STRATS):
            s = summary[key][strat]
            ax.barh(len(STRATS) - 1 - j, s["mean"], height=0.62,
                    color=C_STRAT[strat], edgecolor="none", zorder=3)
        bon = summary[key]["bon_full"]["mean"]
        for j, strat in enumerate(STRATS):
            s = summary[key][strat]
            ax.text(s["mean"] + 0.02 * bon, len(STRATS) - 1 - j,
                    f"{fmt.format(s['mean'])}" +
                    (f" ({100 * s['mean'] / bon:.0f}%)" if strat != "bon_full" else ""),
                    va="center", ha="left", fontsize=5.4, color=INK)
        ax.set_title(lab, fontsize=6.5, pad=2)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, bon * 1.75)
        ax.grid(visible=False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=C_STRAT[s],
                          label=LBL[s]) for s in STRATS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035),
               columnspacing=1.4, handletextpad=0.4)
    fig.savefig(FIG_OUT)
    fig.savefig(FIG_OUT.replace(".pdf", ".png"))
    print("figure written:", FIG_OUT)
    make_figure_compact(summary, n)


def make_figure_compact(summary, n):
    """Main-flow 3-panel version (design-overhaul Lane B): the two headline
    quality metrics (ImageReward = the searched verifier; VideoScore avg = an
    independent reference-free suite) plus LPIPS-to-full-reference (the fidelity
    price keep pays and commit does not). The full 11-panel breakdown is the
    appendix figure (make_figure above). Each panel >=1.8in on the page."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from style import C_STRAT, INK, MUTED, apply_style
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import style; style.SHOW_UNCERTAINTY = True  # figs 9/10 are uncrowded: show CI bars

    apply_style()
    LBL = {"single": "single (no search)", "bon_full": "best-of-8 full",
           "cached_keep": "CachedSearch-keep", "cached_commit": "CachedSearch-commit"}
    panels = [("ImageReward $\\uparrow$", "imagereward", None),
              ("VideoScore avg $\\uparrow$", "videoscore_avg", None),
              ("LPIPS to full ref. $\\downarrow$", "lpips_keep_vs_ref", (-0.008, 0.132))]
    # 3 panels across the 5.5in text column; sized to \linewidth so no
    # downscale (each panel ~1.6in wide on the page -- the column width caps
    # three-across below the 1.8in target, but 8pt text is now legible).
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    for ax, (title, key, ylim) in zip(axes, panels):
        for j, strat in enumerate(STRATS):
            s = summary.get(key, {}).get(strat)
            if s is None:
                if key == "lpips_keep_vs_ref":  # 0 by construction (full-delivery strategies)
                    ax.scatter([j], [0.0], marker="o", s=13, color=C_STRAT[strat],
                               zorder=3)
                continue
            ax.errorbar([j], [s["mean"]],
                        yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                        fmt="o", ms=4.2, color=C_STRAT[strat], mfc=C_STRAT[strat],
                        elinewidth=0.7, capsize=1.6, capthick=0.6, zorder=3)
        b = summary.get(key, {}).get("bon_full")
        if b is not None and key != "lpips_keep_vs_ref":
            ax.axhline(b["mean"], color=C_STRAT["bon_full"], lw=0.7, ls=":",
                       alpha=0.7, zorder=1)
        if key == "lpips_keep_vs_ref":
            # mid-left, clear of the keep marker + CI that sit near the top
            ax.annotate("0 by construction\n(full-delivery\nstrategies)", (0.04, 0.52),
                        xycoords="axes fraction", fontsize=5.6, color=MUTED,
                        ha="left", va="center")
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks(range(len(STRATS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.5, len(STRATS) - 0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, color=C_STRAT[s], label=LBL[s])
               for s in STRATS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.5, handletextpad=0.4, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.85, bottom=0.20, wspace=0.30)
    fig.savefig(FIG_OUT_COMPACT)
    fig.savefig(FIG_OUT_COMPACT.replace(".pdf", ".png"))
    print("compact figure written:", FIG_OUT_COMPACT, f"(n={n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--n", type=int, default=150)
    l = sub.add_parser("lpips"); l.add_argument("--device", default="cuda")
    a = sub.add_parser("aggregate"); a.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    {"select": cmd_select, "lpips": cmd_lpips, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Per-architecture tau calibration (wave-4 final integration): tab:taucal.

For every measured (model, tau) operating point, recompute from raw jsonl --
under EXACTLY the make_figs2.py / scaling_analysis.py conventions (dedupe by
(prompt, seed, variant); scores AND latencies seeds 0-7 only; tau dirs are
cached-only and join full references from the model's own v0 dir; capture =
mean per-prompt gain-capture ratio, eq:capture / tab:tau estimator; skip =
pooled skips / (skips+computes) over the cached records' internal counters) --
then select the recommended operating point tau* per model by a single stated
rule:

    tau* = the most aggressive measured tau (highest candidate speedup) whose
           gain capture is >= 85%, i.e. the cheapest setting still inside the
           healthy operating band; if no measured tau qualifies, the most
           conservative measured tau is reported with a boundary flag (the
           model would need a smaller tau than we measured -- or has an
           intrinsically lower frontier).

The 85% band is the paper's own working notion of "healthy": every point the
paper recommends (Wan tau sweep, CogVideoX tau=0.05) sits inside it and every
point it flags as degraded (CogVideoX tau=0.10/0.20, LTX tau=0.10, steps100)
falls outside. Capture is the decision-relevant metric of the paper; median
rho is reported alongside at tau*.

Prints the per-model curve mini-summaries (report edition, tab:taucal caption
source) and the tab:taucal rows; every quoted number greps as [TAUCAL].
Safe on partial data: live dirs (wan14b_tau05fix / tau20fix) are loaded with
a torn-tail-tolerant reader and reported at whatever complete-prompt coverage
exists, with an explicit [LIVE n=..] flag.

Usage:  MALLOC_ARENA_MAX=2 \
        python code/paper_figs/taucal_analysis.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import RESULTS, per_prompt                       # noqa: E402
from make_figs2 import capture_mor, grid, lat_groups, latencies  # noqa: E402
from scaling_analysis import skip_fraction                       # noqa: E402

# model -> list of (tau, tag, full_source_tag or None if the dir has full rows)
# Wan-14B: tau=0.005/0.02 are the historical mis-enqueued arms (records carry
# these true values; 0.005 fires zero skips = bit-identical control); the
# tau=0.05/0.20 arms are the 2026-07-08 "fix" reruns at the intended values.
CURVES = {
    "Wan2.1-1.3B": [(0.05, "tau005", "v0"), (0.10, "v0", None),
                    (0.20, "tau020", "v0")],
    # Wan2.2: the FIRST tau arms (b1_gate_wan22_tau005/020) were enqueued
    # without --height 704 --width 1280 --frames 121 and generated at the
    # 1.3B default recipe -- their scores cannot be ranked against the native
    # 704p/121f full references (med rho ~0.1, nonsense 6.8x/9.6x speedups).
    # INVALID: excluded. The *fix dirs are the corrected reruns (2026-07-08).
    "Wan2.2-TI2V-5B": [(0.05, "wan22_tau005fix", "wan22_5b"),
                       (0.10, "wan22_5b", None),
                       (0.20, "wan22_tau020fix", "wan22_5b")],
    "Wan2.1-14B": [(0.005, "wan14b_tau005", "wan14b"),
                   (0.02, "wan14b_tau020", "wan14b"),
                   (0.05, "wan14b_tau05fix", "wan14b"),
                   (0.10, "wan14b", None),
                   (0.20, "wan14b_tau20fix", "wan14b")],
    "CogVideoX-5B": [(0.05, "cog5b_tau005", "cog5b_v0"),
                     (0.10, "cog5b_v0", None),
                     (0.20, "cog5b_tau020", "cog5b_v0")],
    "HunyuanVideo-13B": [(0.05, "hunyuan_hun_tau005", "hunyuan_v0"),
                         (0.10, "hunyuan_v0", None),
                         (0.20, "hunyuan_hun_tau020", "hunyuan_v0")],
    "LTX-Video-2B": [(0.02, "ltx_ltx_tau002", "ltx_v0"),
                     (0.05, "ltx_ltx_tau005", "ltx_v0"),
                     (0.10, "ltx_v0", None)],
}

# selection rule (stated in the tab:taucal caption): capture-only band. The
# earlier draft also required median rho >= 0.85, but capture is the paper's
# decision-relevant metric (the rho column is still reported at tau*), and no
# measured point with capture >= 85% has median rho below 0.81.
CAPTURE_MIN = 0.85

# cross-checks: every point already quoted in the paper must reproduce
# (tab:tau, tab:scale, f10 caption, sec:abl-cog prose). New arms have no
# expectation yet.
EXPECTED = {
    ("Wan2.1-1.3B", 0.05): (0.936, 1.58), ("Wan2.1-1.3B", 0.10): (0.901, 1.97),
    ("Wan2.1-1.3B", 0.20): (0.883, 2.41),
    ("Wan2.2-TI2V-5B", 0.10): (0.860, 2.05),
    ("Wan2.1-14B", 0.005): (1.000, 1.01), ("Wan2.1-14B", 0.02): (0.921, 1.21),
    ("Wan2.1-14B", 0.10): (0.875, 2.05),
    ("CogVideoX-5B", 0.05): (0.859, 1.78), ("CogVideoX-5B", 0.10): (0.752, 2.06),
    ("CogVideoX-5B", 0.20): (0.647, 2.65),
    ("HunyuanVideo-13B", 0.10): (0.799, 2.19),
    ("LTX-Video-2B", 0.10): (0.676, 2.63),
}


def load_rows_tolerant(tag):
    """make_figs2.load_rows, but skipping torn tail lines of live shards and
    verifying the tau field is uniform (guards the 14B mis-enqueue class)."""
    rows, taus = {}, set()
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}",
                                           "scores_shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["prompt"], r["seed"], r["variant"])] = r
            # tau uniformity checked on seeds 0-7 ONLY: the seed 8-15
            # width-extension shards of tau005/tau020 were mis-enqueued at
            # tau=0.005/0.002 (same class as the 14B arms) but are excluded
            # from every paper number by the seeds<8 filter.
            if r["variant"] == "cached" and r["seed"] < 8:
                taus.add(r.get("tau"))
    if len(taus) > 1:
        raise RuntimeError(f"b1_gate_{tag}: mixed tau values on seeds 0-7: {taus}")
    return rows, (taus.pop() if taus else None)


def point_stats(tag, full_src, rows_cache):
    if tag not in rows_cache:
        rows_cache[tag], _ = load_rows_tolerant(tag)
    rows = rows_cache[tag]
    if not rows:
        return None
    if full_src is None:
        pa = grid(rows, 8)
        Cf = latencies(rows).get("full", float("nan"))
        fg = lat_groups(rows, "full")
    else:
        if full_src not in rows_cache:
            rows_cache[full_src], _ = load_rows_tolerant(full_src)
        pa = grid(rows, 8, full_rows=rows_cache[full_src])
        Cf = latencies(rows_cache[full_src]).get("full", float("nan"))
        fg = lat_groups(rows_cache[full_src], "full")
    if not pa:
        return None
    lat = latencies(rows)
    Cc = lat.get("cached", float("nan"))
    pp = per_prompt(pa)
    return dict(n=len(pa),
                med_rho=float(np.median(pp["rho"])),
                p10_rho=float(np.percentile(pp["rho"], 10)),
                top1=float(np.mean(pp["top1"])),
                capture=capture_mor(pa),
                skip=skip_fraction(rows),
                Cf=Cf, Cc=Cc, speedup=Cf / Cc,
                pa=pa, fg=fg, cg=lat_groups(rows, "cached"))


def pick_tau_star(points):
    """points: {tau: stats}. tau* = most aggressive measured tau with
    capture >= CAPTURE_MIN; if none qualifies, the most conservative
    measured tau, flagged as a boundary."""
    ok = [t for t, s in points.items() if s and s["capture"] >= CAPTURE_MIN]
    if ok:
        return max(ok, key=lambda t: points[t]["speedup"]), True
    # boundary: nothing qualifies -> most conservative measured tau
    measured = [t for t, s in points.items() if s]
    return min(measured), False


def main():
    rows_cache = {}
    print("[TAUCAL] rule: tau* = most aggressive measured tau with "
          f"capture >= {CAPTURE_MIN:.0%}")
    print(f"{'model':<18} {'tau':>6} {'n':>3} {'med rho':>8} {'p10':>6} "
          f"{'top1':>5} {'capt':>6} {'skip':>5} {'Cf':>6} {'Cc':>6} {'speed':>6}")
    table = {}
    for model, curve in CURVES.items():
        pts = {}
        for tau, tag, full_src in curve:
            s = point_stats(tag, full_src, rows_cache)
            pts[tau] = s
            if s is None:
                print(f"{model:<18} {tau:>6} -- no complete prompts yet ({tag})")
                continue
            live = " [LIVE n=%d]" % s["n"] if s["n"] < 44 else ""
            print(f"{model:<18} {tau:>6} {s['n']:>3} {s['med_rho']:>8.3f} "
                  f"{s['p10_rho']:>6.2f} {s['top1']:>4.0%} {s['capture']:>6.1%} "
                  f"{s['skip']:>5.0%} {s['Cf']:>6.1f} {s['Cc']:>6.1f} "
                  f"{s['speedup']:>5.2f}x{live}")
            if (model, tau) in EXPECTED:
                ce, se = EXPECTED[(model, tau)]
                if abs(s["capture"] - ce) > 0.011 or abs(s["speedup"] - se) > 0.03:
                    print(f"  <<< WARNING {model} tau={tau}: capture/speedup "
                          f"{s['capture']:.3f}/{s['speedup']:.2f} vs paper {ce}/{se}")
        star, healthy = pick_tau_star(pts)
        table[model] = (star, healthy, pts)
        s = pts[star]
        flag = "" if healthy else "  <<< BOUNDARY: no measured tau qualifies"
        print(f"[TAUCAL] {model}: tau* = {star}  ->  med rho {s['med_rho']:.3f}, "
              f"capture {s['capture']:.1%}, skip {s['skip']:.0%}, "
              f"speedup {s['speedup']:.2f}x{flag}")
        print()

    # tab:taucal row block (latex-ish, for transcription)
    print("=" * 78)
    print("[TAUCAL] tab:taucal rows (model / tau* / med rho / capture / skip / speedup)")
    for model, (star, healthy, pts) in table.items():
        s = pts[star]
        mark = "" if healthy else r" (boundary)"
        print(f"  {model:<18} tau*={star:<5}{mark} rho={s['med_rho']:.3f} "
              f"capt={s['capture']:.1%} skip={s['skip']:.0%} sp={s['speedup']:.2f}x n={s['n']}")

    # per-model own log-linear frontier capture = a + b ln(speedup) over the
    # measured non-degenerate arms (skip > 0), and where it crosses the band --
    # the boundary diagnosis for LTX (where does ITS curve reach 85%?)
    print("\n[TAUCAL] per-model own frontiers (fit on measured arms, skip>0):")
    for model, (star, healthy, pts) in table.items():
        arms = [(s["speedup"], s["capture"]) for s in pts.values()
                if s and s["skip"] > 0.005 and s["n"] >= 44]  # no LIVE partials
        if len(arms) < 3:
            continue
        sp = np.array([a[0] for a in arms])
        cap = 100 * np.array([a[1] for a in arms])
        A = np.vstack([np.ones_like(sp), np.log(sp)]).T
        (a, b), res = np.linalg.lstsq(A, cap, rcond=None)[:2]
        ss = ((cap - cap.mean()) ** 2).sum()
        r2 = 1 - (cap - A @ np.array([a, b])).var() * len(cap) / ss if ss else float("nan")
        cross = float(np.exp((100 * CAPTURE_MIN - a) / b)) if b < 0 else float("nan")
        print(f"  {model:<18} capture = {a:6.1f} {b:+5.1f} ln(sp)  R2={r2:.2f}  "
              f"crosses {CAPTURE_MIN:.0%} at {cross:.2f}x")


if __name__ == "__main__":
    main()

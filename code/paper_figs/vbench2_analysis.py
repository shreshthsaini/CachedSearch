"""VBench-2.0 cross-suite replication (wave-4): the standard E2-style analysis
on results/b1_gate_vbench2 (1,013 VBench-2.0 prompts x 8 seeds, full vs cached
at tau=0.10, Wan2.1-1.3B, identical protocol to the official-VBench run).

Conventions IDENTICAL to make_figs.py / bootstrap_ci.py ("vbench" group):
records deduped by (prompt, seed, variant); seeds 0-7; complete-coverage
prompts only; capture = mean per-prompt gain-capture ratio (eq:capture) AND
ratio-of-means, both printed; latencies seeds 0-7.

Also prints the spread-corruption stat corr(spread, rho) (Spearman, prompt
bootstrap CI, per corruption_n944.py) so a capture shift on the harder
compositional suite can be connected to the mechanism, and the three-suite
comparison row block for tab:vbench2. Every quoted number greps as [VB2].

Usage:  MALLOC_ARENA_MAX=2 \
        python code/paper_figs/vbench2_analysis.py
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import per_prompt                                 # noqa: E402
from make_figs2 import capture_mor, grid, latencies              # noqa: E402
from taucal_analysis import load_rows_tolerant                   # noqa: E402

B = 10_000
RNG = np.random.default_rng(0)


def stats_block(rows, name):
    pa = grid(rows, 8)
    lat = latencies(rows)
    pp = per_prompt(pa)
    n = len(pa)
    rho = pp["rho"]
    ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
    ratios = ratios[np.isfinite(ratios)]
    sp = lat["full"] / lat["cached"]
    print(f"\n[VB2] === {name}: n={n} complete prompts "
          f"({sum(len(g[0]) + len(g[1]) for g in pa.values())} scored rollouts) ===")
    print(f"[VB2] median rho {np.median(rho):.3f} | mean {rho.mean():.3f} | "
          f"p10 {np.percentile(rho, 10):.3f} | rho<0.7 share {np.mean(rho < 0.7):.1%}")
    print(f"[VB2] top-1 {pp['top1'].mean():.1%} | zero-regret {(pp['regret'] == 0).mean():.1%}")
    print(f"[VB2] regret {pp['regret'].mean():.4f} | rand {pp['rand'].mean():.3f} | "
          f"capture mor {ratios.mean():.1%} / rom {1 - pp['regret'].mean() / pp['rand'].mean():.1%}")
    print(f"[VB2] latency {lat['full']:.1f} -> {lat['cached']:.1f}s | speedup {sp:.2f}x")
    # spread-corruption mechanism (corruption_n944.py convention)
    r, p_asym = spearmanr(pp["spread"], rho)
    idx = RNG.integers(0, n, (B, n))
    bs = np.array([spearmanr(pp["spread"][i], rho[i])[0] for i in idx[:2000]])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"[VB2] corr(spread, rho) Spearman {r:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"(p={p_asym:.2g}) | mean spread {pp['spread'].mean():.3f}")
    return dict(pa=pa, pp=pp, n=n, sp=sp, lat=lat)


def main():
    rows_vb2, _ = load_rows_tolerant("vbench2")
    vb2 = stats_block(rows_vb2, "VBench-2.0 (1,013 prompts)")
    # reference suites for the tab:vbench2 comparison block
    rows_vb, _ = load_rows_tolerant("vbench")
    vb = stats_block(rows_vb, "official VBench (946 prompts)")
    rows_v0, _ = load_rows_tolerant("v0")
    stats_block(rows_v0, "gate grid (50 prompts)")

    # suite-vs-suite quick tests (two-sample bootstrap on mean rho / capture)
    def two_sample(a, b, name):
        pa_, pb_ = np.asarray(a, float), np.asarray(b, float)
        d = pa_.mean() - pb_.mean()
        da = pa_[RNG.integers(0, len(pa_), (B, len(pa_)))].mean(1)
        db = pb_[RNG.integers(0, len(pb_), (B, len(pb_)))].mean(1)
        lo, hi = np.percentile(da - db, [2.5, 97.5])
        print(f"[VB2] vb2 - vbench {name}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    two_sample(vb2["pp"]["rho"], vb["pp"]["rho"], "mean rho")
    ra = (vb2["pp"]["rand"] - vb2["pp"]["regret"]) / vb2["pp"]["rand"]
    rb = (vb["pp"]["rand"] - vb["pp"]["regret"]) / vb["pp"]["rand"]
    two_sample(ra[np.isfinite(ra)], rb[np.isfinite(rb)], "capture (mor)")
    two_sample(vb2["pp"]["spread"], vb["pp"]["spread"], "mean spread")


if __name__ == "__main__":
    main()

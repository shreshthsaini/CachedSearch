"""Qualitative keep-vs-commit figure (paper/figs/fig_qualitative.pdf).

Shows the E1 winner pairs (results/b1_temporal/videos/<hash>_<tag>_{keep,commit}.mp4)
for the 3 prompts with the LARGEST keep-commit motion difference, motion measured
by the mean |frame_t - frame_{t-1}| proxy (computed frame-incrementally via
videogen1.video_io.read_video_np; cached at
results/b1_temporal/motion_proxy_framediff.json -- delete to recompute, ~6 min).
All top-3 pairs land at tau=0.20, matching the -8.0% mean-flow dampening in the
ablations. Per-row annotations: RAFT mean optical-flow magnitude (measured, from
results/b1_temporal/vbench_temporal_fallback.jsonl/scores_shard0.jsonl) and the
frame-diff proxy. hash->prompt mapping: the CLIP-retrieval manifest verified by
the eval-stack agent (results/b1_verifiers_vbench/manifest.jsonl, 50-group
bijection; filenames use per-process-randomized hash(prompt)).

Run: MALLOC_ARENA_MAX=2 python make_fig_qualitative.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from videogen1.video_io import read_video_np

# canonical rcParams + palette: paper_figs/style.py (design-overhaul Lane 0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import INK, MUTED, STRATEGY_COLORS, apply_style, pill

apply_style()

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
VID = os.path.join(RESULTS, "b1_temporal", "videos")
PROXY = os.path.join(RESULTS, "b1_temporal", "motion_proxy_framediff.json")
FLOW = os.path.join(RESULTS, "b1_temporal", "vbench_temporal_fallback.jsonl", "scores_shard0.jsonl")
MANIFEST = os.path.join(RESULTS, "b1_verifiers_vbench", "manifest.jsonl")
OUT = os.path.join(
    os.environ.get("CACHEDSEARCH_FIGURES", "./assets"),
    "fig_qualitative.pdf",
)
N_FRAMES = 4

# entity-fixed strategy colors (style.STRATEGY_COLORS): keep=green, commit=blue
C_KEEP = STRATEGY_COLORS["cached_keep"]
C_COMMIT = STRATEGY_COLORS["cached_commit"]


def frame_diff_proxy(path):
    v = read_video_np(path)  # uint8 [N,H,W,C]; incremental (login vmem limit)
    acc = 0.0
    for t in range(1, len(v)):
        acc += float(np.abs(v[t].astype(np.int16) - v[t - 1].astype(np.int16)).mean())
    return acc / (len(v) - 1)


def load_proxy():
    done = json.load(open(PROXY)) if os.path.exists(PROXY) else {}
    files = sorted(glob.glob(os.path.join(VID, "*.mp4")))
    todo = [f for f in files if os.path.basename(f) not in done]
    for i, f in enumerate(todo):
        done[os.path.basename(f)] = frame_diff_proxy(f)
        if i % 20 == 0:
            json.dump(done, open(PROXY, "w"))
            print(f"proxy {i}/{len(todo)}", flush=True)
    if todo:
        json.dump(done, open(PROXY, "w"))
    return done


def frame_axis(ax, img, color, lw=1.1):
    ax.imshow(img, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)  # apply_style() hides top/right
        s.set_edgecolor(color); s.set_linewidth(lw)


def chip(ax, text, color, xy, ha="left", va="bottom", fs=5.7):
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=fs, color="white",
                ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="round,pad=0.24", fc=color, ec="none", alpha=0.95))


def main():
    proxy = load_proxy()
    manifest = {r["hash"]: r["prompt"] for r in map(json.loads, open(MANIFEST))}
    flow = {(r["hash"], r["tau_tag"], r["arm"]): r["dynamic_degree_flow"]
            for r in map(json.loads, open(FLOW))}

    # rank (hash, tag) pairs by |proxy_keep - proxy_commit|; keep best tag per prompt
    pair = defaultdict(dict)
    for name, v in proxy.items():
        h, tag, arm = name.replace(".mp4", "").split("_")
        pair[(h, tag)][arm] = v
    ranked = sorted(pair.items(), key=lambda kv: -abs(kv[1]["keep"] - kv[1]["commit"]))
    picks, seen = [], set()
    for (h, tag), d in ranked:
        if h not in seen:
            picks.append((h, tag, d))
            seen.add(h)
        if len(picks) == 3:
            break
    print("selected pairs (largest |keep-commit| frame-diff):")
    for h, tag, d in picks:
        print(f"  {h} {tag}  proxy keep={d['keep']:.2f} commit={d['commit']:.2f}"
              f"  :: {manifest[h]}")
    tau_lbl = {"tau005": "0.05", "v0": "0.10", "tau020": "0.20"}

    # load frames + numbers first, then place manually so prompt groups get real
    # vertical padding (fixes the caption/strip overlap of the old uniform grid).
    groups = []
    for h, tag, d in picks:
        rows, idxs = {}, {}
        for arm in ("keep", "commit"):
            v = read_video_np(os.path.join(VID, f"{h}_{tag}_{arm}.mp4"))
            idx = np.linspace(0, len(v) - 1, N_FRAMES).astype(int)
            rows[arm] = [v[t].copy() for t in idx]
            idxs[arm] = idx
            del v
        groups.append((h, tag, d, rows, idxs))
    aspect = groups[0][3]["keep"][0].shape[0] / groups[0][3]["keep"][0].shape[1]

    W = 6.6
    lm, rm, wsp = 0.52, 0.02, 0.014     # left margin holds the keep/commit pill
    frame_w = (W - lm - rm - (N_FRAMES - 1) * wsp) / N_FRAMES
    frame_h = frame_w * aspect
    cap_h, row_gap, grp_gap, head_h, bot = 0.17, 0.022, 0.30, 0.17, 0.04
    grp_h = cap_h + 2 * frame_h + row_gap
    H = head_h + 3 * grp_h + 2 * grp_gap + bot
    fig = plt.figure(figsize=(W, H))
    arms = [("keep", C_KEEP), ("commit", C_COMMIT)]
    for p, (h, tag, d, rows, idxs) in enumerate(groups):
        y_grp_top = H - head_h - p * (grp_h + grp_gap)
        fig.text(lm / W, (y_grp_top + 0.01) / H,
                 f"``{manifest[h]}''  ($\\tau_c={tau_lbl[tag]}$)",
                 fontsize=6.8, color=INK, ha="left", va="bottom", fontstyle="italic")
        for a, (arm, col) in enumerate(arms):
            row_top = y_grp_top - cap_h - a * (frame_h + row_gap)
            fl = flow[(h, tag, arm)]
            for c, t in enumerate(idxs[arm]):
                x = lm + c * (frame_w + wsp)
                ax = fig.add_axes([x / W, (row_top - frame_h) / H,
                                   frame_w / W, frame_h / H])
                frame_axis(ax, rows[arm][c], col)
                if p == 0 and a == 0:  # time headers on the very top row
                    ax.set_title(f"$t={t}$", fontsize=6.6, color=MUTED, pad=2)
                if c == 0:  # keep/commit pill + measured motion chip
                    pill(ax, arm, col, xy=(-0.03, 0.5), ha="right", va="center",
                         fontsize=6.4)
                    chip(ax, rf"flow {fl:.1f}  $|\Delta|$ {d[arm]:.0f}", col,
                         xy=(0.03, 0.09))
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"))  # QA copy
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Figure 2 (paper/figs/fig2_method.pdf): the CachedSearch method schematic --
explore cheap, commit full -- as a clean 3-stage flow with REAL Wan2.1-1.3B
frames (design-overhaul Lane A, BUILD 2).

  (1) explore N candidates under aggressive caching  -> faded draft thumbnails
  (2) verifier scores -> pick the winner              -> small bars, winner star
  (3) commit: re-generate ONLY the winner at full compute -> one sharp frame,
      "bit-exact full-compute sample"
Cost under each stage: N.C_c  /  +eps (verifier)  /  +C_f.

Real data only:
  - Frames: b1_gate_vbench video files (Wan2.1-1.3B, tau=0.10, 8 seeds/prompt),
    read via videogen1.video_io. The generating prompt was recovered by
    ImageReward argmax over the VBench prompt list (identify_fig2.py; filename
    hashes are irreversible, PYTHONHASHSEED unset in generation).
  - Verifier scores + winner: the RECORDED per-seed ImageReward cached scores
    for that prompt (b1_gate_vbench scores jsonl). The commit frame is the
    winner seed's full-compute rollout.
The identification cache (fig2_ident.json) is produced once by identify_fig2.py;
this script only reads it. Style: paper_figs/style.py (Lane 0).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import (apply_style, BASE, C_BLUE, C_YELLOW, GRID, INK, MUTED)

apply_style()

import sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_DIR)
from videogen1 import video_io

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
OUT = os.environ.get("CACHEDSEARCH_FIGURES", "./assets")
# Frozen example (committed): the identified prompt + recorded per-seed cached
# scores, chosen once by identify_fig2.py (confident ID, agreeing winner, clear
# margin). Video frames are read from the repo copy, falling back to scratch.
EXAMPLE = os.path.join(os.path.dirname(__file__), "fig2_example.json")
VID_REPO = os.path.join(RESULTS, "b1_gate_vbench", "videos")
VID_SCRATCH = VID_REPO
NDISP = 3  # illustrative candidates shown in the explore stage

FW, FH = 6.6, 2.05  # figure size (inches; kept short so the conference edition
#                     holds it in the method section within the p.10 budget)


def pick_example():
    c = json.load(open(EXAMPLE))
    cached = np.array(c["cached"])
    win = int(np.argmax(cached))
    # display seeds: winner + (NDISP-1) others spanning the score range.
    losers = [s for s in range(8) if s != win]
    losers.sort(key=lambda s: cached[s])
    idx = np.linspace(0, len(losers) - 1, NDISP - 1).astype(int)
    show = [losers[i] for i in idx] + [win]
    show.sort(key=lambda s: -cached[s])  # descending: winner first (top of stack)
    return c, win, show, cached


def read_mid(pid, seed, variant):
    name = f"{pid}_{seed}_{variant}.mp4"
    path = os.path.join(VID_REPO, name)
    if not os.path.exists(path):
        path = os.path.join(VID_SCRATCH, name)
    v = video_io.read_video_np(path)
    return video_io.to_uint8(v[len(v) // 2])


def read_frames(pid, seed, variant, fracs):
    """Frames at the given trajectory fractions (for the stage-3 video stack)."""
    name = f"{pid}_{seed}_{variant}.mp4"
    path = os.path.join(VID_REPO, name)
    if not os.path.exists(path):
        path = os.path.join(VID_SCRATCH, name)
    v = video_io.read_video_np(path)
    return [video_io.to_uint8(v[int(f * (len(v) - 1))]) for f in fracs]


def thumb(fig, img, cx, cy, w_in, ec, lw, alpha=1.0, z=3):
    """Place a framed thumbnail centered at (cx, cy) in inches; height from aspect."""
    h_in = w_in * img.shape[0] / img.shape[1]
    ax = fig.add_axes([(cx - w_in / 2) / FW, (cy - h_in / 2) / FH, w_in / FW, h_in / FH],
                      zorder=z)
    ax.imshow(img, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(ec); s.set_linewidth(lw)
    ax.grid(False)
    return ax, h_in


def arrow(fig, x0, x1, y, color=INK, lw=1.4):
    ax = fig.add_axes([0, 0, 1, 1], zorder=8); ax.axis("off"); ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))
    return ax


def main():
    c, win, show, cached = pick_example()
    pid = c["pid"]
    fig = plt.figure(figsize=(FW, FH))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, FW); bg.set_ylim(0, FH)

    # stage header + cost labels
    def header(x, n, cost, color=INK):
        bg.text(x, FH - 0.14, n, ha="center", va="top", fontsize=8.5, color=color,
                fontweight="bold")
    def cost(x, t, color=MUTED):
        bg.text(x, 0.16, t, ha="center", va="bottom", fontsize=8, color=color)

    # ---- Stage 1: explore N under cache (faded draft thumbnails) ----
    x1 = 1.15
    header(x1, r"1  explore $N$ under cache", None)
    ys = np.linspace(1.52, 0.58, NDISP)  # winner first -> top of the stack
    tw = 0.98
    for yi, seed in zip(ys, show):
        img = read_mid(pid, seed, "cached")
        thumb(fig, img, x1, yi, tw, ec=BASE, lw=0.7, alpha=0.55)
        # seed tag: ties each candidate to its score bar in stage 2
        bg.text(x1 - tw / 2 - 0.05, yi, rf"$s_{seed}$", ha="right", va="center",
                fontsize=6.6, color=MUTED, zorder=6)
    cost(x1, r"cost  $N\!\cdot\!C_c$  ($\tau{=}0.10$)")

    # ---- Stage 2: verifier scores -> pick winner (bars) ----
    x2c = 3.30
    header(x2c, "2  score & pick winner", None)
    axb = fig.add_axes([(x2c - 0.70) / FW, 0.44 / FH, 1.42 / FW, 1.14 / FH], zorder=4)
    disp_scores = cached[show]
    ymin = min(disp_scores.min(), 0) - 0.05
    ypos = np.arange(NDISP)[::-1]  # show[0]=winner -> top row
    colors = [C_BLUE if s == win else GRID for s in show]
    axb.barh(ypos, disp_scores - ymin, left=ymin, color=colors, height=0.6,
             edgecolor="white", linewidth=0.5, zorder=3)
    # per-bar lettering: seed tag at the bar base, measured score at the tip
    for row, seed in zip(ypos, show):
        axb.text(ymin - 0.03, row, rf"$s_{seed}$", ha="right", va="center",
                 fontsize=6.6, color=C_BLUE if seed == win else MUTED, clip_on=False)
        off = 0.14 if seed == win else 0.03  # winner: clear the star at the tip
        axb.text(cached[seed] + off, row, f"{cached[seed]:+.2f}", ha="left",
                 va="center", fontsize=6.0,
                 color=C_BLUE if seed == win else MUTED, clip_on=False)
    # winner star at the bar tip
    wrow = ypos[show.index(win)]
    axb.scatter([cached[win]], [wrow], marker="*", s=95, facecolors="white",
                edgecolors=C_BLUE, linewidths=1.0, zorder=5, clip_on=False)
    axb.text(cached[win], wrow + 0.42, "winner", ha="right", va="bottom", fontsize=6.6,
             color=C_BLUE, fontweight="bold")
    axb.set_xlim(ymin, disp_scores.max() + 0.12)
    axb.set_ylim(-0.6, NDISP - 0.3)
    axb.set_yticks([]); axb.set_xticks([])
    axb.set_xlabel("ImageReward", fontsize=7, labelpad=1, color=MUTED)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)
    axb.spines["bottom"].set_color(BASE)
    axb.grid(False)
    cost(x2c, r"cost  $+\,\epsilon$ (verify)")

    # ---- Stage 3: commit winner at full compute (video = 3D frame stack) ----
    x3 = 5.58
    header(x3, "3  commit at full compute", C_BLUE)
    cw = 1.48
    # two later frames of the SAME winning full-compute video, offset up-right
    # and translucent behind the front frame: reads as a video, not a still.
    front, mid_f, back = read_frames(pid, win, "full", [0.5, 0.7, 0.9])
    dx, dy = 0.11, 0.075
    thumb(fig, back, x3 + 2 * dx, 1.04 + 2 * dy, cw, ec=BASE, lw=0.8, alpha=0.35, z=3)
    thumb(fig, mid_f, x3 + dx, 1.04 + dy, cw, ec=BASE, lw=1.0, alpha=0.60, z=4)
    ax3, h3 = thumb(fig, front, x3, 1.04, cw, ec=C_BLUE, lw=1.6, alpha=1.0, z=5)
    bg.text(x3, 1.04 - h3 / 2 - 0.08, "bit-exact\nfull-compute video", ha="center",
            va="top", fontsize=7.2, color=C_BLUE)
    cost(x3, r"cost  $+\,C_f$")

    # ---- arrows between stages (labeled with the action they carry) ----
    a1x0, a1x1 = x1 + tw / 2 + 0.06, x2c - 0.88
    a2x0, a2x1 = x2c + 0.88, x3 - cw / 2 - 0.14
    arrow(fig, a1x0, a1x1, 1.02, color=INK)
    arrow(fig, a2x0, a2x1, 1.02, color=INK)
    bg.text((a1x0 + a1x1) / 2, 1.10, "verify", ha="center", va="bottom",
            fontsize=6.4, color=INK)
    bg.text((a2x0 + a2x1) / 2, 1.10, "re-generate\nwinner's seed", ha="center",
            va="bottom", fontsize=6.4, color=INK)

    out = os.path.join(OUT, "fig2_method.pdf")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {out}  (prompt={c['prompt'][:50]!r}, pid={pid}, winner=s{win})")


if __name__ == "__main__":
    main()

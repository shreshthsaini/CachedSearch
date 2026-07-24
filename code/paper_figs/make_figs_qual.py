"""Qualitative-comparison figure suite (paper/figs/fig_qual_*.pdf).

Design-overhaul Lane C (2026-07-08): video-paper-grade presentation
(Seedance p13 / Bernini p2 / NVIDIA p9 conventions) -- full-width money shot,
colored pill() strategy labels, per-video reward chips, thin red zoom-boxes on
the discriminating region, italic prompt captions, condition/model left row
labels. NO number changes: every frame is read via
videogen1.video_io.read_video_np, every annotation (reward, LPIPS, seed, skip
count) is read from the measured jsonl records -- nothing hand-entered.

  fig_qual_search.pdf   (MONEY SHOT)  THREE official-suite prompts, three
      delivered videos each: single sample (seed 0, full), best-of-8 over full
      rollouts, and CachedSearch-commit (explore 8 cached rollouts, recommit the
      winner at full compute). Rewards = the gate's ImageReward records
      (results/b1_gate_vbench/scores_shard*.jsonl, seeds 0-7, dedup by
      (prompt,seed,variant)). hash->prompt via the CLIP-retrieval manifest
      (results/b1_verifiers_vbench/manifest.jsonl) restricted to VALIDATED hash
      groups per the e3_analysis.py rule (median |IR_e3 - IR_gate| < 0.10).
      The two prompts cut for the money shot go to fig_qual_search_gallery.pdf
      (the appendix gallery). The script re-derives all numbers and warns loudly
      if a pick stops validating.

  fig_qual_search_gallery.pdf  appendix gallery: the remaining validated picks
      (same 3-column search layout, no zoom-boxes), so no prompt is lost when
      the money shot is trimmed to three.

  fig_qual_methods.pdf   FOUR-method budget grid (2026-07-14, Flash-BoN-Fig-5
      style): three validated official-suite prompts as rows; single |
      best-of-8 full | commit | keep as columns, each cell the actual
      delivered video's 2-frame strip with measured reward chips and the
      strategy's measured wall-clock (as % of best-of-8 full) in the header.
      keep = the cached winner rollout itself (nominal r-hat); commit = the
      full-compute regeneration of the same seed (bit-exact chip when the
      pick agrees with best-of-8).

  fig_qual_flow.pdf      motion dampening made VISIBLE (sec:abl-motion): two
      high-motion gate-50 prompts at the aggressive arm (tau=0.20), same-seed
      full (commit) vs cached (keep) rows, each with a temporal-mean Farneback
      optical-flow magnitude map on a shared color scale; the cached map is
      visibly dimmer (eagle -32%, blacksmith -12% mean |flow|, live-computed;
      chips carry the measured means).

  fig_qual_fidelity.pdf  4 gate-50 prompts, same-seed full vs cached (tau=0.10)
      winner pairs from E1 (results/b1_temporal/videos/<hash>_v0_{commit,keep}.mp4;
      keep = cached rollout, commit = full-compute regeneration of the SAME seed).
      Seedance-Fig8 layout: conditions (Full / Cached tau=0.10) as LEFT ROW
      LABELS, time as columns. LPIPS(keep, commit) from
      results/b1_temporal/lpips_fixed.jsonl; rewards from
      results/b1_temporal/records_shard*.jsonl. Picks span the LPIPS range
      (min 0.021 / mean 0.142 / max 0.343 at tau=0.10, n=50).

  fig_qual_models.pdf    one shared prompt (experiments/prompts_qual1.txt), full
      vs cached (tau=0.10) at seed 0 on all six models; regenerated via the task
      spool into results/b1_gate_{qual13b,ltx_qual,qualwan22,cog5b_qual,
      hunyuan_qual,qual14b}/videos (2 videos + 1 jsonl each). Model names are
      left row labels in MODEL_COLORS (logo-margin gap reserved); reward + skip
      counts are small corner chips. SKIPPED with a message for any model whose
      videos have not landed yet.

Run: MALLOC_ARENA_MAX=2 python make_figs_qual.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from videogen1.video_io import read_video_np

# canonical rcParams + palette: paper_figs/style.py (design-overhaul Lane 0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (INK, MUTED, BASE, C_RED, STRATEGY_COLORS, MODEL_COLORS,
                   apply_style, pill)

apply_style()

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
GATE_DIR = os.path.join(RESULTS, "b1_gate_vbench")
E3_DIR = os.path.join(RESULTS, "b1_verifiers_vbench")
TEMP_DIR = os.path.join(RESULTS, "b1_temporal")
FIGS = os.environ.get("CACHEDSEARCH_FIGURES", "./assets")
GROUP_DIFF_TOL = 0.10  # e3_analysis.py validation rule

# entity-fixed strategy colors (style.STRATEGY_COLORS):
C_SINGLE, C_BON, C_COMMIT, C_KEEP = (
    STRATEGY_COLORS["single"], STRATEGY_COLORS["bon_full"],
    STRATEGY_COLORS["cached_commit"], STRATEGY_COLORS["cached_keep"])

# ----------------------------------------------------------------- selections
# fig_qual_search MONEY SHOT: three validated official-suite hashes chosen for
# visual diversity + the identical/near-equal-pick story (2 identical, 1
# different-seed-equal-reward). 23502052 is the different-seed case (dr=-0.03).
PICKS_SEARCH = ["15225086", "29871135", "23502052"]  # harp, robot, Seine/Eiffel
# remaining validated picks -> appendix gallery (no prompt is lost).
PICKS_SEARCH_GALLERY = ["99838320", "45176935"]       # bigfoot, steam train
# thin red zoom-box on the discriminating region, per money-shot pick. Coords are
# axes fraction (x0,y0,x1,y1); the box targets the FIRST frame of the 3-frame
# strip and is drawn on all three strategy strips so the eye compares the SAME
# region (absent in single -> present after search). Tuned on the render.
ZOOM_SEARCH = {
    "15225086": (0.045, 0.10, 0.315, 0.94),  # harpist + harp
    "29871135": (0.045, 0.06, 0.315, 0.97),  # robot figure
    "23502052": (0.045, 0.30, 0.315, 0.99),  # Eiffel Tower / skyline band
}
# fig_qual_fidelity: gate-50 hashes spanning the tau=0.10 LPIPS range.
PICKS_FID = ["85411900", "32600487", "14098442", "72437844"]
# fig_qual_methods: the four-method budget grid (single | best-of-8 full |
# commit | keep). Validated official-suite hashes with complete 16-video sets,
# distinct in content from PICKS_SEARCH/GALLERY; chosen for prompt diversity
# (object / two-object compositional / spatial relation) with one
# different-pick row (89174265) so the near-tie case is shown, not hidden.
PICKS_METHODS = ["36545761", "89174265", "61969579"]
# fig_qual_flow: motion dampening made visible. Gate-50 hashes at the
# AGGRESSIVE arm (tau=0.20, the -8%-mean-flow regime of sec:abl-motion),
# shortlisted by motion_proxy_framediff.json then ranked by the figure's own
# Farneback estimator over all high-motion candidates (2026-07-14 scan):
# eagle -32% and blacksmith -12% mean |flow|, the two largest measured drops.
PICKS_FLOW = ["1592225", "42625758"]
# fig_qual_models: (label, results subdir, MODEL_COLORS key). Wan family first
# (blue ramp), the over-driven boundary model LTX last (the drift punchline).
MODELS = [
    ("Wan2.1 1.3B", "b1_gate_qual13b", "Wan2.1-1.3B"),
    ("Wan2.2 5B", "b1_gate_qualwan22", "Wan2.2-5B"),
    ("Wan2.1 14B", "b1_gate_qual14b", "Wan2.1-14B"),
    ("CogVideoX 5B", "b1_gate_cog5b_qual", "CogVideoX-5B"),
    ("Hunyuan 13B", "b1_gate_hunyuan_qual", "HunyuanVideo-13B"),
    ("LTX-Video 2B", "b1_gate_ltx_qual", "LTX-Video-2B"),
]
GAP_PX = 6  # white gap inside multi-frame strips


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


def strip_image(video, times, gap=GAP_PX):
    """Horizontally concatenated frames with white gaps."""
    frames = [video[t] for t in times]
    h = frames[0].shape[0]
    white = np.full((h, gap, 3), 255, dtype=np.uint8)
    parts = []
    for i, f in enumerate(frames):
        if i:
            parts.append(white)
        parts.append(f)
    return np.concatenate(parts, axis=1)


def even_idx(v, n, lo=0.10, hi=0.92):
    """n evenly spaced interior frame indices (skip the near-static endpoints)."""
    return list(np.linspace(len(v) * lo, len(v) * (1 - (1 - hi)), n)
                .clip(0, len(v) - 1).astype(int))


def trunc(s, n):
    """Truncate a prompt to n chars (word-safe-ish) so a caption fits its panel."""
    return s if len(s) <= n else s[:n - 1].rstrip(" ,;") + "…"


def frame_axis(ax, img, color, lw=1.2, aspect_mode=None):
    ax.imshow(img) if aspect_mode is None else ax.imshow(img, aspect=aspect_mode)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)  # apply_style() hides top/right; frames need all 4
        s.set_edgecolor(color)
        s.set_linewidth(lw)


def chip(ax, text, color, xy, ha="left", va="bottom", fs=6.0, fc=None,
         text_color="white", ec="none"):
    """Small rounded corner chip on a frame (reward / LPIPS / skips). Solid fill
    in the entity color by default; white-on-color reads at any frame size."""
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=fs,
                color=text_color, ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="round,pad=0.24", fc=fc or color, ec=ec,
                          lw=0.5, alpha=0.95))


def redbox(ax, box, lw=0.9):
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes,
                           fill=False, ec=C_RED, lw=lw, zorder=7,
                           joinstyle="miter"))


# ------------------------------------------------------------- shared loaders
def load_gate_scores():
    rows = {}
    for f in sorted(glob.glob(os.path.join(GATE_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            rows[(r["prompt"], r["seed"], r["variant"])] = r["score"]
    return rows


def validated_hashes(gate):
    """hash -> prompt for E2 hash groups that pass the e3_analysis.py check."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(E3_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            recs[r["file"]] = r
    by_hash = defaultdict(list)
    for r in recs.values():
        if "seed" in r and "imagereward" in r:
            by_hash[r["hash"]].append(r)
    out = {}
    for h, rs in by_hash.items():
        diffs = [abs(r["imagereward"] - gate[(r["prompt"], r["seed"], r["variant"])])
                 for r in rs if (r["prompt"], r["seed"], r["variant"]) in gate]
        if diffs and float(np.median(diffs)) < GROUP_DIFF_TOL:
            out[h] = rs[0]["prompt"]
    return out


def _search_rows(picks, gate, valid):
    rows = []
    for h in picks:
        if h not in valid:
            print(f"<<< WARNING fig_qual_search: hash {h} no longer validates -- "
                  f"replace it (see docstring selection procedure)")
            continue
        prompt = valid[h]
        fu = np.array([gate[(prompt, s, "full")] for s in range(8)])
        ca = np.array([gate[(prompt, s, "cached")] for s in range(8)])
        s_bon, s_com = int(np.argmax(fu)), int(np.argmax(ca))
        rows.append(dict(h=h, prompt=prompt, fu=fu, s_bon=s_bon, s_com=s_com))
        print(f"  {h}  single={fu[0]:+.2f}  bon=s{s_bon} {fu[s_bon]:+.2f}  "
              f"commit=s{s_com} {fu[s_com]:+.2f}  "
              f"{'IDENTICAL' if s_bon == s_com else f'diff (dr={fu[s_com]-fu[s_bon]:+.3f})'}"
              f"  :: {prompt[:60]}")
    return rows


# ------------------------------------------------------------ fig_qual_search
def _render_search(rows, out, zoom=None, n_frames=3):
    """Full-width 3-column money shot: single | best-of-8 | CachedSearch-commit.
    Pills as column headers, reward chips in-frame, red zoom-boxes on the
    discriminating region, italic prompt caption under each row. Manual axes
    placement so each cell exactly matches the strip aspect (no letterboxing)."""
    vdir = os.path.join(GATE_DIR, "videos")
    heads = [("single sample", C_SINGLE), ("best-of-8, full", C_BON),
             ("CachedSearch-commit", C_COMMIT)]
    n = len(rows)
    # load strips + numbers first (needed to fix the figure height to the aspect)
    grid = []
    for r in rows:
        cells = [(f"{r['h']}_0_full.mp4", r["fu"][0], C_SINGLE, None),
                 (f"{r['h']}_{r['s_bon']}_full.mp4", r["fu"][r["s_bon"]], C_BON,
                  r["s_bon"]),
                 (f"{r['h']}_{r['s_com']}_full.mp4", r["fu"][r["s_com"]], C_COMMIT,
                  r["s_com"])]
        cimgs = []
        for name, reward, col, seed in cells:
            v = read_video_np(os.path.join(vdir, name))
            cimgs.append((strip_image(v, even_idx(v, n_frames)), reward, col, seed))
            del v
        grid.append(cimgs)
    aspect = grid[0][0][0].shape[0] / grid[0][0][0].shape[1]

    W = 6.6
    lm, rm, wsp = 0.05, 0.05, 0.06         # inches
    col_w = (W - lm - rm - 2 * wsp) / 3
    strip_h = col_w * aspect
    head_pad, cap_gap, bot = 0.30, 0.35, 0.05
    H = head_pad + n * strip_h + n * cap_gap + bot
    fig = plt.figure(figsize=(W, H))
    for i, r in enumerate(rows):
        y_top = H - head_pad - i * (strip_h + cap_gap)
        y_bot = y_top - strip_h
        for j, (img, reward, col, seed) in enumerate(grid[i]):
            x = lm + j * (col_w + wsp)
            ax = fig.add_axes([x / W, y_bot / H, col_w / W, strip_h / H])
            frame_axis(ax, img, col, aspect_mode="auto")
            if zoom and r["h"] in zoom:
                redbox(ax, zoom[r["h"]])
            if i == 0:  # strategy pills as column headers on the top row only
                pill(ax, heads[j][0], heads[j][1], xy=(0.5, 1.16), ha="center",
                     va="bottom", fontsize=7.0)
            ctxt = rf"$r\,{{=}}\,{reward:+.2f}$" + (rf"  s{seed}" if seed is not None else "")
            chip(ax, ctxt, col, xy=(0.014, 0.08))
            if j == 2:  # pick-status tag on the commit strip (top-right)
                if r["s_com"] == r["s_bon"]:
                    chip(ax, "same pick", C_COMMIT, xy=(0.986, 0.90), ha="right",
                         va="top", fs=5.7)
                else:
                    dr = r["fu"][r["s_com"]] - r["fu"][r["s_bon"]]
                    chip(ax, rf"$\Delta r\,{{=}}\,{dr:+.2f}$", MUTED,
                         xy=(0.986, 0.90), ha="right", va="top", fs=5.7)
        # italic prompt caption under the row (figure coords, left-aligned)
        fig.text(lm / W, (y_bot - 0.07) / H, f"``{r['prompt']}''", fontsize=7.0,
                 color=INK, ha="left", va="top", fontstyle="italic")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", out)


def fig_search():
    gate = load_gate_scores()
    valid = validated_hashes(gate)
    rows = _search_rows(PICKS_SEARCH, gate, valid)
    _render_search(rows, os.path.join(FIGS, "fig_qual_search.pdf"),
                   zoom=ZOOM_SEARCH)
    grows = _search_rows(PICKS_SEARCH_GALLERY, gate, valid)
    _render_search(grows, os.path.join(FIGS, "fig_qual_search_gallery.pdf"))


# ----------------------------------------------------------- fig_qual_methods
def load_gate_latencies():
    """Mean per-variant latency over the official-suite gate records (seeds
    0-7). Used only for the cost chips of fig_qual_methods; the ratios match
    tab:main because cost fractions depend only on Cc/Cf."""
    lat = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(GATE_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            if r["seed"] < 8:
                lat[r["variant"]].append(r["latency"])
    return {v: float(np.mean(xs)) for v, xs in lat.items()}


def fig_methods(n_frames=2):
    """Four-method budget grid (Flash-BoN-Fig-5 style, adapted to video):
    prompts as rows, delivery strategies as columns -- single sample |
    best-of-8 full | CachedSearch-commit | CachedSearch-keep -- each cell a
    2-frame strip of the actual delivered video, with its measured reward chip
    and the strategy's measured cost (as % of best-of-8 full) in the header.
    keep shows the CACHED winner rollout itself (verifier-nominal reward
    r-hat); commit shows the full-compute regeneration of the same seed."""
    gate = load_gate_scores()
    valid = validated_hashes(gate)
    lat = load_gate_latencies()
    Cf, Cc = lat["full"], lat["cached"]
    cost = {  # strategy -> fraction of best-of-8-full wall-clock
        "single": Cf / (8 * Cf), "bon": 1.0,
        "commit": (8 * Cc + Cf) / (8 * Cf), "keep": (8 * Cc) / (8 * Cf)}
    print(f"  gate latencies: Cf={Cf:.1f}s Cc={Cc:.1f}s -> cost chips "
          + ", ".join(f"{k} {100*v:.0f}%" for k, v in cost.items()))
    rows = _search_rows(PICKS_METHODS, gate, valid)
    vdir = os.path.join(GATE_DIR, "videos")
    heads = [("single sample", C_SINGLE, cost["single"]),
             ("best-of-8, full", C_BON, cost["bon"]),
             ("CachedSearch-commit", C_COMMIT, cost["commit"]),
             ("CachedSearch-keep", C_KEEP, cost["keep"])]
    n = len(rows)
    grid = []
    for r in rows:
        prompt = r["prompt"]
        ca_sc = gate[(prompt, r["s_com"], "cached")]
        cells = [  # (file, reward, nominal?, color, seed tag)
            (f"{r['h']}_0_full.mp4", r["fu"][0], False, C_SINGLE, None),
            (f"{r['h']}_{r['s_bon']}_full.mp4", r["fu"][r["s_bon"]], False,
             C_BON, r["s_bon"]),
            (f"{r['h']}_{r['s_com']}_full.mp4", r["fu"][r["s_com"]], False,
             C_COMMIT, r["s_com"]),
            (f"{r['h']}_{r['s_com']}_cached.mp4", ca_sc, True, C_KEEP,
             r["s_com"])]
        cimgs = []
        for name, reward, nominal, col, seed in cells:
            v = read_video_np(os.path.join(vdir, name))
            cimgs.append((strip_image(v, even_idx(v, n_frames)), reward,
                          nominal, col, seed))
            del v
        grid.append(cimgs)
    aspect = grid[0][0][0].shape[0] / grid[0][0][0].shape[1]

    W = 6.6
    lm, rm, wsp = 0.05, 0.05, 0.055
    col_w = (W - lm - rm - 3 * wsp) / 4
    strip_h = col_w * aspect
    head_pad, cap_gap, bot = 0.42, 0.33, 0.03
    H = head_pad + n * strip_h + n * cap_gap + bot
    fig = plt.figure(figsize=(W, H))
    for i, r in enumerate(rows):
        y_top = H - head_pad - i * (strip_h + cap_gap)
        y_bot = y_top - strip_h
        for j, (img, reward, nominal, col, seed) in enumerate(grid[i]):
            x = lm + j * (col_w + wsp)
            ax = fig.add_axes([x / W, y_bot / H, col_w / W, strip_h / H])
            frame_axis(ax, img, col, aspect_mode="auto")
            if i == 0:  # header pill + measured cost fraction, top row only
                pill(ax, heads[j][0], heads[j][1], xy=(0.5, 1.34), ha="center",
                     va="bottom", fontsize=6.4)
                ax.annotate(f"{100 * heads[j][2]:.0f}% of best-of-8 cost",
                            xy=(0.5, 1.07), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=5.7, color=MUTED)
            rsym = r"\hat r" if nominal else "r"
            ctxt = rf"${rsym}\,{{=}}\,{reward:+.2f}$" + (
                rf"  s{seed}" if seed is not None else "")
            chip(ax, ctxt, col, xy=(0.014, 0.08), fs=5.7)
            if j == 2:
                if r["s_com"] == r["s_bon"]:
                    chip(ax, "same pick, bit-exact", C_COMMIT, xy=(0.986, 0.90),
                         ha="right", va="top", fs=5.4)
                else:
                    dr = r["fu"][r["s_com"]] - r["fu"][r["s_bon"]]
                    chip(ax, rf"$\Delta r\,{{=}}\,{dr:+.2f}$", MUTED,
                         xy=(0.986, 0.90), ha="right", va="top", fs=5.4)
        fig.text(lm / W, (y_bot - 0.06) / H, f"``{r['prompt']}''", fontsize=7.0,
                 color=INK, ha="left", va="top", fontstyle="italic")
    out = os.path.join(FIGS, "fig_qual_methods.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", out)


# -------------------------------------------------------------- fig_qual_flow
def flow_mean_mag(video, step=2, scale=0.5):
    """Per-pixel temporal-mean optical-flow magnitude (Farneback, grayscale,
    downscaled) and its scalar mean. Deterministic; no learned model."""
    import cv2
    frames = [cv2.resize(cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY), None,
                         fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
              for i in range(0, len(video), step)]
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for a, b in zip(frames[:-1], frames[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        acc += np.linalg.norm(fl, axis=2)
    acc /= max(len(frames) - 1, 1)
    return acc, float(acc.mean())


def fig_flow(n_frames=3, tag="tau020"):
    """Motion dampening made visible (sec:abl-motion): same-seed full (commit)
    vs cached (keep) winner pairs at the aggressive arm tau=0.20, each row a
    frame strip PLUS a temporal-mean optical-flow magnitude map (Farneback,
    shared color scale within a block). Chips carry the measured mean |flow|;
    the cached row moves visibly less. Everything is computed live from the
    delivered E1 videos; nothing is hand-entered."""
    manifest = {r["hash"]: r["prompt"] for r in
                map(json.loads, open(os.path.join(E3_DIR, "manifest.jsonl")))}
    conds = [("commit", "Full", C_COMMIT), ("keep", "Cached", C_KEEP)]
    blocks = []
    for h in PICKS_FLOW:
        strips, flows, means = {}, {}, {}
        for arm, _, _ in conds:
            v = read_video_np(os.path.join(TEMP_DIR, "videos", f"{h}_{tag}_{arm}.mp4"))
            strips[arm] = strip_image(v, even_idx(v, n_frames, lo=0.04, hi=0.96))
            flows[arm], means[arm] = flow_mean_mag(v)
            del v
        drop = 100 * (1 - means["keep"] / means["commit"])
        print(f"  {h}  mean|flow| full={means['commit']:.2f} "
              f"cached={means['keep']:.2f}  drop={drop:.1f}%  "
              f":: {manifest[h][:60]}")
        blocks.append((manifest[h], strips, flows, means, drop))
    aspect = blocks[0][1]["commit"].shape[0] / blocks[0][1]["commit"].shape[1]

    W = 6.6
    lm, rm, gx, gflow = 0.42, 0.03, 0.46, 0.035
    block_w = (W - lm - rm - gx) / 2
    # inside a block: frame strip + flow map (flow map is ~one frame wide)
    flow_w = block_w / (n_frames + 1)
    strip_w = block_w - flow_w - gflow
    frame_h = strip_w * aspect
    cap_h, row_gap, top = 0.17, 0.028, 0.06
    H = top + cap_h + 2 * frame_h + row_gap + 0.05
    fig = plt.figure(figsize=(W, H))
    for b, (prompt, strips, flows, means, drop) in enumerate(blocks):
        x0 = lm + b * (block_w + gx)
        vmax = max(np.percentile(flows["commit"], 99),
                   np.percentile(flows["keep"], 99))
        fig.text(x0 / W, (H - top + 0.005) / H, f"``{trunc(prompt, 52)}''",
                 fontsize=6.4, color=INK, ha="left", va="bottom",
                 fontstyle="italic")
        for a, (arm, lbl, col) in enumerate(conds):
            sy_top = H - top - cap_h + 0.10 - a * (frame_h + row_gap)
            ax = fig.add_axes([x0 / W, (sy_top - frame_h) / H,
                               strip_w / W, frame_h / H])
            frame_axis(ax, strips[arm], col, lw=1.1, aspect_mode="auto")
            ax.annotate(lbl, xy=(-0.018, 0.5), xycoords="axes fraction",
                        rotation=90, ha="right", va="center", color=col,
                        fontsize=6.2)
            chip(ax, rf"mean $|$flow$|$ {means[arm]:.2f}", col,
                 xy=(0.02, 0.09), fs=5.7)
            if arm == "keep":
                chip(ax, rf"$\tau_c{{=}}0.20$: $-{drop:.0f}\%$ motion", C_KEEP,
                     xy=(0.98, 0.09), ha="right", fs=5.7)
            axf = fig.add_axes([(x0 + strip_w + gflow) / W,
                                (sy_top - frame_h) / H,
                                flow_w / W, frame_h / H])
            axf.imshow(flows[arm], cmap="magma", vmin=0, vmax=vmax,
                       aspect="auto")
            axf.set_xticks([]); axf.set_yticks([])
            for s in axf.spines.values():
                s.set_visible(True); s.set_edgecolor(col); s.set_linewidth(1.1)
            if a == 1 and b == 0:
                axf.annotate("mean $|$flow$|$ map", xy=(0.5, -0.10),
                             xycoords="axes fraction", ha="center",
                             va="top", fontsize=5.7, color=MUTED)
    out = os.path.join(FIGS, "fig_qual_flow.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------- fig_qual_fidelity
def fig_fidelity(n_frames=2):
    """Seedance-Fig8 blocks (conditions = left row labels Full / Cached; time =
    columns) arranged 2x2 so the figure stays compact enough to float next to
    its reference in the report while keeping frames above the 1.4in floor."""
    manifest = {r["hash"]: r["prompt"] for r in
                map(json.loads, open(os.path.join(E3_DIR, "manifest.jsonl")))}
    lp = {r["key"]: r["lpips"] for r in
          map(json.loads, open(os.path.join(TEMP_DIR, "lpips_fixed.jsonl")))}
    rec = {}
    for f in glob.glob(os.path.join(TEMP_DIR, "records_shard*.jsonl")):
        for r in read_jsonl(f):
            rec[(r["prompt"], r["tau_tag"])] = r
    all_v0 = np.array([v for k, v in lp.items() if k.endswith("_v0")])
    print(f"  LPIPS(tau=0.10) suite: n={len(all_v0)} mean={all_v0.mean():.3f} "
          f"min={all_v0.min():.3f} max={all_v0.max():.3f}")

    conds = [("commit", "Full", C_COMMIT), ("keep", r"Cached $\tau_c{=}0.10$", C_KEEP)]
    # load strips + numbers (needed to size the figure to the strip aspect)
    blocks = []
    for h in PICKS_FID:
        prompt = manifest[h]
        r = rec[(prompt, "v0")]
        lpips = lp[f"{h}_v0"]
        print(f"  {h}  seed={r['seed']}  lpips={lpips:.3f}  "
              f"r_full={r['reward_commit']:+.2f}  r_cached={r['reward_keep']:+.2f}"
              f"  :: {prompt[:60]}")
        strips = {}
        for arm, _, _ in conds:
            v = read_video_np(os.path.join(TEMP_DIR, "videos", f"{h}_v0_{arm}.mp4"))
            strips[arm] = strip_image(v, even_idx(v, n_frames, lo=0.04, hi=0.96))
            del v
        blocks.append((prompt, r, lpips, strips))
    aspect = blocks[0][3]["commit"].shape[0] / blocks[0][3]["commit"].shape[1]

    W = 6.6
    lm, rm, gx = 0.42, 0.03, 0.46          # left label margin, right, inter-block gap
    block_w = (W - lm - rm - gx) / 2
    frame_h = block_w * aspect
    cap_h, row_gap, gy, top = 0.17, 0.028, 0.24, 0.05
    block_h = cap_h + 2 * frame_h + row_gap
    H = top + 2 * block_h + gy + 0.05
    fig = plt.figure(figsize=(W, H))
    for b, (prompt, r, lpips, strips) in enumerate(blocks):
        br, bc = b // 2, b % 2
        x0 = lm + bc * (block_w + gx)
        y_block_top = H - top - br * (block_h + gy)
        fig.text(x0 / W, (y_block_top + 0.005) / H,
                 f"``{trunc(prompt, 46)}''  (s{r['seed']})", fontsize=6.4,
                 color=INK, ha="left", va="bottom", fontstyle="italic")
        for a, (arm, lbl, col) in enumerate(conds):
            sy_top = y_block_top - cap_h - a * (frame_h + row_gap)
            ax = fig.add_axes([x0 / W, (sy_top - frame_h) / H, block_w / W, frame_h / H])
            frame_axis(ax, strips[arm], col, lw=1.1, aspect_mode="auto")
            ax.annotate(lbl, xy=(-0.018, 0.5), xycoords="axes fraction", rotation=90,
                        ha="right", va="center", color=col, fontsize=6.4)
            reward = r["reward_commit"] if arm == "commit" else r["reward_keep"]
            chip(ax, rf"$r\,{{=}}\,{reward:+.2f}$", col, xy=(0.02, 0.09), fs=5.8)
            if arm == "keep":
                chip(ax, rf"LPIPS ${lpips:.3f}$", C_KEEP, xy=(0.98, 0.09),
                     ha="right", fs=5.8)
    out = os.path.join(FIGS, "fig_qual_fidelity.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------ fig_qual_models
def fig_models(n_frames=4):
    ready, missing = [], []
    for label, sub, ckey in MODELS:
        vids = sorted(glob.glob(os.path.join(RESULTS, sub, "videos", "*.mp4")))
        full = [v for v in vids if v.endswith("_0_full.mp4")]
        cach = [v for v in vids if v.endswith("_0_cached.mp4")]
        recs = [r for f in glob.glob(os.path.join(RESULTS, sub, "scores_shard*.jsonl"))
                for r in read_jsonl(f)]
        by_var = {r["variant"]: r for r in recs if r["seed"] == 0}
        if full and cach and {"full", "cached"} <= set(by_var):
            ready.append((label, ckey, full[0], cach[0], by_var))
        else:
            missing.append(label)
    if missing:
        print(f"  fig_qual_models: waiting on {missing} -- "
              f"{'rendering ' + str(len(ready)) + ' rows' if len(ready) == len(MODELS) else 'SKIPPED (want all 6)'}")
    if len(ready) < len(MODELS):
        return False

    # one 2-frame strip per (model, variant); row heights track strip aspect so
    # frames stay flush across models with different native resolutions.
    strips, ratios, meta = [], [], []
    for label, ckey, fp, cp, by_var in ready:
        row = []
        for p in (fp, cp):
            v = read_video_np(p)
            row.append(strip_image(v, even_idx(v, n_frames)))
            del v
        strips.append(row)
        ratios.append(row[0].shape[0] / row[0].shape[1])
        st = by_var["cached"]["stats"]["branch0"]
        meta.append((label, ckey, by_var["full"]["score"], by_var["cached"]["score"],
                     st["skips"], st["skips"] + st["computes"]))
        print(f"  {label:14s} r_full={meta[-1][2]:+.2f} r_cached={meta[-1][3]:+.2f} "
              f"skips {st['skips']}/{st['skips'] + st['computes']}")

    n = len(ready)
    width = 6.6
    left, right, wsp = 0.155, 0.997, 0.03  # left margin reserves a logo+name gap
    col_w = (right - left) / (2 + wsp) * width  # actual per-column width (no letterbox)
    hs = [col_w * a for a in ratios]  # axes height in inches per row
    fig_h = sum(hs) + 0.20 * n + 0.24
    fig, axes = plt.subplots(n, 2, figsize=(width, fig_h), squeeze=False,
                             gridspec_kw=dict(height_ratios=hs))
    plt.subplots_adjust(left=left, right=right, top=0.955, bottom=0.006,
                        wspace=wsp, hspace=0.20 * n / fig_h * 3.4)
    for i, ((simg_f, simg_c), (label, ckey, rf_, rc_, sk, tot)) in enumerate(zip(strips, meta)):
        mcol = MODEL_COLORS[ckey]
        for j, (img, col) in enumerate([(simg_f, C_COMMIT), (simg_c, C_KEEP)]):
            ax = axes[i, j]
            frame_axis(ax, img, col)
            if j == 0:
                chip(ax, rf"$r\,{{=}}\,{rf_:+.2f}$", C_COMMIT, xy=(0.02, 0.09))
            else:
                chip(ax, rf"$r\,{{=}}\,{rc_:+.2f}$", C_KEEP, xy=(0.02, 0.09))
                chip(ax, rf"skips ${sk}/{tot}$", MUTED, xy=(0.98, 0.09), ha="right")
            if i == 0:  # spanned condition headers as pills
                pill(ax, "full compute" if j == 0 else r"cached ($\tau_c{=}0.10$)",
                     col, xy=(0.5, 1.10), ha="center", va="bottom", fontsize=6.8)
        # model name as a horizontal left row label in MODEL_COLORS (a logo-margin
        # gap is left blank to its left); horizontal avoids the rotated-label
        # collisions that short 4-frame rows would otherwise cause.
        axes[i, 0].annotate(label, xy=(-0.03, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=6.8,
                            color=mcol, fontweight="bold")
    out = os.path.join(FIGS, "fig_qual_models.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", out)
    return True


if __name__ == "__main__":
    print("[fig_qual_search]")
    fig_search()
    print("[fig_qual_methods]")
    fig_methods()
    print("[fig_qual_flow]")
    fig_flow()
    print("[fig_qual_fidelity]")
    fig_fidelity()
    print("[fig_qual_models]")
    fig_models()

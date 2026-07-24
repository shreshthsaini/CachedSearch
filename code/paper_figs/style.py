"""Central style module for every CachedSearch paper figure (design-overhaul Lane 0).

Single source of truth for:
  1. rcParams -- apply_style(): 8pt DejaVu serif EVERYWHERE (fixes the historical
     7pt drift in multimetric.py), 0.6pt axes, #e1e0d9 grid, Agg backend.
  2. Palette -- colorblind-validated accents/neutrals (dataviz six-checks
     validator on white: lightness band / chroma floor / adjacent CVD dE>=17 /
     contrast >=3:1 all PASS), entity-fixed STRATEGY_COLORS (values unchanged
     from the original make_figs.py bindings), the tau one-hue ordinal ramp,
     and the NEW MODEL_COLORS map: fixed per-model bindings used in every
     figure, table dot, and legend chip hereafter (Wan family = blue ramp;
     competitors distinct).
  3. Figure-grammar helpers implementing the shared conventions
     (PLAN-design-overhaul.md, principle 1/5):
       fit_line       solid = fit within the measured range, dashed = continuation
                      beyond it (ScaleRL grammar);
       direct_label   end-of-line labels in series color (over legends), with an
                      optional thin leader line;
       callout        white-boxed on-plot annotation with a leader arrow
                      (ScaleRL style; kills floating colored text);
       guides_to_axes dotted drop-lines from a point to both axes + value
                      annotations (Chinchilla style);
       star_point     landmark/prediction * marker;
       chip_legend    one-row colored square chips + names below the panel
                      (logo-fallback ladder: chip + wordmark).
  4. Qualitative-strip constants: FRAME_MIN_IN (per-frame width floor, inches)
     and pill() rounded strategy/condition labels (Bernini pattern).
  5. ERR_KW -- the paper-wide error-bar convention (0.6pt caps, style guide).

Every figure script imports from here; no script re-declares rcParams or
palette values locally. Marker grammar (unchanged, stated in Setup): filled =
directly measured, open = simulated over seed subsets of the measured grid,
dashed = model prediction/fit, x = held-out validation, * = landmark.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# --------------------------------------------------------------- palette
# Neutrals.
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# Accents.
C_BLUE, C_GREEN, C_YELLOW, C_RED, C_VIOLET = (
    "#2a78d6", "#199e70", "#c98500", "#e34948", "#4a3aa7")

# Strategy colors are entity-fixed everywhere (values unchanged since v1).
STRATEGY_COLORS = {
    "single": MUTED,            # #898781
    "bon_full": C_VIOLET,       # #4a3aa7
    "cached_keep": C_GREEN,     # #199e70
    "cached_commit": C_BLUE,    # #2a78d6
}
C_STRAT = STRATEGY_COLORS  # historical alias used across the fig scripts
LBL_STRAT = {"single": "single (no search)", "bon_full": "best-of-$N$ full",
             "cached_keep": "CachedSearch-keep", "cached_commit": "CachedSearch-commit"}

# tau is ordered -> one-hue ordinal blue ramp (light->dark = weak->aggressive caching).
C_TAU = {0.05: "#86b6ef", 0.10: "#256abf", 0.20: "#0d366b"}

# NEW (Lane 0): fixed per-model bindings, used EVERYWHERE hereafter -- figures,
# table dots, legend chips. Wan2.1/2.2 stay in one blue family (ramp: light 5B
# -> mid 1.3B reference -> dark 14B); each competitor family owns one accent.
MODEL_COLORS = {
    "Wan2.1-1.3B": "#2a78d6",       # = C_BLUE (the reference backbone)
    "Wan2.2-5B": "#56a0e8",         # lighter blue, same family
    "Wan2.1-14B": "#0d366b",        # dark blue, family ramp
    "CogVideoX-5B": "#c98500",      # = C_YELLOW
    "HunyuanVideo-13B": "#199e70",  # = C_GREEN
    "LTX-Video-2B": "#e34948",      # = C_RED
}
MODEL_SHORT = {  # compact display names for tight panels / chips
    "Wan2.1-1.3B": "Wan-1.3B",
    "Wan2.2-5B": "Wan2.2-5B",
    "Wan2.1-14B": "Wan-14B",
    "CogVideoX-5B": "Cog-5B",
    "HunyuanVideo-13B": "Hunyuan-13B",
    "LTX-Video-2B": "LTX-2B",
}
MODEL_MARKERS = {  # one marker per model, fixed everywhere
    "Wan2.1-1.3B": "o",
    "Wan2.2-5B": "s",
    "Wan2.1-14B": "D",
    "CogVideoX-5B": "^",
    "HunyuanVideo-13B": "v",
    "LTX-Video-2B": "P",
}
MODEL_TAGS = {  # results-dir tag (b1_gate_<tag>) -> canonical model name
    "v0": "Wan2.1-1.3B",
    "wan22_5b": "Wan2.2-5B",
    "wan14b": "Wan2.1-14B",
    "cog5b_v0": "CogVideoX-5B",
    "hunyuan_v0": "HunyuanVideo-13B",
    "ltx_v0": "LTX-Video-2B",
}

# Error-bar convention: 0.6pt caps (STYLE-GUIDE.md section 1.1).
ERR_KW = dict(elinewidth=0.7, capsize=1.5, capthick=0.6)

# ---------------------------------------------------------------------------
# Uncertainty display policy (user directive 2026-07): plots omit error bars
# and CI/variance bands for readability -- the user finds them cluttering.
# Uncertainty is reported in the text and the appendix CI tables instead.
# Rather than edit every figure, we neutralize errorbar's yerr/xerr and the
# translucent CI fill_between globally at import; markers, lines, fits, and
# fitted point estimates are unaffected. Set SHOW_UNCERTAINTY = True (before
# importing style) to restore the bars/bands.
# ---------------------------------------------------------------------------
SHOW_UNCERTAINTY = False
import matplotlib.axes as _mpl_axes
_ORIG_ERRORBAR = _mpl_axes.Axes.errorbar
_ORIG_FILL_BETWEEN = _mpl_axes.Axes.fill_between

def _clean_errorbar(self, *args, **kwargs):
    if not SHOW_UNCERTAINTY:
        kwargs["yerr"] = None
        kwargs["xerr"] = None
    return _ORIG_ERRORBAR(self, *args, **kwargs)

def _clean_fill_between(self, *args, **kwargs):
    # Skip only the translucent variance/CI bands (alpha < 0.5); leave any
    # opaque/structural fill untouched.
    if not SHOW_UNCERTAINTY and kwargs.get("alpha", 1.0) < 0.5:
        return None
    return _ORIG_FILL_BETWEEN(self, *args, **kwargs)

_mpl_axes.Axes.errorbar = _clean_errorbar
_mpl_axes.Axes.fill_between = _clean_fill_between

# Qualitative strips: per-frame rendered width floor (inches). Any frame strip
# whose frames would fall below this must drop frames or go full-width.
FRAME_MIN_IN = 1.4


# --------------------------------------------------------------- rcParams
def apply_style():
    """Canonical paper rcParams -- the single source of truth (8pt serif).

    Call once at module import in every figure script. Idempotent.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": BASE, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ------------------------------------------------------ figure-grammar helpers
def fit_line(ax, x, y, fit_range=None, color=MUTED, lw=0.9, alpha=1.0,
             zorder=2, dash=(0, (4, 2.5)), label=None):
    """Fitted-curve grammar (ScaleRL): SOLID inside `fit_range` (the measured
    support the fit was made on), DASHED continuation beyond it
    (extrapolation). `x`, `y`: the full curve to draw; `fit_range`: (lo, hi)
    in x-units, or None to draw the whole curve solid. Returns the line(s)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lines = []
    if fit_range is None:
        lines += ax.plot(x, y, "-", color=color, lw=lw, alpha=alpha,
                         zorder=zorder, label=label)
        return lines
    lo, hi = fit_range
    m = (x >= lo) & (x <= hi)
    lines += ax.plot(x[m], y[m], "-", color=color, lw=lw, alpha=alpha,
                     zorder=zorder, label=label)
    for seg in (x <= lo, x >= hi):  # boundary point repeated for continuity
        if seg.sum() > 1:
            lines += ax.plot(x[seg], y[seg], ls=dash, color=color, lw=lw,
                             alpha=alpha, zorder=zorder)
    return lines


def direct_label(ax, x, y, text, color, dx=4, dy=0, ha=None, va="center",
                 fontsize=6.5, leader=False, **kw):
    """End-of-line label in the series color -- direct labeling over legends
    (<=5 series rule). (dx, dy): offset in points from the anchor; `leader`
    draws a thin leader line from the point to the label (use when the offset
    is large enough that adjacency alone would be ambiguous)."""
    if ha is None:
        ha = "left" if dx >= 0 else "right"
    arrowprops = (dict(arrowstyle="-", lw=0.6, color=color, shrinkA=2, shrinkB=1)
                  if leader else None)
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                       ha=ha, va=va, fontsize=fontsize, color=color,
                       arrowprops=arrowprops, **kw)


def callout(ax, text, xy, xytext, color=INK, arrow_color=None, fontsize=6.2,
            ha="left", va="center", box=True, zorder=6, **kw):
    """White-boxed on-plot annotation with a thin leader arrow (ScaleRL style):
    the claim sits next to its evidence, never as floating colored text.
    `xy`: the data point; `xytext`: label position in data coords."""
    bbox = (dict(boxstyle="round,pad=0.28", fc="white", ec=BASE, lw=0.5, alpha=0.92)
            if box else None)
    return ax.annotate(text, xy=xy, xytext=xytext, textcoords="data",
                       fontsize=fontsize, color=color, ha=ha, va=va, bbox=bbox,
                       zorder=zorder,
                       arrowprops=dict(arrowstyle="-", lw=0.6,
                                       color=arrow_color or color,
                                       shrinkA=6, shrinkB=3), **kw)


def guides_to_axes(ax, x, y, color=MUTED, lw=0.7, fmt_x="{:g}", fmt_y="{:g}",
                   fontsize=6.0, annotate_x=True, annotate_y=True, zorder=1):
    """Dotted drop-lines from (x, y) to both axes + value annotations at the
    axis feet (Chinchilla style: the figure states its own numbers).
    Call AFTER the axis limits are final (reads xlim/ylim; does not autoscale)."""
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    ls = (0, (1, 1.5))
    ax.plot([x, x], [y0, y], ls=ls, lw=lw, color=color, zorder=zorder,
            scalex=False, scaley=False)
    ax.plot([x0, x], [y, y], ls=ls, lw=lw, color=color, zorder=zorder,
            scalex=False, scaley=False)
    if annotate_x:
        ax.annotate(fmt_x.format(x), (x, y0), xytext=(2, 2),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=fontsize, color=color)
    if annotate_y:
        ax.annotate(fmt_y.format(y), (x0, y), xytext=(2, 2),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=fontsize, color=color)


def star_point(ax, x, y, color, size=70, filled=False, zorder=5, label=None):
    """Landmark * marker (prediction / headline operating point). Open (white
    face) by default, matching the open-marker = simulated/derived grammar."""
    return ax.scatter([x], [y], marker="*", s=size,
                      facecolors=color if filled else "white",
                      edgecolors=color, linewidths=0.9, zorder=zorder,
                      label=label)


def chip_legend(host, mapping, y=None, fontsize=6.2, ncol=None, ms=4.5, **kw):
    """One-row legend of colored square chips + names (logo-fallback ladder:
    chip + wordmark), placed below the panel. `host`: an Axes (chips go under
    that panel) or a Figure (chips go under the whole figure). `mapping`:
    {label: color}, insertion-ordered. Extra kwargs pass through to legend()."""
    handles = [Line2D([], [], marker="s", ls="", ms=ms, mfc=c, mec="none", label=l)
               for l, c in mapping.items()]
    ncol = ncol or len(mapping)
    common = dict(handles=handles, ncol=ncol, frameon=False, fontsize=fontsize,
                  handletextpad=0.35, columnspacing=1.1, **kw)
    if isinstance(host, Figure):
        return host.legend(loc="lower center",
                           bbox_to_anchor=(0.5, y if y is not None else -0.02),
                           **common)
    return host.legend(loc="upper center",
                       bbox_to_anchor=(0.5, y if y is not None else -0.22),
                       **common)


def pill(ax, text, color, xy, fontsize=6.2, text_color="white",
         xycoords="axes fraction", ha="left", va="center", zorder=6, **kw):
    """Rounded 'pill' label (Bernini pattern) -- strategy/condition tags on
    qualitative frames. `color` fills the pill; text defaults to white."""
    return ax.annotate(text, xy=xy, xycoords=xycoords, fontsize=fontsize,
                       color=text_color, ha=ha, va=va, zorder=zorder,
                       bbox=dict(boxstyle="round,pad=0.32", fc=color, ec="none"),
                       **kw)

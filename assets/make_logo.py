"""Generate the transparent CachedSearch wordmark."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


WIDTH = 1200
HEIGHT = 260
DPI = 100
BLUE = "#1f77b4"
# Two variants: INK for light backgrounds, LIGHT for dark backgrounds
# (GitHub picks via <picture media="(prefers-color-scheme: ...)"> in README).
VARIANTS = {"logo.png": "#111827", "logo-dark.png": "#e6edf3"}


def render(out_name: str, ink: str) -> None:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cached = ax.text(
        0,
        0.5,
        "Cached",
        color=BLUE,
        fontsize=92,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        va="center",
    )
    search = ax.text(
        0,
        0.5,
        "Search",
        color=ink,
        fontsize=92,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        va="center",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cached_width = cached.get_window_extent(renderer).width
    search_width = search.get_window_extent(renderer).width

    bolt_width = 68
    bolt_gap = 30
    word_gap = 3
    total_width = bolt_width + bolt_gap + cached_width + word_gap + search_width
    start = (WIDTH - total_width) / 2
    text_start = start + bolt_width + bolt_gap

    cached.set_x(text_start / WIDTH)
    search.set_x((text_start + cached_width + word_gap) / WIDTH)

    x0 = start / WIDTH
    y0 = 0.5
    bolt = Polygon(
        [
            (x0 + 0.034, y0 + 0.35),
            (x0 + 0.003, y0 + 0.02),
            (x0 + 0.026, y0 + 0.02),
            (x0 + 0.014, y0 - 0.35),
            (x0 + 0.060, y0 + 0.10),
            (x0 + 0.036, y0 + 0.10),
        ],
        closed=True,
        facecolor=BLUE,
        edgecolor="none",
    )
    ax.add_patch(bolt)

    output = Path(__file__).with_name("logo.png")
    fig.savefig(Path(__file__).parent / out_name, transparent=True)
    plt.close(fig)
    print(output)


def main() -> None:
    for name, ink in VARIANTS.items():
        render(name, ink)


if __name__ == "__main__":
    main()

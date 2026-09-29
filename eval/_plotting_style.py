"""Shared Matplotlib defaults for evaluation notebooks."""

import matplotlib as mpl
import numpy as np
from cmcrameri import cm
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


CAT_COLORS = mpl.cm.Set2.colors
SEQ_COLORS = cm.batlow(np.linspace(0, 1, 8))

_RCPARAMS = {
    # figure
    "figure.figsize": (5, 3.5),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    # fonts
    "font.size": 6,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    # colors
    "axes.prop_cycle": mpl.cycler(
        "color",
        [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b in CAT_COLORS
        ],
    ),
    "image.cmap": "cmc.batlow",
    # axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.autolimit_mode": "round_numbers",
    "axes.xmargin": 0.0,
    "axes.ymargin": 0.0,
    # ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # lines / markers
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    # legend
    "legend.frameon": False,
    # layout
    "axes.grid": False,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
}

_drawing = False


def _visible_ticks(ticks, vmin, vmax):
    """Return only ticks that fall within [vmin, vmax]."""
    ticks = np.asarray(ticks)
    return ticks[(ticks >= vmin - 1e-10) & (ticks <= vmax + 1e-10)]


def _detach_spines(ax, offset=5):
    if ax.images:
        return

    ax.spines["left"].set_position(("outward", offset))
    has_vertical_bars = any(
        isinstance(container, BarContainer) and container.orientation == "vertical"
        for container in ax.containers
    )
    ax.spines["bottom"].set_position(
        ("outward", 0 if has_vertical_bars else offset)
    )

    xticks = _visible_ticks(ax.get_xticks(), *ax.get_xlim())
    yticks = _visible_ticks(ax.get_yticks(), *ax.get_ylim())
    if has_vertical_bars:
        ax.spines["bottom"].set_bounds(None, None)
    elif len(xticks) > 1:
        ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])
    if len(yticks) > 1:
        ax.spines["left"].set_bounds(yticks[0], yticks[-1])

    if getattr(ax, "extend_clip", True):
        bbox = ax.get_window_extent()
        bbox = bbox.expanded(
            1.0 + offset / bbox.width,
            1.0 + offset / bbox.height,
        )
        for artist in ax.collections + ax.lines + ax.patches:
            artist.set_clip_box(bbox)


def _install_detached_spines():
    """Install the draw hook once, including across notebook cell re-runs."""
    if getattr(Axes.draw, "_implicit_nuisance_detached_spines", False):
        return

    original_draw = Axes.draw

    def draw_with_detached_spines(self, renderer):
        global _drawing

        if not _drawing and getattr(self, "detach_spines", True):
            _drawing = True
            try:
                _detach_spines(self)
            finally:
                _drawing = False
        return original_draw(self, renderer)

    Axes.draw = draw_with_detached_spines


def setup_plotting():
    """Apply the plot style and enable automatically detached spines.

    Set ``ax.detach_spines = False`` to disable detached spines for an axes.
    Set ``ax.extend_clip = False`` to retain Matplotlib's original clip box.
    Image axes are skipped automatically.
    """
    mpl.rcParams.update(_RCPARAMS)
    _install_detached_spines()


setup_plotting()

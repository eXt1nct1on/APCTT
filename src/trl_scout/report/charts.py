"""Matplotlib chart generation for self-contained reports."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trl_scout.schemas import LevelScore


def _data_uri(figure: plt.Figure) -> str:
    """Render a figure as a PNG data URI."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight", transparent=False)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def trl_ladder(levels: Sequence[LevelScore], current: int) -> str:
    """Create the nine-step TRL ladder graphic."""
    scores = {level.level: level.score for level in levels}
    figure, axis = plt.subplots(figsize=(10, 1.65))
    axis.set_xlim(0.5, 9.5)
    axis.set_ylim(0, 1)
    axis.axis("off")
    for level in range(1, 10):
        achieved = level < current or scores.get(level, 0) >= 0.6
        is_current = level == current
        face = "#1f6f78" if achieved else "#ffffff"
        edge = "#0e3f45" if is_current else "#9cb7b9"
        line_width = 2.4 if is_current else 1.2
        rectangle = plt.Rectangle(
            (level - 0.38, 0.25),
            0.76,
            0.5,
            facecolor=face,
            edgecolor=edge,
            linewidth=line_width,
        )
        axis.add_patch(rectangle)
        axis.text(
            level,
            0.5,
            str(level),
            ha="center",
            va="center",
            color="white" if achieved else "#183234",
            fontsize=11,
            fontweight="bold",
        )
    axis.text(0.5, 0.05, "Basic principles", ha="left", va="bottom", fontsize=8, color="#526668")
    axis.text(9.5, 0.05, "Operational system", ha="right", va="bottom", fontsize=8, color="#526668")
    return _data_uri(figure)

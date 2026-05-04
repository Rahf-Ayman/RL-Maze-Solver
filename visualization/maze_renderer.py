"""Maze rendering utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def render_maze_snapshot(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    path: Optional[Iterable[Tuple[int, int]]] = None,
    title: str = "Maze Snapshot",
    save_path: Optional[str] = None,
):
    """Render a static maze image using matplotlib.

    Args:
        grid: 2D maze grid where 1=wall and 0=free.
        agent_pos: Current (row, col) of agent.
        goal_pos: Goal (row, col) position.
        path: Optional visited path to overlay.
        title: Plot title.
        save_path: Optional file path to save the plot.

    Returns:
        (figure, axes) tuple for downstream customization.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for maze rendering") from exc

    canvas = np.array(grid, dtype=np.float32)
    display = np.zeros_like(canvas)
    display[canvas == 1] = 0.15
    display[canvas == 0] = 0.95

    if path is not None:
        for row, col in path:
            if 0 <= row < display.shape[0] and 0 <= col < display.shape[1] and display[row, col] > 0.2:
                display[row, col] = 0.75

    goal_row, goal_col = goal_pos
    agent_row, agent_col = agent_pos
    display[goal_row, goal_col] = 0.55
    display[agent_row, agent_col] = 0.25

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title)

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure, axes

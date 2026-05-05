"""Maze rendering utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def render_maze_snapshot(
    grid: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    start_pos: Optional[Tuple[int, int]] = None,
    path: Optional[Iterable[Tuple[int, int]]] = None,
    title: str = "Maze Snapshot",
    save_path: Optional[str] = None,
    ax=None,
    figure=None,
):
    """Render a static maze image using matplotlib.

    Args:
        grid: 2D maze grid where 1=wall and 0=free.
        agent_pos: Current (row, col) of agent.
        goal_pos: Goal (row, col) position.
        start_pos: Optional start/source position to mark on the maze.
        path: Optional visited path to overlay.
        title: Plot title.
        save_path: Optional file path to save the plot.
        ax: Optional matplotlib axes to draw onto.
        figure: Optional matplotlib figure to use when ax is provided.

    Returns:
        (figure, axes) tuple for downstream customization.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for maze rendering") from exc

    # canvas = np.array(grid, dtype=np.float32)
    display = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=float)
    display[grid == 1] = (0, 0, 0) # walls = black
    display[grid == 0] = (1, 1, 1) # free cells = white

    if path is not None:
        for row, col in path:
            if 0 <= row < display.shape[0] and 0 <= col < display.shape[1] and grid[row, col] == 0:
                display[row, col] = (0.6, 0.8, 1) # path = light blue

    if start_pos is not None:
        start_row, start_col = start_pos
        if 0 <= start_row < display.shape[0] and 0 <= start_col < display.shape[1]:
            display[start_row, start_col] = (1, 0.8, 0) # source/start = yellow

    goal_row, goal_col = goal_pos
    agent_row, agent_col = agent_pos
    display[goal_row, goal_col] = (0, 1, 0) # goal = green
    display[agent_row, agent_col] = (1, 0, 0) # agent = red

    if ax is None:
        figure, axes = plt.subplots(figsize=(6, 6))
    else:
        axes = ax
        figure = figure or axes.figure
        axes.clear()

    axes.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title)

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure, axes

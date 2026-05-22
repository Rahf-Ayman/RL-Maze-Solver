"""Maze rendering utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

COLORS_RGB = {
    "wall": (0.1686, 0.1765, 0.2588),      # #2B2D42
    "unvisited": (0.9725, 0.9765, 0.9804), # #F8F9FA
    "visited": (0.95, 0.65, 0.68),   # #B8C5D6
    "start": (0.3059, 0.6588, 0.8706),     # #4EA8DE
    "goal": (1.0000, 0.8196, 0.4000),      # #FFD166
    "agent": (0.9373, 0.2784, 0.4353)     # #EF476F
}

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
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for maze rendering") from exc

    # canvas = np.array(grid, dtype=np.float32)
    display = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=float)
    display[grid == 1] = COLORS_RGB["wall"] # walls = black
    display[grid == 0] = COLORS_RGB["unvisited"] # free cells = white

    if path is not None:
        for row, col in path:
            if 0 <= row < display.shape[0] and 0 <= col < display.shape[1] and grid[row, col] == 0:
                display[row, col] = COLORS_RGB["visited"] # path = light pink

    if start_pos is not None:
        start_row, start_col = start_pos
        if 0 <= start_row < display.shape[0] and 0 <= start_col < display.shape[1]:
            display[start_row, start_col] = COLORS_RGB["start"] # source/start = blue

    goal_row, goal_col = goal_pos
    agent_row, agent_col = agent_pos
    display[goal_row, goal_col] = COLORS_RGB["goal"] # goal = yellow
    display[agent_row, agent_col] = COLORS_RGB["agent"] # agent = red

    if ax is None:
        figure, axes = plt.subplots(figsize=(6, 6))
    else:
        axes = ax
        figure = figure or axes.figure
        axes.clear()

    axes.imshow(display)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title)

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure, axes

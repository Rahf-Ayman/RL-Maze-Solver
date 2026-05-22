"""Q-Learning policy heatmap visualization."""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents.q_learning_agent import QLearningAgent

ACTION_TO_ARROW = {
    0: "U",  # up
    1: "D",  # down
    2: "L",  # left
    3: "R",  # right
}


def plot_q_policy_heatmap(
    env,
    agent: QLearningAgent,
    title: str = "Q-Learning Policy Heatmap",
    annotate_policy: bool = True,
    save_path: Optional[str] = None,
):
    """Visualize max-Q values over the maze and best actions per free cell.

    Args:
        env: MazeEnv-compatible object with grid, height, width, and _encode_state.
        agent: QLearningAgent instance with q_table.
        title: Plot title.
        annotate_policy: Draw action arrows at each free cell.
        save_path: Optional output image path.

    Returns:
        (figure, axes) tuple.
    """

    if not isinstance(agent, QLearningAgent):
        raise TypeError("Policy heatmap currently supports QLearningAgent only.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for policy heatmap plotting") from exc

    values = np.full((env.height, env.width), np.nan, dtype=float)
    best_actions = np.full((env.height, env.width), -1, dtype=int)

    for row in range(env.height):
        for col in range(env.width):
            if int(env.grid[row, col]) == 1: # wall
                continue
            state = env._encode_state((row, col))
            q_values = np.asarray(agent.q_table[state], dtype=float)
            values[row, col] = float(q_values.max())
            best_actions[row, col] = int(np.argmax(q_values))

    figure, axes = plt.subplots(figsize=(7, 6))
    image = axes.imshow(values, cmap="viridis")
    axes.set_title(title)
    axes.set_xticks([])
    axes.set_yticks([])

    # Overlay walls to make blocked cells explicit.
    for row in range(env.height):
        for col in range(env.width):
            if int(env.grid[row, col]) == 1:
                axes.text(col, row, "#", ha="center", va="center", color="black", fontsize=7)
            elif annotate_policy:
                action = best_actions[row, col]
                if action >= 0:
                    axes.text(
                        col,
                        row,
                        ACTION_TO_ARROW.get(action, "?"),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

    goal_row, goal_col = env.goal_pos
    start_row, start_col = env.start_pos
    axes.text(goal_col, goal_row, "G", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    axes.text(start_col, start_row, "S", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="max Q(s, a)")
    figure.tight_layout()

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure, axes

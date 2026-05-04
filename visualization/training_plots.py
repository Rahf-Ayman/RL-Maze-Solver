"""Training metric plots for Q-Learning runs."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return np.array([])

    if len(values) < window:
        window = len(values)

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def plot_training_curves(
    logger,
    moving_average_window: int = 50,
    title_prefix: str = "Q-Learning Training",
    save_path: Optional[str] = None,
):
    """Plot reward, steps, and rolling success rate from a training logger.

    Args:
        logger: training.Logger instance.
        moving_average_window: Window for smoothing curves.
        title_prefix: Prefix used in subplot titles.
        save_path: Optional output image path.

    Returns:
        (figure, axes) for further customization.
    """

    if not logger.rewards:
        raise ValueError("Logger is empty. Train the agent before plotting.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for training plots") from exc

    episodes = np.arange(1, len(logger.rewards) + 1)
    reward_ma = _rolling_mean(logger.rewards, moving_average_window)
    step_ma = _rolling_mean([float(step) for step in logger.steps], moving_average_window)
    success_float = [1.0 if value else 0.0 for value in logger.success]
    success_ma = _rolling_mean(success_float, moving_average_window)

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(episodes, logger.rewards, alpha=0.35, linewidth=1.0, label="Reward")
    reward_x = np.arange(moving_average_window, moving_average_window + len(reward_ma))
    axes[0].plot(reward_x, reward_ma, linewidth=2.0, label=f"Reward MA({moving_average_window})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"{title_prefix}: Rewards")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, logger.steps, alpha=0.35, linewidth=1.0, label="Steps")
    steps_x = np.arange(moving_average_window, moving_average_window + len(step_ma))
    axes[1].plot(steps_x, step_ma, linewidth=2.0, label=f"Steps MA({moving_average_window})")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title(f"{title_prefix}: Episode Length")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    success_x = np.arange(moving_average_window, moving_average_window + len(success_ma))
    axes[2].plot(success_x, success_ma, linewidth=2.0, label=f"Success Rate MA({moving_average_window})")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(f"{title_prefix}: Rolling Success")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return figure, axes

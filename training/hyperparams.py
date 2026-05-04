"""Training hyperparameters for the RL maze solver project.

The project plan describes hyperparameter sets for Q-Learning on each difficulty.
These are referenced by the Trainer to configure the agent and episode budget.
"""

from __future__ import annotations

from typing import Any, Dict

# ============================================================================
# Q-Learning Hyperparameters
# ============================================================================

Q_LEARNING_CONFIG: Dict[str, Any] = {
    "alpha": 0.1,  # Learning rate — how fast Q-values update (Bellman equation)
    "gamma": 0.99,  # Discount factor — weight future rewards vs immediate
    "epsilon_start": 1.0,  # Start fully exploratory (random actions)
    "epsilon_end": 0.01,  # End almost fully exploitative (greedy)
    "epsilon_decay": 0.995,  # Multiply epsilon by this each episode
}

# Per-difficulty training budgets (episodes to train)
EPISODE_CONFIG: Dict[str, Dict[str, int]] = {
    "easy": {
        "n_episodes": 500,  # Easy mazes converge quickly
        "eval_interval": 50,  # Log every 50 episodes
    },
    "medium": {
        "n_episodes": 1000,  # Medium needs more exploration
        "eval_interval": 100,
    },
    "hard": {
        "n_episodes": 2000,  # Hard mazes need extensive training
        "eval_interval": 200,
    },
}


def get_q_learning_config() -> Dict[str, Any]:
    """Return a copy of the Q-Learning hyperparameter config."""
    return Q_LEARNING_CONFIG.copy()


def get_episode_config(difficulty: str) -> Dict[str, int]:
    """Return episode budget for the given difficulty level."""
    key = difficulty.lower().strip()
    if key not in EPISODE_CONFIG:
        valid = ", ".join(sorted(EPISODE_CONFIG))
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of: {valid}")
    return EPISODE_CONFIG[key].copy()

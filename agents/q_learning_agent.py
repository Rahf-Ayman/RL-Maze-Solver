"""Tabular Q-Learning agent implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Simple tabular Q-Learning agent.

    The agent expects a discrete observation space (states encoded as integers)
    and a discrete action space. The agent can be constructed with either an
    environment object that exposes `observation_space` and `action_space`, or
    by providing `n_states` and `n_actions` directly.
    """

    def __init__(
        self,
        env: Optional[object] = None,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if env is not None:
            # env should expose observation_space.n and action_space.n
            n_states = getattr(env.observation_space, "n", None) if n_states is None else n_states
            n_actions = getattr(env.action_space, "n", None) if n_actions is None else n_actions

        if n_states is None or n_actions is None:
            raise ValueError("n_states and n_actions must be provided (or an env with discrete spaces)")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.rng = np.random.default_rng(seed)

        # Q-table: shape (n_states, n_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""

        if self.rng.random() < self.epsilon: ## explore
            return int(self.rng.integers(self.n_actions))

        # argmax with tie-breaking using random choice among ties
        q_values = self.q_table[int(state)] ## expolit
        top = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(top))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Standard Bellman TD update for tabular Q-Learning."""

        s = int(state)
        a = int(action)
        ns = int(next_state)

        best_next = 0.0 if done else float(self.q_table[ns].max())
        td_target = float(reward) + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def on_episode_end(self) -> None:
        """Decay epsilon after each episode (clamped at epsilon_end)."""

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        """Reset any per-episode state — Q-table stays intact."""

        pass

"""Abstract base agent definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for agents used by the trainer.

    Concrete agents should implement `select_action` and `update` at minimum.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """Return an action for the given state."""

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Update the agent from a single transition."""

    def on_episode_end(self) -> None:
        """Called after each episode — useful for epsilon decay or cleanup."""

    def reset(self) -> None:
        """Reset internal agent state between training runs if needed."""

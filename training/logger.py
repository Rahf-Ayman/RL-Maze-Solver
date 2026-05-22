"""Basic training logger for the RL maze solver project.

The logger keeps the core episode metrics described in the project plan:
episode number, total reward, episode length, and success flag. It also
provides a moving average helper, a rolling success-rate helper, and CSV export
for later analysis or plotting.
"""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import Iterable, List, Sequence


class Logger:
    """Collect and summarize training metrics across episodes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all recorded metrics."""

        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self.success: List[bool] = []

    def log(self, episode: int, reward: float, steps: int, success: bool = False) -> None:
        """Record one episode of training metrics."""

        self.episodes.append(int(episode))
        self.rewards.append(float(reward))
        self.steps.append(int(steps))
        self.success.append(bool(success))

    def moving_average(self, window: int = 50) -> List[float]:
        """Return a simple moving average of rewards.

        If fewer than ``window`` episodes have been logged, an empty list is
        returned so callers can decide how to plot or handle the partial data.
        """

        if window <= 0:
            raise ValueError("window must be a positive integer")
        if len(self.rewards) < window:
            return []

        averages: List[float] = []
        running_total = sum(self.rewards[:window])
        averages.append(running_total / window)

        for index in range(window, len(self.rewards)):
            running_total += self.rewards[index] - self.rewards[index - window]
            averages.append(running_total / window)

        return averages

    def success_rate(self, last_n: int = 100) -> float:
        """Return the success rate over the most recent ``last_n`` episodes."""

        if last_n <= 0:
            raise ValueError("last_n must be a positive integer")
        if not self.success:
            return 0.0

        recent = self.success[-last_n:]
        return sum(1.0 for value in recent if value) / len(recent)

    def export_csv(self, path: str | Path) -> Path:
        """Write all logged metrics to a CSV file and return the written path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = DictWriter(file_handle, fieldnames=["episode", "reward", "steps", "success"])
            writer.writeheader()
            for episode, reward, steps, success in zip(self.episodes, self.rewards, self.steps, self.success):
                writer.writerow(
                    {
                        "episode": episode,
                        "reward": reward,
                        "steps": steps,
                        "success": success,
                    }
                )

        return output_path

    def as_rows(self) -> List[dict[str, int | float | bool]]:
        """Return the logged metrics as a list of row dictionaries."""

        return [
            {
                "episode": episode,
                "reward": reward,
                "steps": steps,
                "success": success,
            }
            for episode, reward, steps, success in zip(self.episodes, self.rewards, self.steps, self.success)
        ]

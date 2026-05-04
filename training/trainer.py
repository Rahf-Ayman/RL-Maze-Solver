"""Training orchestration for the RL maze solver project.

The Trainer manages the core training loop: episodes, environment resets,
agent updates, logging, and metric tracking. It's designed to work with
any agent that inherits from BaseAgent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents import QLearningAgent
from envs import MazeEnv
from training.logger import Logger
from visualization import plot_q_policy_heatmap, plot_training_curves, render_maze_snapshot


class Trainer:
    """Orchestrate training episodes and agent learning.

    Attributes:
        env: The Gymnasium-compatible environment.
        agent: The learning agent (must have select_action, update, on_episode_end).
        logger: Tracks episode metrics.
        is_running: Flag to pause/resume training.
    """

    def __init__(
        self,
        env: MazeEnv,
        agent: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            env: A MazeEnv instance.
            agent: A BaseAgent subclass instance.
            config: Optional dict with hyperparameters (alpha, gamma, epsilon_*, etc).
        """
        self.env = env
        self.agent = agent
        self.config = config or {}
        self.logger = Logger()
        self.is_running = False
        self.current_episode = 0

    def train(
        self,
        n_episodes: int,
        verbose: bool = True,
        verbose_interval: int = 100,
    ) -> Logger:
        """Run n episodes of training.

        Args:
            n_episodes: Number of episodes to train.
            verbose: Whether to print progress every verbose_interval episodes.
            verbose_interval: Print progress every N episodes.

        Returns:
            The logger instance containing all recorded metrics.
        """
        self.is_running = True
        self.logger.reset()

        for episode in range(n_episodes):
            if not self.is_running:
                break

            self.current_episode = episode
            episode_reward, episode_steps, episode_success = self._run_episode()

            # Log this episode
            self.logger.log(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
                success=episode_success,
            )

            # Post-episode hook (e.g., epsilon decay)
            self.agent.on_episode_end()

            # Periodic verbose output
            if verbose and (episode + 1) % verbose_interval == 0:
                ma = self.logger.moving_average(window=verbose_interval)
                recent_avg = ma[-1] if ma else episode_reward
                recent_success = self.logger.success_rate(last_n=verbose_interval)
                print(
                    f"Episode {episode + 1:5d}/{n_episodes} | "
                    f"Reward: {recent_avg:7.2f} | "
                    f"Success Rate: {recent_success:5.1%} | "
                    f"Steps: {episode_steps:4d}"
                )

        self.is_running = False
        return self.logger

    def _run_episode(self) -> tuple[float, int, bool]:
        """Execute one full episode.

        Returns:
            (total_reward, episode_length, success_flag)
        """
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Agent selects action
            action = self.agent.select_action(obs)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Agent learns from the transition
            self.agent.update(obs, action, reward, next_obs, terminated or truncated)

            total_reward += reward
            steps += 1
            done = terminated or truncated
            obs = next_obs

        # Success if episode terminated (not truncated)
        success = terminated if "terminated" in locals() else False
        return total_reward, steps, success

    def pause(self) -> None:
        """Stop the training loop gracefully."""
        self.is_running = False

    def resume(self, n_episodes: int, **kwargs) -> Logger:
        """Resume training for n more episodes."""
        return self.train(n_episodes=n_episodes, **kwargs)

    def render_maze(self, path=None, title: Optional[str] = None, save_path: Optional[str] = None):
        """Render a static maze snapshot for the current environment state."""

        plot_title = title or f"Maze ({self.env.difficulty.title()})"
        return render_maze_snapshot(
            grid=self.env.grid,
            agent_pos=self.env.agent_pos,
            goal_pos=self.env.goal_pos,
            path=path,
            title=plot_title,
            save_path=save_path,
        )

    def plot_training(self, moving_average_window: int = 50, save_path: Optional[str] = None):
        """Plot reward, episode length, and rolling success from logger data."""

        return plot_training_curves(
            logger=self.logger,
            moving_average_window=moving_average_window,
            title_prefix="Q-Learning Training",
            save_path=save_path,
        )

    def plot_q_policy_heatmap(self, annotate_policy: bool = True, save_path: Optional[str] = None):
        """Plot Q-value heatmap for QLearningAgent only."""

        if not isinstance(self.agent, QLearningAgent):
            raise TypeError("Policy heatmap is only implemented for QLearningAgent.")

        return plot_q_policy_heatmap(
            env=self.env,
            agent=self.agent,
            annotate_policy=annotate_policy,
            save_path=save_path,
        )

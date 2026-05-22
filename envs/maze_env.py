"""Gymnasium-compatible maze environment for reinforcement learning.

The environment follows the project plan:
- discrete state space encoded as row * width + col
- four cardinal actions
- shaped rewards for goal reach, wall collisions, and timeouts
- a DFS-generated maze built from the selected difficulty preset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from .maze_configs import DEFAULT_START_POS, get_difficulty_config
from .maze_generator import generate_maze
from visualization import  render_maze_snapshot
from gymnasium import spaces


ACTION_MAP = {
	0: (-1, 0), ## up
	1: (1, 0),  ## down
	2: (0, -1), ## left
	3: (0, 1),  ## right
}


class MazeEnv(gym.Env):
	"""Maze navigation environment with a discrete state and action space."""

	def __init__(self, difficulty: str = "medium", goal: Optional[Tuple[int, int]] = None, render_mode: Optional[str] = None, seed: Optional[int] = None):
		super().__init__()

		self.difficulty = difficulty
		self.config = get_difficulty_config(difficulty)
		self._base_seed = self.config.get("seed") if seed is None else seed

		self.height = int(self.config["height"])
		self.width = int(self.config["width"])
		self.start_pos = tuple(self.config.get("start_pos", DEFAULT_START_POS))
		self.max_steps = int(self.config["max_steps"])

		self.grid = generate_maze(self.height, self.width, seed=self._base_seed)
		self.goal_pos = self._validate_goal(goal or self.config["default_goal"])

		self.observation_space = spaces.Discrete(self.height * self.width)
		self.action_space = spaces.Discrete(len(ACTION_MAP))

		self.agent_pos = self.start_pos
		self.steps = 0
		self._figure = None
		self._axes = None

	def _validate_goal(self, goal: Tuple[int, int]) -> Tuple[int, int]:
		row, col = int(goal[0]), int(goal[1])
		if not (0 < row < self.height - 1 and 0 < col < self.width - 1) or self.grid[row, col] == 1:
			raise ValueError("Goal cannot be placed on a wall or outside the maze boundaries.")
		self.grid[row, col] = 0
		return row, col

	def _encode_state(self, position: Tuple[int, int]) -> int:
		return position[0] * self.width + position[1]

	def _decode_state(self, state: int) -> Tuple[int, int]:
		row = int(state) // self.width # ignore remainder
		col = int(state) % self.width
		return row, col

	def compute_reward(self, hit_wall: bool, terminated: bool, truncated: bool) -> float:
		if terminated:
			return 10.0
		if truncated:
			return -1.0
		if hit_wall:
			return -0.5
		return -0.01

	def reset(self, *, seed: Optional[int] = None):
		super().reset(seed=seed)

		if seed is not None and seed != self._base_seed:
			self._base_seed = seed
			self.grid = generate_maze(self.height, self.width, seed=seed)
			self.goal_pos = self._validate_goal(self.goal_pos)

		self.agent_pos = self.start_pos
		self.steps = 0
		observation = self._encode_state(self.agent_pos)
		info = {
			"position": self.agent_pos,
			"goal": self.goal_pos,
			"steps": self.steps,
		}
		return observation, info

	def step(self, action: int):
		if not self.action_space.contains(action):
			raise ValueError(f"Invalid action {action}. Expected one of 0, 1, 2, 3.")

		row, col = self.agent_pos
		delta_row, delta_col = ACTION_MAP[int(action)]
		next_row = row + delta_row
		next_col = col + delta_col

		hit_wall = False
		if 0 <= next_row < self.height and 0 <= next_col < self.width and self.grid[next_row, next_col] == 0:
			self.agent_pos = (next_row, next_col)
		else:
			hit_wall = True

		self.steps += 1
		terminated = self.agent_pos == self.goal_pos
		truncated = self.steps >= self.max_steps and not terminated
		reward = self.compute_reward(hit_wall=hit_wall, terminated=terminated, truncated=truncated)

		observation = self._encode_state(self.agent_pos)
		info = {
			"position": self.agent_pos,
			"goal": self.goal_pos,
			"steps": self.steps,
			"hit_wall": hit_wall,
			"state": observation,
			"manhattan_distance": abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1]),
		}
		return observation, reward, terminated, truncated, info

	def set_goal(self, new_goal: Tuple[int, int]) -> None:
		self.goal_pos = self._validate_goal(new_goal)

	def render(self): ## for test in notebook
		self._figure, self._axes = render_maze_snapshot(
			grid=self.grid,
			agent_pos=self.agent_pos,
			goal_pos=self.goal_pos,
			start_pos=self.start_pos,
			path=None,  # or your visited path if you track it
			title=f"MazeEnv: {self.difficulty.title()} | Seed {self._base_seed}",
		)
		return self._figure

	def close(self):
		if self._figure is not None:
			try:
				import matplotlib.pyplot as plt

				plt.close(self._figure)
			except ImportError:
				pass
			finally:
				self._figure = None
				self._axes = None


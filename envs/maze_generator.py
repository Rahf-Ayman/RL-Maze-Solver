"""Procedural maze generation helpers.

The project plan uses a recursive DFS/backtracking maze generator to create a
perfect maze: every reachable passage is connected and there is exactly one
path between any two free cells.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_dimensions(height: int, width: int) -> None:
	if height < 3 or width < 3:
		raise ValueError("Maze dimensions must be at least 3x3.")
	if height % 2 == 0 or width % 2 == 0:
		raise ValueError("Maze dimensions must be odd so passages and walls alternate cleanly.")


def generate_maze(height: int, width: int, seed: Optional[int] = None) -> np.ndarray:
	"""Generate a perfect maze using DFS backtracking.

	Walls are encoded as 1 and free passages as 0.
	The outer border remains walls, and the generator carves passages on odd
	coordinates starting from (1, 1).
	"""

	_validate_dimensions(height, width)

	rng = np.random.default_rng(seed)
	grid = np.ones((height, width), dtype=np.int8)

	start = (1, 1)
	grid[start] = 0
	stack = [start]
	directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

	while stack:
		row, col = stack[-1]
		neighbors = []

		for dr, dc in directions:
			next_row = row + dr
			next_col = col + dc
			if 0 < next_row < height - 1 and 0 < next_col < width - 1 and grid[next_row, next_col] == 1:
				neighbors.append((next_row, next_col, dr // 2, dc // 2))

		if not neighbors:
			stack.pop()
			continue

		chosen_index = int(rng.integers(len(neighbors)))
		next_row, next_col, wall_row, wall_col = neighbors[chosen_index]
		grid[row + wall_row, col + wall_col] = 0
		grid[next_row, next_col] = 0
		stack.append((next_row, next_col))

	return grid


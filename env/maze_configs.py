"""Difficulty presets for the maze environment.

The project plan describes three presets:
easy, medium, and hard.  The sizes below keep the mazes odd so the
DFS maze generator can preserve wall corridors between passages.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_START_POS = (1, 1)


DIFFICULTY_CONFIGS: Dict[str, Dict[str, Any]] = {
	"easy": {
		"height": 7,
		"width": 7,
		"max_steps": 7 * 7 * 2,
		"default_goal": (5, 5),
		"wall_density": "low",
		"seed": 7,
	},
	"medium": {
		"height": 11,
		"width": 11,
		"max_steps": 11 * 11 * 2,
		"default_goal": (9, 9),
		"wall_density": "medium",
		"seed": 11,
	},
	"hard": {
		"height": 15,
		"width": 15,
		"max_steps": 15 * 15 * 2,
		"default_goal": (13, 13),
		"wall_density": "high",
		"seed": 15,
	},
}


def get_difficulty_config(difficulty: str) -> Dict[str, Any]:
	"""Return a copy of the preset config for the requested difficulty."""

	key = difficulty.lower().strip()
	if key not in DIFFICULTY_CONFIGS:
		valid = ", ".join(sorted(DIFFICULTY_CONFIGS))
		raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of: {valid}")
	return deepcopy(DIFFICULTY_CONFIGS[key])


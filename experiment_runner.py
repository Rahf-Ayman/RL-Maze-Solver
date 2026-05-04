"""Manual experiment runner for Q-Learning maze training.

Edit the USER CONFIG section below, then run:
    C:/Users/HP/AppData/Local/Python/pythoncore-3.14-64/python.exe experiment_runner.py

This script exports, for every run:
- episode metrics CSV
- maze snapshot PNG
- training curves PNG
- Q-policy heatmap PNG
- per-run JSON summary

It also writes experiment-level summary CSV/JSON.
"""

from __future__ import annotations

import json
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agents import QLearningAgent
from envs import MazeEnv
from training import Trainer, get_episode_config, get_q_learning_config


# ============================================================================
# USER CONFIG (EDIT THESE VALUES)
# ============================================================================
EXPERIMENT_NAME = "manual_qlearning_experiment"
OUTPUT_ROOT = Path("experiments")

DIFFICULTIES = ["medium"]
RUNS_PER_DIFFICULTY = 1

# Use project defaults when value is None.
# Example override: {"easy": 200, "medium": 600, "hard": 1200}
EPISODES_OVERRIDE_BY_DIFFICULTY = {"medium": 500}

# Q-Learning hyperparameter overrides.
# Example: {"alpha": 0.05, "epsilon_decay": 0.997}
Q_CONFIG_OVERRIDES: Dict[str, float] = {}

# Seed control for reproducibility.
# If GLOBAL_SEED is None, runs are non-deterministic.
GLOBAL_SEED = 42
SEED_STEP = 1000

# Plot and logging behavior
MOVING_AVERAGE_WINDOW = 50
VERBOSE_TRAINING = True
VERBOSE_INTERVAL = 100
SHOW_PLOTS_INTERACTIVELY = False

# Export toggles
EXPORT_MAZE_SNAPSHOT = True
EXPORT_TRAINING_CURVES = True
EXPORT_POLICY_HEATMAP = True
# ============================================================================


def _resolve_episodes(difficulty: str) -> int:
    if EPISODES_OVERRIDE_BY_DIFFICULTY is not None and difficulty in EPISODES_OVERRIDE_BY_DIFFICULTY:
        return int(EPISODES_OVERRIDE_BY_DIFFICULTY[difficulty])
    return int(get_episode_config(difficulty)["n_episodes"])


def _run_seed(run_index: int) -> int | None:
    if GLOBAL_SEED is None:
        return None
    return int(GLOBAL_SEED + run_index * SEED_STEP)


def _create_experiment_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def _write_experiment_config(experiment_dir: Path) -> None:
    config_data = {
        "experiment_name": EXPERIMENT_NAME,
        "difficulties": DIFFICULTIES,
        "runs_per_difficulty": RUNS_PER_DIFFICULTY,
        "episodes_override_by_difficulty": EPISODES_OVERRIDE_BY_DIFFICULTY,
        "q_config_overrides": Q_CONFIG_OVERRIDES,
        "global_seed": GLOBAL_SEED,
        "seed_step": SEED_STEP,
        "moving_average_window": MOVING_AVERAGE_WINDOW,
        "verbose_training": VERBOSE_TRAINING,
        "verbose_interval": VERBOSE_INTERVAL,
        "export_maze_snapshot": EXPORT_MAZE_SNAPSHOT,
        "export_training_curves": EXPORT_TRAINING_CURVES,
        "export_policy_heatmap": EXPORT_POLICY_HEATMAP,
    }
    with (experiment_dir / "experiment_config.json").open("w", encoding="utf-8") as file_handle:
        json.dump(config_data, file_handle, indent=2)


def _run_single(
    experiment_dir: Path,
    difficulty: str,
    run_number: int,
    run_index: int,
) -> Dict[str, float | int | str | bool]:
    run_dir = experiment_dir / difficulty / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_seed = _run_seed(run_index)
    episodes = _resolve_episodes(difficulty)

    env = MazeEnv(difficulty=difficulty, seed=run_seed)

    q_config = get_q_learning_config()
    q_config.update(Q_CONFIG_OVERRIDES)
    agent = QLearningAgent(env=env, seed=run_seed, **q_config)

    trainer = Trainer(env=env, agent=agent, config=q_config)
    logger = trainer.train(
        n_episodes=episodes,
        verbose=VERBOSE_TRAINING,
        verbose_interval=VERBOSE_INTERVAL,
    )

    metrics_csv = logger.export_csv(run_dir / "metrics.csv")

    if EXPORT_MAZE_SNAPSHOT:
        trainer.render_maze(
            title=f"Maze Snapshot | {difficulty.title()} | Run {run_number}",
            save_path=str(run_dir / "maze_snapshot.png"),
        )

    if EXPORT_TRAINING_CURVES:
        trainer.plot_training(
            moving_average_window=MOVING_AVERAGE_WINDOW,
            save_path=str(run_dir / "training_curves.png"),
        )

    if EXPORT_POLICY_HEATMAP:
        trainer.plot_q_policy_heatmap(
            annotate_policy=True,
            save_path=str(run_dir / "q_policy_heatmap.png"),
        )

    if not SHOW_PLOTS_INTERACTIVELY:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass

    final_reward = float(logger.rewards[-1]) if logger.rewards else 0.0
    avg_reward = sum(logger.rewards) / len(logger.rewards) if logger.rewards else 0.0
    avg_steps = sum(logger.steps) / len(logger.steps) if logger.steps else 0.0
    success_rate_all = logger.success_rate(last_n=max(1, len(logger.success))) if logger.success else 0.0
    success_rate_last_100 = logger.success_rate(last_n=100)

    row: Dict[str, float | int | str | bool] = {
        "difficulty": difficulty,
        "run_number": run_number,
        "run_seed": -1 if run_seed is None else run_seed,
        "episodes": episodes,
        "final_reward": final_reward,
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "success_rate_all": float(success_rate_all),
        "success_rate_last_100": float(success_rate_last_100),
        "metrics_csv": str(metrics_csv),
        "run_dir": str(run_dir),
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(row, file_handle, indent=2)

    return row


def _write_summary(experiment_dir: Path, rows: List[Dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return

    fieldnames = [
        "difficulty",
        "run_number",
        "run_seed",
        "episodes",
        "final_reward",
        "avg_reward",
        "avg_steps",
        "success_rate_all",
        "success_rate_last_100",
        "metrics_csv",
        "run_dir",
    ]

    summary_csv = experiment_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file_handle:
        writer = DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(rows, file_handle, indent=2)


def run_experiments() -> Path:
    experiment_dir = _create_experiment_dir()
    _write_experiment_config(experiment_dir)

    all_rows: List[Dict[str, float | int | str | bool]] = []
    run_index = 0

    print("=" * 80)
    print(f"Starting experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 80)

    for difficulty in DIFFICULTIES:
        print(f"\nDifficulty: {difficulty}")
        for run_number in range(1, RUNS_PER_DIFFICULTY + 1):
            run_index += 1
            print(f"  - Run {run_number}/{RUNS_PER_DIFFICULTY}")
            row = _run_single(
                experiment_dir=experiment_dir,
                difficulty=difficulty,
                run_number=run_number,
                run_index=run_index,
            )
            all_rows.append(row)

    _write_summary(experiment_dir, all_rows)

    print("\n" + "=" * 80)
    print("Experiment finished")
    print(f"Summary CSV: {experiment_dir / 'summary.csv'}")
    print("=" * 80)

    return experiment_dir


if __name__ == "__main__":
    run_experiments()

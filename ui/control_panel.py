"""Dark-mode training UI for the RL maze solver.

This interface is intentionally Q-learning focused. Policy-gradient controls
are omitted so the app stays aligned with the current codebase.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from agents import QLearningAgent
from envs import MazeEnv
from training import Trainer, get_q_learning_config
from visualization import plot_q_policy_heatmap, render_maze_snapshot


BG = "#0f141c"
PANEL = "#151c26"
CARD = "#182230"
CARD_ALT = "#111827"
TEXT = "#e5eef8"
MUTED = "#97a6ba"
ACCENT = "#16b8c5"
ACCENT_2 = "#f59e0b"
GOOD = "#22c55e"
BAD = "#ef4444"
BORDER = "#243142"
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 10)


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if window <= 0 or not values:
        return np.array([])
    if len(values) < window:
        window = len(values)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


class ControlPanel:
    """Tkinter dashboard for training and visualizing Q-learning runs."""

    def __init__(self, root: Optional[tk.Tk] = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("RL Maze Solver | Dark Mode")
        self.root.geometry("1520x920")
        self.root.minsize(1280, 820)
        self.root.configure(bg=BG)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.difficulty_var = tk.StringVar(value="medium")
        self.episodes_var = tk.IntVar(value=250)
        self.window_var = tk.IntVar(value=50)
        self.seed_var = tk.StringVar(value="42")
        self.goal_row_var = tk.StringVar(value="")
        self.goal_col_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready to train")
        self.run_dir_var = tk.StringVar(value="No export yet")
        self.pause_label_var = tk.StringVar(value="Pause")

        self.trainer: Optional[Trainer] = None
        self.agent: Optional[QLearningAgent] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.is_training = False
        self.current_difficulty: str = self.difficulty_var.get()
        self.current_seed: Optional[int] = None
        self.config_widgets: list[tk.Widget] = []

        self._build_style()
        self._build_layout()
        self._bind_events()
        self._initialize_session(self.difficulty_var.get())

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("AltCard.TFrame", background=CARD_ALT, relief="flat")
        style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_BODY)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=FONT_SUBTITLE)
        style.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("CardText.TLabel", background=CARD, foreground=TEXT, font=FONT_BODY)
        style.configure("Accent.TLabel", background=CARD, foreground=ACCENT, font=FONT_BODY)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=FONT_BODY)
        style.configure("TButton", font=FONT_BODY, padding=(12, 8))
        style.map("TButton", foreground=[("active", TEXT)], background=[("active", BORDER)])

        style.configure(
            "Dark.TCombobox",
            fieldbackground=CARD_ALT,
            background=CARD_ALT,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=6,
        )

        style.configure(
            "Dark.TEntry",
            fieldbackground=CARD_ALT,
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=6,
        )

        style.configure(
            "Dark.Horizontal.TScale",
            background=CARD,
            troughcolor=BORDER,
        )

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=BG, padx=20, pady=16)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = tk.Label(header, text="RL Maze Solver", bg=BG, fg=TEXT, font=FONT_TITLE)
        title.grid(row=0, column=0, sticky="w")

        subtitle = tk.Label(
            header,
            text="Q-learning training dashboard with live maze rendering and exported run artifacts.",
            bg=BG,
            fg=MUTED,
            font=FONT_SUBTITLE,
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.status_chip = tk.Label(
            header,
            textvariable=self.status_var,
            bg=BORDER,
            fg=TEXT,
            font=FONT_BODY,
            padx=12,
            pady=6,
        )
        self.status_chip.grid(row=0, column=1, rowspan=2, sticky="e")

        body = tk.Frame(self.root, bg=BG, padx=20, pady=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=4)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_rowconfigure(0, weight=3)
        left.grid_rowconfigure(1, weight=2)
        left.grid_columnconfigure(0, weight=1)

        maze_card = self._card(left)
        maze_card.grid(row=0, column=0, sticky="nsew", pady=(0, 16))
        maze_card.grid_rowconfigure(1, weight=1)
        maze_card.grid_columnconfigure(0, weight=1)

        tk.Label(maze_card, text="Maze View", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 8)
        )

        self.maze_fig, self.maze_ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
        self.maze_fig.patch.set_facecolor(CARD)
        self.maze_ax.set_facecolor(CARD_ALT)
        self.maze_canvas = FigureCanvasTkAgg(self.maze_fig, master=maze_card)
        self.maze_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        progress_card = self._card(left)
        progress_card.grid(row=1, column=0, sticky="nsew")
        progress_card.grid_rowconfigure(1, weight=1)
        progress_card.grid_columnconfigure(0, weight=1)

        tk.Label(progress_card, text="Training Progress", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 8)
        )

        self.progress_fig, self.progress_axes = plt.subplots(3, 1, figsize=(6.4, 4.8), dpi=100, sharex=True)
        self.progress_fig.patch.set_facecolor(CARD)
        for axis in self.progress_axes:
            axis.set_facecolor(CARD_ALT)
        self.progress_canvas = FigureCanvasTkAgg(self.progress_fig, master=progress_card)
        self.progress_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(4, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.controls_card = self._card(right)
        self.controls_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        self._build_controls(self.controls_card)

        stats_card = self._card(right)
        stats_card.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        self._build_stats(stats_card)

        export_card = self._card(right)
        export_card.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        self._build_export_section(export_card)

        info_card = self._card(right)
        info_card.grid(row=3, column=0, sticky="nsew")
        self._build_info_section(info_card)

        footer = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(footer, textvariable=self.run_dir_var, bg=BG, fg=MUTED, font=FONT_MONO).grid(row=0, column=0, sticky="w")

    def _card(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        return frame

    def _build_controls(self, parent: tk.Frame) -> None:
        parent.grid_columnconfigure(1, weight=1)

        tk.Label(parent, text="Control Center", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 10)
        )

        self._labeled_combo(parent, "Difficulty", self.difficulty_var, ["easy", "medium", "hard"], 1)
        self._labeled_spinbox(parent, "Episodes", self.episodes_var, 1, 10000, 2)
        self._labeled_spinbox(parent, "Window", self.window_var, 5, 500, 3)
        self._labeled_entry(parent, "Seed", self.seed_var, 4)

        goal_label = tk.Label(parent, text="Goal cell", bg=CARD, fg=MUTED, font=FONT_BODY)
        goal_label.grid(row=5, column=0, sticky="w", padx=16, pady=(12, 4))
        goal_row = tk.Frame(parent, bg=CARD)
        goal_row.grid(row=5, column=1, sticky="ew", padx=16, pady=(12, 4))
        goal_row.grid_columnconfigure(0, weight=1)
        goal_row.grid_columnconfigure(1, weight=1)
        self._mini_entry(goal_row, self.goal_row_var, 0, "row")
        self._mini_entry(goal_row, self.goal_col_var, 1, "col")

        button_row = tk.Frame(parent, bg=CARD)
        button_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 12))
        button_row.grid_columnconfigure((0, 1), weight=1)

        self.start_button = tk.Button(
            button_row,
            text="Start Training",
            command=self.start_training,
            bg=ACCENT,
            fg="#041014",
            activebackground="#25d2de",
            activeforeground="#041014",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.pause_button = tk.Button(
            button_row,
            textvariable=self.pause_label_var,
            command=self.toggle_pause,
            bg=ACCENT_2,
            fg="#140d01",
            activebackground="#ffb84d",
            activeforeground="#140d01",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        action_row = tk.Frame(parent, bg=CARD)
        action_row.grid(row=7, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 14))
        action_row.grid_columnconfigure((0, 1), weight=1)

        self.reset_button = tk.Button(
            action_row,
            text="Reset Maze",
            command=self.reset_session,
            bg=BORDER,
            fg=TEXT,
            activebackground="#314255",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            font=("Segoe UI", 10),
            padx=12,
            pady=8,
        )
        self.reset_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.set_goal_button = tk.Button(
            action_row,
            text="Set Goal",
            command=self.apply_goal_from_inputs,
            bg=BORDER,
            fg=TEXT,
            activebackground="#314255",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            font=("Segoe UI", 10),
            padx=12,
            pady=8,
        )
        self.set_goal_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _build_stats(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Training Stats", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 10)
        )

        self.stat_episode = self._stat_row(parent, 1, "Episode", "—")
        self.stat_reward = self._stat_row(parent, 2, "Reward", "—")
        self.stat_steps = self._stat_row(parent, 3, "Steps", "—")
        self.stat_success = self._stat_row(parent, 4, "Success", "—")
        self.stat_epsilon = self._stat_row(parent, 5, "Epsilon", "—")
        self.stat_goal = self._stat_row(parent, 6, "Goal", "—")

    def _build_export_section(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Export", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 10)
        )

        export_button = tk.Button(
            parent,
            text="Save current run",
            command=self.export_current_run,
            bg=GOOD,
            fg="#06140b",
            activebackground="#38d17a",
            activeforeground="#06140b",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        export_button.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        tk.Label(
            parent,
            text="Exports maze snapshot, progress plot, Q-policy heatmap, metrics CSV, and JSON summary.",
            bg=CARD,
            fg=MUTED,
            wraplength=320,
            justify="left",
            font=FONT_BODY,
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 14))

    def _build_info_section(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Notes", bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 10)
        )

        notes = (
            "• Click a free cell in the maze to move the goal.\n"
            "• Training runs one episode at a time so Pause/Resume stays responsive.\n"
            "• Q-learning only. Policy-gradient controls are intentionally hidden.\n"
            "• Use Reset to rebuild the maze for the selected difficulty."
        )
        tk.Label(parent, text=notes, bg=CARD, fg=MUTED, justify="left", anchor="nw", font=FONT_BODY).grid(
            row=1, column=0, sticky="nw", padx=16, pady=(0, 14)
        )

    def _labeled_combo(self, parent: tk.Frame, label: str, variable: tk.StringVar, values: list[str], row: int) -> None:
        tk.Label(parent, text=label, bg=CARD, fg=MUTED, font=FONT_BODY).grid(row=row, column=0, sticky="w", padx=16, pady=6)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", style="Dark.TCombobox")
        combo.grid(row=row, column=1, sticky="ew", padx=16, pady=6)
        self.config_widgets.append(combo)
        if label.lower() == "difficulty":
            self.difficulty_combo = combo

    def _labeled_spinbox(self, parent: tk.Frame, label: str, variable: tk.IntVar, minimum: int, maximum: int, row: int) -> None:
        tk.Label(parent, text=label, bg=CARD, fg=MUTED, font=FONT_BODY).grid(row=row, column=0, sticky="w", padx=16, pady=6)
        spin = ttk.Spinbox(parent, from_=minimum, to=maximum, textvariable=variable, style="Dark.TEntry")
        spin.grid(row=row, column=1, sticky="ew", padx=16, pady=6)
        self.config_widgets.append(spin)

    def _labeled_entry(self, parent: tk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        tk.Label(parent, text=label, bg=CARD, fg=MUTED, font=FONT_BODY).grid(row=row, column=0, sticky="w", padx=16, pady=6)
        entry = ttk.Entry(parent, textvariable=variable, style="Dark.TEntry")
        entry.grid(row=row, column=1, sticky="ew", padx=16, pady=6)
        self.config_widgets.append(entry)

    def _mini_entry(self, parent: tk.Frame, variable: tk.StringVar, column: int, placeholder: str) -> None:
        entry = ttk.Entry(parent, textvariable=variable, style="Dark.TEntry", width=10)
        entry.grid(row=0, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0))
        entry.insert(0, placeholder)
        entry.bind("<FocusIn>", lambda event, target=entry, text=placeholder: self._clear_placeholder(event, target, text))

    def _clear_placeholder(self, event, widget: ttk.Entry, placeholder: str) -> None:
        if widget.get() == placeholder:
            widget.delete(0, tk.END)

    def _stat_row(self, parent: tk.Frame, row: int, label: str, value: str) -> tk.Label:
        container = tk.Frame(parent, bg=CARD)
        container.grid(row=row, column=0, sticky="ew", padx=16, pady=4)
        container.grid_columnconfigure(1, weight=1)
        tk.Label(container, text=label, bg=CARD, fg=MUTED, font=FONT_BODY).grid(row=0, column=0, sticky="w")
        stat = tk.Label(container, text=value, bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold"), anchor="e")
        stat.grid(row=0, column=1, sticky="e")
        return stat

    def _bind_events(self) -> None:
        self.root.bind("<Escape>", lambda event: self.stop_training())

    def _initialize_session(self, difficulty: str) -> None:
        self.stop_training(wait=False)

        seed_value = self._parse_seed()
        episodes = max(1, int(self.episodes_var.get()))

        self.env = MazeEnv(difficulty=difficulty, seed=seed_value)
        q_config = get_q_learning_config()
        self.agent = QLearningAgent(env=self.env, seed=seed_value, **q_config)
        self.trainer = Trainer(env=self.env, agent=self.agent, config=q_config)
        self.trainer.logger.reset()
        self.trainer.last_episode_path = [self.env.start_pos]

        self._sync_goal_inputs()
        self.status_var.set(f"Loaded {difficulty.title()} maze | {episodes} planned episodes")
        self._refresh_dashboard()

    def _parse_seed(self) -> Optional[int]:
        raw = self.seed_var.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            messagebox.showwarning("Seed", "Seed must be an integer. Using no fixed seed instead.")
            return None

    def _sync_goal_inputs(self) -> None:
        if not hasattr(self, "env"):
            return
        self.current_difficulty = self.env.difficulty
        self.current_seed = self._parse_seed()
        self.goal_row_var.set(str(self.env.goal_pos[0]))
        self.goal_col_var.set(str(self.env.goal_pos[1]))
        self.stat_goal.config(text=f"{self.env.goal_pos}")

    def _on_difficulty_changed(self, event=None) -> None:
        if self.is_training:
            return
        self._initialize_session(self.difficulty_var.get())

    def apply_goal_from_inputs(self) -> None:
        if not hasattr(self, "env"):
            return
        try:
            row = int(self.goal_row_var.get().strip())
            col = int(self.goal_col_var.get().strip())
        except ValueError:
            messagebox.showerror("Goal", "Goal row and col must be integers.")
            return

        try:
            self.env.set_goal((row, col))
        except ValueError as exc:
            messagebox.showerror("Goal", str(exc))
            return

        self.status_var.set(f"Goal moved to ({row}, {col})")
        self._refresh_dashboard()

    def start_training(self) -> None:
        if self.is_training:
            return

        selected_seed = self._parse_seed()
        if (
            self.trainer is None
            or self.agent is None
            or self.current_difficulty != self.difficulty_var.get()
            or self.current_seed != selected_seed
        ):
            self._initialize_session(self.difficulty_var.get())

        self.stop_event.clear()
        self.pause_event.clear()
        self.pause_label_var.set("Pause")
        self.is_training = True
        self._set_controls_enabled(False)
        self.status_var.set("Training running...")

        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()

    def toggle_pause(self) -> None:
        if not self.is_training:
            return

        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_label_var.set("Pause")
            self.status_var.set("Training resumed")
        else:
            self.pause_event.set()
            self.pause_label_var.set("Resume")
            self.status_var.set("Training paused")

    def stop_training(self, wait: bool = True) -> None:
        self.stop_event.set()
        if wait and self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=0.2)

    def reset_session(self) -> None:
        if self.is_training:
            self.status_var.set("Stopping current run before reset...")
            self.stop_event.set()
            self.root.after(100, self._finish_reset_when_idle)
            return

        self._initialize_session(self.difficulty_var.get())
        self.status_var.set("Session reset")

    def _finish_reset_when_idle(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            self.root.after(100, self._finish_reset_when_idle)
            return

        self.is_training = False
        self._set_controls_enabled(True)
        self._initialize_session(self.difficulty_var.get())
        self.status_var.set("Session reset")

    def _training_worker(self) -> None:
        assert self.trainer is not None and self.agent is not None

        total_episodes = max(1, int(self.episodes_var.get()))
        logger = self.trainer.logger
        logger.reset()
        self.trainer.last_episode_path = [self.env.start_pos]

        for episode in range(total_episodes):
            if self.stop_event.is_set():
                break

            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)

            if self.stop_event.is_set():
                break

            reward, steps, success = self.trainer.run_episode()
            logger.log(episode=episode, reward=reward, steps=steps, success=success)
            self.agent.on_episode_end()
            self.trainer.current_episode = episode

            self.root.after(
                0,
                lambda episode=episode, reward=reward, steps=steps, success=success: self._on_episode_complete(
                    episode, reward, steps, success
                ),
            )

        self.root.after(0, self._on_training_finished)

    def _on_episode_complete(self, episode: int, reward: float, steps: int, success: bool) -> None:
        if self.trainer is None or self.agent is None:
            return

        self.status_var.set(
            f"Episode {episode + 1} | reward {reward:.2f} | steps {steps} | success {'yes' if success else 'no'}"
        )
        self._refresh_dashboard()

    def _on_training_finished(self) -> None:
        self.is_training = False
        self.pause_event.clear()
        self.pause_label_var.set("Pause")
        self._set_controls_enabled(True)

        if self.stop_event.is_set():
            self.status_var.set("Training stopped")
        else:
            self.status_var.set("Training complete")

        self._refresh_dashboard()

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in [self.start_button, self.reset_button, self.set_goal_button, *self.config_widgets]:
            widget.config(state=state)
        self.pause_button.config(state="normal")
        self.pause_label_var.set("Pause")

    def _refresh_dashboard(self) -> None:
        self._refresh_maze_plot()
        self._refresh_progress_plot()
        self._refresh_stats()

    def _refresh_maze_plot(self) -> None:
        if self.trainer is None:
            return

        render_maze_snapshot(
            grid=self.env.grid,
            agent_pos=self.env.agent_pos,
            goal_pos=self.env.goal_pos,
            start_pos=self.env.start_pos,
            path=self.trainer.last_episode_path,
            title=f"{self.env.difficulty.title()} Maze | goal {self.env.goal_pos}",
            ax=self.maze_ax,
            figure=self.maze_fig,
        )
        self.maze_canvas.draw_idle()

    def _refresh_progress_plot(self) -> None:
        if self.trainer is None:
            return

        logger = self.trainer.logger
        window = max(1, int(self.window_var.get()))
        # Snapshot logger lists to avoid race conditions where another thread
        # appends to the lists between computing x and y arrays.
        rewards_list = list(logger.rewards)
        steps_list = list(logger.steps)
        success_list = [1.0 if v else 0.0 for v in list(logger.success)]

        episodes = np.arange(1, len(rewards_list) + 1)
        rewards = np.asarray(rewards_list, dtype=float)
        steps = np.asarray(steps_list, dtype=float)
        success = np.asarray(success_list, dtype=float)

        for axis in self.progress_axes:
            axis.clear()
            axis.set_facecolor(CARD_ALT)
            axis.grid(alpha=0.15, color="#516176")

        if len(rewards):
            self.progress_axes[0].plot(episodes, rewards, color=ACCENT, alpha=0.25, linewidth=1.0)
            reward_ma = _moving_average(logger.rewards, window)
            if len(reward_ma):
                x = np.arange(window, window + len(reward_ma))
                self.progress_axes[0].plot(x, reward_ma, color=ACCENT, linewidth=2.0)
        self.progress_axes[0].set_ylabel("Reward", color=TEXT)

        if len(steps):
            self.progress_axes[1].plot(episodes, steps, color=ACCENT_2, alpha=0.25, linewidth=1.0)
            steps_ma = _moving_average(logger.steps, window)
            if len(steps_ma):
                x = np.arange(window, window + len(steps_ma))
                self.progress_axes[1].plot(x, steps_ma, color=ACCENT_2, linewidth=2.0)
        self.progress_axes[1].set_ylabel("Steps", color=TEXT)

        if len(success):
            self.progress_axes[2].plot(episodes, success, color=GOOD, alpha=0.2, linewidth=1.0)
            success_ma = _moving_average([float(value) for value in logger.success], window)
            if len(success_ma):
                x = np.arange(window, window + len(success_ma))
                self.progress_axes[2].plot(x, success_ma, color=GOOD, linewidth=2.0)
        self.progress_axes[2].set_ylabel("Success", color=TEXT)
        self.progress_axes[2].set_xlabel("Episode", color=TEXT)

        for axis in self.progress_axes:
            axis.tick_params(colors=TEXT)
            for spine in axis.spines.values():
                spine.set_color(BORDER)

        self.progress_fig.tight_layout()
        self.progress_canvas.draw_idle()

    def _refresh_stats(self) -> None:
        if self.trainer is None:
            return

        logger = self.trainer.logger
        if logger.rewards:
            episode = len(logger.rewards)
            reward = logger.rewards[-1]
            steps = logger.steps[-1]
            success_rate = logger.success_rate(last_n=min(100, len(logger.success)))
            epsilon = getattr(self.agent, "epsilon", 0.0) if self.agent else 0.0
            self.stat_episode.config(text=str(episode))
            self.stat_reward.config(text=f"{reward:.2f}")
            self.stat_steps.config(text=str(steps))
            self.stat_success.config(text=f"{success_rate:.1%}")
            self.stat_epsilon.config(text=f"{epsilon:.3f}")
        else:
            self.stat_episode.config(text="—")
            self.stat_reward.config(text="—")
            self.stat_steps.config(text="—")
            self.stat_success.config(text="—")
            self.stat_epsilon.config(text="—")

        self.stat_goal.config(text=str(self.env.goal_pos))

    def export_current_run(self) -> None:
        if self.trainer is None:
            messagebox.showinfo("Export", "No run has been created yet.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("experiments") / f"ui_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.trainer.logger.export_csv(run_dir / "metrics.csv")
        render_maze_snapshot(
            grid=self.env.grid,
            agent_pos=self.env.agent_pos,
            goal_pos=self.env.goal_pos,
            start_pos=self.env.start_pos,
            path=self.trainer.last_episode_path,
            title=f"{self.env.difficulty.title()} Maze",
            save_path=str(run_dir / "maze_snapshot.png"),
        )
        self.trainer.plot_training(
            moving_average_window=max(1, int(self.window_var.get())),
            save_path=str(run_dir / "training_curves.png"),
        )
        self.trainer.plot_q_policy_heatmap(save_path=str(run_dir / "q_policy_heatmap.png"))

        summary = {
            "difficulty": self.env.difficulty,
            "goal": self.env.goal_pos,
            "episodes": len(self.trainer.logger.rewards),
            "final_reward": self.trainer.logger.rewards[-1] if self.trainer.logger.rewards else None,
            "success_rate": self.trainer.logger.success_rate(last_n=min(100, len(self.trainer.logger.success)))
            if self.trainer.logger.success
            else 0.0,
            "epsilon": getattr(self.agent, "epsilon", None),
        }
        with (run_dir / "run_summary.json").open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2)

        self.run_dir_var.set(f"Exported to {run_dir}")
        self.status_var.set("Current run exported")
        messagebox.showinfo("Export", f"Saved run artifacts to\n{run_dir}")

    def _on_maze_click(self, event) -> None:
        if self.trainer is None or event.inaxes != self.maze_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.env.height and 0 <= col < self.env.width):
            return

        if self.env.grid[row, col] == 1:
            self.status_var.set(f"Cell ({row}, {col}) is a wall")
            return

        self.env.set_goal((row, col))
        self.goal_row_var.set(str(row))
        self.goal_col_var.set(str(col))
        self.status_var.set(f"Goal moved to ({row}, {col})")
        self._refresh_dashboard()

    def _on_close(self) -> None:
        """Handle window close event to properly clean up threads."""
        self.stop_event.set()
        self.pause_event.clear()

        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=0.5)

        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        self.maze_canvas.mpl_connect("button_press_event", self._on_maze_click)
        if hasattr(self, "difficulty_combo"):
            self.difficulty_combo.bind("<<ComboboxSelected>>", self._on_difficulty_changed)
        self.root.mainloop()


def launch_app() -> None:
    root = tk.Tk()
    app = ControlPanel(root)
    app.run()

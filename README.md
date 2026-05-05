# RL Maze Solver: Q-Learning in Gymnasium

A professional-grade **Reinforcement Learning** project implementing tabular Q-Learning to solve procedurally generated mazes of varying difficulty. The agent learns an optimal policy through trial-and-error exploration, with interactive visualization and experiment management.

---

## 🎯 Features

- **Q-Learning Agent**: Tabular Q-Learning with epsilon-greedy exploration and decay scheduling
- **Custom Gymnasium Environment**: Discrete state/action spaces, DFS-generated mazes, shaped reward signal
- **Three Difficulty Levels**: Easy (7×7), Medium (11×11), Hard (15×15) mazes with configurable goals
- **Interactive Training Dashboard**: Dark-mode Tkinter UI for real-time monitoring, pause/resume, and export
- **Comprehensive Visualization**:
  - Maze rendering with agent, goal, start position, and visited path overlays
  - Training curves (rewards, steps, success rate) with moving averages
  - Q-value heatmaps with best-action overlays (directional arrows)
- **Jupyter Notebooks**: Interactive demos for environment exploration and Q-Learning walkthrough
- **Reproducible Results**: Seed-based experiment control and detailed metrics export

---

## 📁 Project Architecture

### Core Modules

#### **`envs/` — Environment**
Gymnasium-compatible custom environment implementing the MDP (Markov Decision Process) for maze navigation.

- **`maze_env.py`**: Main `MazeEnv` class
  - Discrete observation space: States encoded as `row * width + col` (integer)
  - Discrete action space: 4 actions (UP=0, DOWN=1, LEFT=2, RIGHT=3)
  - Reward shaping: +10 for goal, -1 for timeout, -0.5 for walls, -0.01 per step
  - Methods: `reset()`, `step()`, `render()`

- **`maze_configs.py`**: Difficulty presets
  - **Easy**: 7×7 grid, goal at (5,5), max_steps=98
  - **Medium**: 11×11 grid, goal at (9,9), max_steps=242
  - **Hard**: 15×15 grid, goal at (13,13), max_steps=450

- **`maze_generator.py`**: DFS-based maze generation with procedural wall placement

#### **`agents/` — Learning Agent**
Abstract base class + Q-Learning implementation.

- **`base_agent.py`**: Abstract `BaseAgent` class defining interface
  - Methods: `select_action(state)`, `update(state, action, reward, next_state, done)`, `on_episode_end()`

- **`q_learning_agent.py`**: `QLearningAgent` implementation
  - **Q-Table**: Numpy array `(n_states, n_actions)` storing state-action values
  - **Learning Rule**: Bellman update: `Q[s,a] = Q[s,a] + α * (r + γ * max(Q[s',·]) - Q[s,a])`
  - **Exploration**: Epsilon-greedy policy (random actions with probability ε)
  - **Hyperparameters**:
    - `alpha=0.1` (learning rate)
    - `gamma=0.99` (discount factor for future rewards)
    - `epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995` (decay schedule)

#### **`training/` — Training Orchestration**
Episode management, logging, and visualization helpers.

- **`trainer.py`**: Main `Trainer` class
  - Orchestrates episodes: environment reset → agent action selection → environment step → agent update
  - Maintains episode metrics (reward, steps, success flag)
  - Tracks agent's last episode path for visualization
  - Export methods: `render_maze()`, `plot_training()`, `plot_q_policy_heatmap()`

- **`hyperparams.py`**: Centralized configuration
  - Q-Learning defaults (alpha, gamma, epsilon schedule)
  - Per-difficulty episode budgets (500 easy, 1000 medium, 2000 hard)
  - Helper functions: `get_q_learning_config()`, `get_episode_config(difficulty)`

- **`logger.py`**: `Logger` class for metric collection
  - Tracks: episodes, rewards, steps, success flags
  - Provides moving average and CSV export: `export_csv(path)`

#### **`visualization/` — Plotting & Rendering**
Matplotlib-based visualization functions (integrated into Trainer).

- **`maze_renderer.py`**: Static maze visualization
  - Function: `render_maze_snapshot(grid, agent_pos, goal_pos, start_pos, path, ...)`
  - Overlays:
    - Walls (black), free cells (white)
    - Start position (yellow), agent (red), goal (green)
    - Visited path (light blue)

- **`training_plots.py`**: Training metric plots
  - Function: `plot_training_curves(logger, moving_average_window, ...)`
  - 3-subplot layout: cumulative reward, episode steps, success rate
  - Moving average smoothing for curve clarity

- **`policy_heatmap.py`**: Q-value visualization
  - Function: `plot_q_policy_heatmap(env, agent, ...)`
  - Heatmap of max Q-values per state (viridis colormap)
  - Directional action arrows (U/D/L/R) showing optimal policy

#### **`ui/` — Interactive Dashboard**
Tkinter-based GUI for live training control.

- **`control_panel.py`**: `ControlPanel` class
  - **Layout**: 4:1 left/right split
    - Left: Live maze view (top), training progress plots (bottom)
    - Right: Controls, stats, export, notes
  - **Controls**:
    - Difficulty dropdown, episodes spinbox, moving average window
    - Seed input, custom goal row/col
    - Start/Pause/Resume/Reset/Export buttons
  - **Features**:
    - Real-time plot updates on episode completion
    - Thread-safe training with pause/resume via `threading.Event()`
    - Automatic cleanup on window close (prevents terminal hang)
    - Export to: `experiments/ui_run_{timestamp}/`

#### **`notebook/` — Interactive Documentation**
Jupyter notebooks for exploration and learning.

- **`01_environment_demo.ipynb`**: Environment walkthrough
  - Reset, step through actions, print observations
  - Renders maze after manual steps

- **`02_q_learning_walkthrough.ipynb`**: Q-Learning mini-tutorial
  - Train agent for 50 episodes
  - Plot training curves and policy heatmap


#### **`main.py`** — Desktop UI Entry Point
Launches the interactive Tkinter dashboard.

```bash
python main.py
```

---

## 🚀 Installation & Setup

### Prerequisites
- [**Python 3.11+**](https://www.python.org/downloads/)
- **VS Code** (recommended for notebooks and debugging)

### Step 1: Clone or Navigate to Project
```bash
cd <your_project_directory>
```

### Step 2: Create Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

*Note*: If you get a PowerShell execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies**:
- `gymnasium >= 0.29.0` — RL environment framework
- `numpy >= 1.24.0` — Numerical computing (Q-tables, grids)
- `matplotlib >= 3.7.0` — Plotting and visualization
- `jupyter >= 1.0.0` — Jupyter notebooks (optional but recommended)
- `ipykernel >= 6.0.0` — IPython kernel for VS Code notebooks (optional)

### Step 5: Verify Installation
```bash
python -c "import gymnasium, numpy, matplotlib; print('✓ All dependencies installed')"
```

---

## 🎮 How to Run the Project

### Option 1: Interactive Tkinter Dashboard (Recommended)
```bash
python main.py
```

**Features**:
- Select difficulty (easy/medium/hard)
- Configure episode count and moving average window
- Set custom goal position
- Real-time maze and training plot visualization
- Pause/resume training mid-run
- Export results (PNG plots, CSV metrics, JSON summary)

### Option 2: Jupyter Notebooks
```bash
jupyter notebook
```

**Navigate to `notebook/`**:
- **`01_environment_demo.ipynb`**: Explore MazeEnv interactively
  - Create environment, reset, take manual steps
  - Render maze at any point
  
- **`02_q_learning_walkthrough.ipynb`**: Short training run
  - Train agent for 50 episodes
  - Plot training curves and policy heatmap


---

## ⚙️ Configuration Guide

### Hyperparameters (in `training/hyperparams.py`)

#### Q-Learning Parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| **alpha** | 0.1 | Learning rate — controls how much Q-values update per experience |
| **gamma** | 0.99 | Discount factor — how much future rewards matter vs immediate rewards |
| **epsilon_start** | 1.0 | Initial exploration probability (100% random actions) |
| **epsilon_end** | 0.01 | Final exploration probability (1% random, 99% greedy) |
| **epsilon_decay** | 0.995 | Multiplicative decay: `ε := ε * decay` each episode |

#### Episode Budgets (per difficulty)
| Difficulty | Episodes | Max Steps | Grid Size | Goal |
|-----------|----------|-----------|-----------|------|
| **Easy** | 500 | 98 | 7×7 | (5, 5) |
| **Medium** | 1000 | 242 | 11×11 | (9, 9) |
| **Hard** | 2000 | 450 | 15×15 | (13, 13) |

---

## 📊 Visualization Outputs

### 1. Maze Snapshot
**Shows**: Current agent position, goal, start position, and visited path overlaid on maze.

- **Walls**: Black
- **Free cells**: White
- **Start position**: Yellow
- **Agent**: Red
- **Goal**: Green
- **Visited path**: Light blue

### 2. Training Curves
**3-subplot layout**:
- **Top**: Cumulative episode reward (with moving average)
- **Middle**: Steps per episode (with moving average)
- **Bottom**: Success rate rolling average

*Moving average smooths noise for clarity; window size configurable (default: 50).*

### 3. Q-Policy Heatmap
**Shows**: Max Q-value for each free cell, colored by value intensity (viridis).

- **Values**: Darker = lower Q-value, brighter = higher Q-value
- **Directional arrows**: U/D/L/R indicate best action per state
- **Walls**: Marked with '#'
- **Start**: Marked 'S'
- **Goal**: Marked 'G'

---

## 🧠 Reinforcement Learning Concepts

### What is Q-Learning?
**Q-Learning** is a **model-free** RL algorithm that learns a **Q-table** — a matrix mapping each (state, action) pair to its expected cumulative reward.

### The Core Update Rule (Bellman Equation)
```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') - Q(s, a)]

```

**Terms**:
- `α` (alpha): How much to trust the new experience
- `r`: Immediate reward from the environment
- `γ` (gamma): How much to value future rewards (0=myopic, 1=farsighted)
- `max Q(s', a')`: Best predicted value of the next state

### Exploration vs. Exploitation
- **Exploration**: Agent takes random actions to discover new paths (early training)
- **Exploitation**: Agent follows learned policy greedily (later training)

**Epsilon-Greedy** balances both:
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q[state, :])  # Exploit
```

With **epsilon decay**, exploration gradually decreases: `ε := ε * decay` each episode.

### Episodes & Convergence
- **Episode**: One complete trajectory from start to goal (or timeout)
- **Convergence**: Q-values stabilize; agent's policy stops improving
- Harder mazes need more episodes to discover optimal paths

---

## 📚 References

- **Gymnasium Docs**: https://gymnasium.farama.org/
---

## 📝 License & Attribution

This project is provided as-is for educational and research purposes.

**Questions?** Refer to the code comments and docstrings, or check the Jupyter notebooks for interactive walkthroughs.

---

**Happy Maze Solving! 🐭🌾**
# 🧠 Reinforcement Learning Maze Solver — Complete Project Plan

> **A 1–2 Week Deep-Dive Roadmap for Building and Understanding RL from Scratch**
>
> *Designed for learners who want to both build a working project and deeply understand Reinforcement Learning.*

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Architecture](#2-project-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Step-by-Step Development Plan (7 Phases)](#4-step-by-step-development-plan)
5. [Maze Environment Design](#5-maze-environment-design)
6. [Algorithms Explained — Teach Me](#6-algorithms-explained--teach-me)
7. [Training Pipeline](#7-training-pipeline)
8. [Visualization System](#8-visualization-system)
9. [UI Design](#9-ui-design)
10. [Algorithm Comparison](#10-algorithm-comparison)
11. [Extensions & Advanced Topics](#11-extensions--advanced-topics)
12. [Learning Notes — Core RL Concepts](#12-learning-notes--core-rl-concepts)
13. [Full Dependency List](#13-full-dependency-list)
14. [Quick-Start Checklist](#14-quick-start-checklist)

---

## 1. Project Overview

### 🎯 What We Are Building

A **Reinforcement Learning agent** that learns — through trial and error — to navigate a maze from a starting cell to a goal cell. The agent begins with zero knowledge of the maze. It explores, gets feedback (rewards and penalties), and gradually learns the optimal path.

### 🧩 The Core Idea in Plain English

Imagine you are dropped blindfolded into a maze. You don't have a map. Every time you hit a wall, someone says "bad move." Every time you get closer to the exit, someone says "good job." Every time you reach the exit, you get a big prize. Over hundreds of attempts, you'd learn the best path. That's exactly what our RL agent does.

### 🗓️ Timeline

| Week | Focus |
|------|-------|
| Week 1, Days 1–2 | Environment + RL theory |
| Week 1, Days 3–4 | Q-Learning agent |
| Week 1, Days 5–7 | Policy Gradient agent + Visualization |
| Week 2, Days 1–3 | UI + Training pipeline |
| Week 2, Days 4–5 | Evaluation + Comparison |
| Week 2, Days 6–7 | Extensions (DQN, A*, saving models) |

---

## 2. Project Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                          │
│          (Tkinter or Matplotlib Interactive Window)              │
│   [Choose Difficulty] [Set Goal] [Start/Stop Training] [Reset]  │
└───────────────────┬──────────────────────┬──────────────────────┘
                    │                      │
                    ▼                      ▼
┌───────────────────────────┐  ┌───────────────────────────────┐
│      TRAINING MANAGER     │  │       VISUALIZATION ENGINE    │
│  - Episode loop           │  │  - Maze grid renderer         │
│  - Hyperparameter config  │  │  - Agent movement animation   │
│  - Logging & metrics      │  │  - Reward/loss curves         │
│  - Algorithm switcher     │  │  - Policy heatmap             │
└──────────┬────────────────┘  └───────────────────────────────┘
           │                                   ▲
           │                                   │ render_frame()
           ▼                                   │
┌──────────────────────────────────────────────┴────────────────┐
│                           AGENT LAYER                          │
│                                                                │
│  ┌──────────────────────┐    ┌────────────────────────────┐   │
│  │    Q-Learning Agent  │    │  Policy Gradient Agent     │   │
│  │  - Q-table           │    │  - Policy network (MLP)    │   │
│  │  - epsilon-greedy    │    │  - REINFORCE algorithm     │   │
│  │  - update rule       │    │  - Baseline subtraction    │   │
│  └──────────┬───────────┘    └───────────┬────────────────┘   │
│             │  select_action()            │  select_action()   │
│             │  update()                   │  update()          │
└─────────────┼─────────────────────────────┼────────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GYMNASIUM MAZE ENVIRONMENT                    │
│                                                                  │
│   MazeEnv(difficulty='medium', goal=(7,7))                      │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Grid-based maze (NumPy 2D array)                        │  │
│   │  0 = free cell   1 = wall                                │  │
│   │  S = start       G = goal                                │  │
│   │                                                          │  │
│   │   step(action) → (next_state, reward, done, info)        │  │
│   │   reset()      → initial_state                           │  │
│   │   render()     → visual frame                            │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Episode Start
     │
     ▼
env.reset() ──────────────────────────────► state = (row, col)
     │
     └──────────────────────────────────┐
                                        │
     ┌──────────────── Training Loop ───┘
     │
     ▼
agent.select_action(state)
     │  (uses Q-table lookup or policy network forward pass)
     ▼
env.step(action)
     │  Returns: (next_state, reward, done, truncated, info)
     ▼
agent.update(state, action, reward, next_state, done)
     │  (updates Q-table or stores trajectory)
     ▼
logger.record(reward, steps)
     │
     ▼
visualizer.render_frame(env.grid, agent.position)
     │
     ▼
 done? ──YES──► log episode ──► epsilon decay ──► new episode
     │
    NO
     │
     └──► state = next_state ──► (loop back)
```

---

## 3. Folder Structure

```
rl_maze_solver/
│
├── README.md                    # Project description and quick-start
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point — launches UI
│
├── envs/
│   ├── __init__.py
│   ├── maze_env.py              # Custom Gymnasium environment
│   ├── maze_generator.py        # Procedural maze generation (DFS/Prim)
│   └── maze_configs.py          # Easy/Medium/Hard difficulty presets
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Abstract base class for all agents
│   ├── q_learning_agent.py      # Tabular Q-Learning implementation
│   └── policy_gradient_agent.py # REINFORCE policy gradient agent
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Episode loop, logging, hyperparams
│   ├── hyperparams.py           # All hyperparameter constants
│   └── logger.py                # Metrics collection and CSV export
│
├── visualization/
│   ├── __init__.py
│   ├── maze_renderer.py         # Renders maze grid with matplotlib
│   ├── training_plots.py        # Reward curves, convergence plots
│   └── policy_heatmap.py        # Q-value and policy visualization
│
├── ui/
│   ├── __init__.py
│   └── control_panel.py         # Tkinter or matplotlib-based UI
│
├── utils/
│   ├── __init__.py
│   ├── model_io.py              # Save/load Q-tables and model weights
│   └── metrics.py               # Performance metrics calculations
│
├── extensions/
│   ├── __init__.py
│   ├── dqn_agent.py             # Deep Q-Network (bonus)
│   ├── pathfinding.py           # Dijkstra and A* for comparison
│   └── replay_buffer.py         # Experience replay for DQN
│
├── notebooks/
│   ├── 01_environment_demo.ipynb
│   ├── 02_q_learning_walkthrough.ipynb
│   └── 03_policy_gradient_walkthrough.ipynb
│
└── saved_models/
    ├── q_table_easy.pkl
    ├── q_table_medium.pkl
    └── pg_model_medium.pt
```

### Module Responsibilities at a Glance

| Module | Responsibility |
|--------|----------------|
| `envs/maze_env.py` | Defines the Gymnasium environment: step, reset, render |
| `envs/maze_generator.py` | Creates maze grids procedurally using DFS |
| `agents/q_learning_agent.py` | Maintains Q-table, selects actions, applies Bellman updates |
| `agents/policy_gradient_agent.py` | Maintains a policy network, stores trajectories, applies REINFORCE |
| `training/trainer.py` | Orchestrates episodes, calls agent + env, logs results |
| `visualization/maze_renderer.py` | Draws the maze, agent position, goal with matplotlib |
| `visualization/training_plots.py` | Plots reward curves and convergence metrics |
| `ui/control_panel.py` | Provides interactive controls to the user |
| `utils/model_io.py` | Saves/loads agents to disk |
| `extensions/pathfinding.py` | Classical solvers for benchmarking |

---

## 4. Step-by-Step Development Plan

---

### 🔵 Phase 1: Environment Design (Days 1–2)

**Goal:** Build a working, standards-compliant Gymnasium environment for the maze.

#### Tasks

1. Install dependencies: `gymnasium`, `numpy`, `matplotlib`, `torch`
2. Create `maze_generator.py` using recursive DFS (Depth-First Search) backtracking
3. Define three difficulty presets in `maze_configs.py`
4. Implement `MazeEnv` inheriting from `gymnasium.Env`
5. Implement `reset()`, `step()`, `render()`, `close()` methods
6. Test environment with random actions

#### Difficulty Presets

| Level | Grid Size | Wall Density | Goal Placement |
|-------|-----------|--------------|----------------|
| Easy | 7×7 | Low | Fixed corner |
| Medium | 11×11 | Medium | User-defined |
| Hard | 15×15 | High | User-defined |

#### Expected Output

- `env = MazeEnv(difficulty='medium', goal=(9,9))`
- `obs, info = env.reset()` → returns starting cell coordinates
- `obs, reward, done, truncated, info = env.step(2)` → moves agent right
- `env.render()` → displays maze in matplotlib window

#### Skills Learned

- How to subclass `gymnasium.Env`
- How to define `observation_space` and `action_space`
- Procedural maze generation with DFS backtracking
- Environment testing patterns

---

### 🟡 Phase 2: RL Basics Implementation (Day 2)

**Goal:** Implement the foundation classes — base agent, reward logic, and training loop skeleton.

#### Tasks

1. Design `base_agent.py` with abstract methods: `select_action`, `update`, `reset`
2. Design the reward signal (see Section 5.4)
3. Implement `logger.py` to track episode rewards, step counts, and success rate
4. Write a basic random agent to validate the environment loop works

#### Expected Output

- A complete episode runs: agent takes random steps, environment responds correctly, episode ends when goal is reached or step limit exceeded
- Reward log is populated after each episode

#### Skills Learned

- Abstract base classes in Python
- Designing reward functions
- How an RL training loop is structured

---

### 🟠 Phase 3: Q-Learning Agent (Days 3–4)

**Goal:** Implement a fully functional tabular Q-Learning agent.

#### Tasks

1. Implement `QAgent` in `q_learning_agent.py`
2. Initialize Q-table as `np.zeros((n_states, n_actions))`
3. Implement epsilon-greedy action selection with linear decay
4. Implement Bellman update rule
5. Train on Easy maze → Medium maze
6. Log Q-values and observe convergence

#### Key Implementation Details

```python
# Q-table shape
# States: (row, col) flattened to integer index
# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

# State encoding
def encode_state(row, col, grid_size):
    return row * grid_size + col

# Q-update (Bellman equation)
def update(self, state, action, reward, next_state, done):
    best_next = np.max(self.q_table[next_state])
    td_target = reward + (1 - done) * self.gamma * best_next
    td_error  = td_target - self.q_table[state, action]
    self.q_table[state, action] += self.alpha * td_error
```

#### Expected Output

- Agent converges on Easy maze in ~200–500 episodes
- Reward curve shows upward trend
- Q-table visualized as heatmap showing learned values

#### Skills Learned

- Temporal Difference learning
- Epsilon-greedy exploration
- Bellman equation in code form
- Convergence behavior of Q-Learning

---

### 🔴 Phase 4: Policy Gradient Agent (Days 4–5)

**Goal:** Implement a REINFORCE policy gradient agent using PyTorch.

#### Tasks

1. Build a simple 2-layer MLP policy network in PyTorch
2. Implement `PolicyGradientAgent` in `policy_gradient_agent.py`
3. Implement trajectory collection: store (state, action, reward) for each episode
4. Compute discounted returns G_t for each timestep
5. Apply REINFORCE update (maximize expected return)
6. Add baseline subtraction to reduce variance
7. Train on Medium maze

#### Network Architecture

```
Input Layer:  state encoding (one-hot or normalized (row,col))
              └─► size: grid_size²  (e.g., 121 for 11×11 maze)

Hidden Layer: 128 neurons, ReLU activation

Output Layer: 4 neurons (one per action: UP, DOWN, LEFT, RIGHT)
              └─► Softmax → action probabilities
```

#### Expected Output

- Agent learns policy on Medium maze in ~1000–3000 episodes
- Trajectory stored each episode; update applied at episode end
- Policy visualized as arrows showing preferred actions per cell

#### Skills Learned

- Policy parameterization with neural networks
- REINFORCE algorithm mechanics
- Variance reduction with baselines
- PyTorch: forward pass, loss, backprop

---

### 🟣 Phase 5: Visualization (Day 5)

**Goal:** Build a rich visualization system showing agent behavior and learning progress.

#### Tasks

1. Implement `MazeRenderer` using matplotlib imshow
2. Animate agent movement frame by frame
3. Plot reward curves (moving average overlay)
4. Build Q-value heatmap renderer
5. Build policy arrow visualization (for policy gradient agent)

#### Expected Output

- Live animation of agent navigating maze
- Side-by-side: maze animation + reward curve
- Static heatmap showing what the agent has learned

#### Skills Learned

- matplotlib animation with `FuncAnimation`
- Seaborn/matplotlib heatmaps
- Visualizing policy and value functions

---

### 🟤 Phase 6: UI Interaction (Day 6)

**Goal:** Give the user interactive control over the system.

#### Tasks

1. Implement `ControlPanel` using Tkinter (or matplotlib widgets)
2. Add difficulty dropdown (Easy/Medium/Hard)
3. Add click-to-set-goal interaction on the maze grid
4. Add Start/Stop/Pause/Reset buttons
5. Add speed slider for animation
6. Connect UI events to trainer and environment

#### Expected Output

- User can change maze difficulty → environment resets
- User can click a cell to set as goal → agent retrains
- Start/Stop buttons control the training loop thread
- Live updates visible while training runs

#### Skills Learned

- Tkinter basics
- Threading for non-blocking UI
- Event-driven programming

---

### ⚪ Phase 7: Evaluation & Comparison (Days 7–8)

**Goal:** Evaluate both agents quantitatively and compare against classical methods.

#### Tasks

1. Define evaluation protocol: 100 deterministic test episodes after training
2. Collect metrics: success rate, average steps to goal, convergence episode
3. Run A* and Dijkstra on same mazes → record optimal path length
4. Plot comparison charts
5. Write a summary table of results

#### Expected Output

- Comparison table: Q-Learning vs Policy Gradient vs A* vs Random
- Convergence plots side-by-side
- Written analysis explaining the differences

#### Skills Learned

- Evaluation methodology for RL agents
- Comparing classical vs learning-based methods
- Scientific reporting of results

---

## 5. Maze Environment Design

### 5.1 Grid Representation

The maze is a 2D NumPy array. Each cell holds one of these values:

```
0 = Free cell (agent can walk here)
1 = Wall (impassable)
```

Example: 7×7 Easy maze

```
1 1 1 1 1 1 1
1 S 0 1 0 0 1
1 0 1 1 0 1 1
1 0 0 0 0 1 1
1 1 1 0 1 0 1
1 0 0 0 0 G 1
1 1 1 1 1 1 1
```

`S` = start (row=1, col=1), `G` = goal (row=5, col=5)

The outer border is always walls, so the agent cannot escape the grid.

### 5.2 State and Action Space

**State:** The agent's position (row, col), encoded as a single integer.

```python
state_index = row * grid_width + col
# e.g., for 7x7 grid: state (3,2) → 3*7 + 2 = 23
```

- `observation_space = gym.spaces.Discrete(grid_height * grid_width)`

**Actions:** 4 cardinal directions.

```python
ACTION_MAP = {
    0: (-1,  0),  # UP    (row decreases)
    1: ( 1,  0),  # DOWN  (row increases)
    2: ( 0, -1),  # LEFT  (col decreases)
    3: ( 0,  1),  # RIGHT (col increases)
}
action_space = gym.spaces.Discrete(4)
```

If the agent tries to move into a wall, it stays in place (bounces back) and receives a wall penalty.

### 5.3 Maze Generation — Depth-First Search (DFS) Backtracking

DFS creates perfect mazes (every cell reachable, exactly one path between any two cells).

**Algorithm:**

```
1. Start with all cells as walls (grid = all 1s)
2. Pick a starting cell, mark it free (0), push to stack
3. While stack is not empty:
   a. Current = top of stack
   b. Find all unvisited neighbors (2 steps away, to maintain walls)
   c. If neighbors exist:
      i.  Pick a random neighbor
      ii. Remove wall between current and neighbor
      iii. Mark neighbor as free (0)
      iv. Push neighbor to stack
   d. Else: pop current from stack (backtrack)
```

**Why 2 steps?** In a maze on a grid, walls are *between* cells. Moving 2 cells at a time ensures wall cells exist between open cells.

```python
def generate_maze(height, width, seed=None):
    """Generate a perfect maze using DFS backtracking."""
    if seed:
        np.random.seed(seed)
    
    # Start with all walls
    grid = np.ones((height, width), dtype=int)
    
    # Start at (1,1) — first free cell
    start = (1, 1)
    grid[start] = 0
    stack = [start]
    
    directions = [(-2,0),(2,0),(0,-2),(0,2)]  # 2-step moves
    
    while stack:
        row, col = stack[-1]
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 < nr < height-1 and 0 < nc < width-1 and grid[nr, nc] == 1:
                neighbors.append((nr, nc, dr//2, dc//2))
        
        if neighbors:
            nr, nc, wr, wc = neighbors[np.random.randint(len(neighbors))]
            grid[row + wr, col + wc] = 0  # Remove wall between
            grid[nr, nc] = 0             # Open neighbor
            stack.append((nr, nc))
        else:
            stack.pop()
    
    return grid
```

### 5.4 Reward System Design

The reward signal shapes the agent's behavior. Design it carefully — bad rewards = bad learning.

| Event | Reward | Reason |
|-------|--------|--------|
| Reach goal | +10.0 | Strong positive reinforcement |
| Hit a wall | -0.5 | Discourage bumping into walls |
| Each step taken | -0.01 | Encourage efficiency (find short paths) |
| Episode timeout | -1.0 | Penalize giving up |

**Why a step cost?** Without it, the agent might wander randomly and still accidentally reach the goal. The step cost pushes it to find the *shortest* path.

**Why not a large wall penalty?** A huge penalty (-10) might make the agent too fearful to explore at all, especially early in training when everything is unknown.

```python
def compute_reward(self, prev_pos, new_pos, done, hit_wall, truncated):
    if done:
        return 10.0          # Reached goal!
    if hit_wall:
        return -0.5          # Bumped a wall
    if truncated:
        return -1.0          # Ran out of steps
    return -0.01             # Regular step cost
```

### 5.5 Gymnasium Integration

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, difficulty='medium', goal=None, render_mode=None):
        super().__init__()
        
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIGS[difficulty]  # loads grid size etc.
        self.render_mode = render_mode
        
        # Generate maze grid
        self.grid = generate_maze(
            self.config['height'],
            self.config['width'],
            seed=self.config.get('seed')
        )
        
        # Set start and goal
        self.start_pos = (1, 1)
        self.goal_pos = goal or self.config['default_goal']
        self.grid[self.goal_pos] = 0  # Ensure goal is free
        
        # Define spaces
        n_states = self.config['height'] * self.config['width']
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(4)
        
        # Current agent position
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = self.config['max_steps']
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._encode_state(self.agent_pos), {}
    
    def step(self, action):
        dr, dc = ACTION_MAP[action]
        row, col = self.agent_pos
        new_row, new_col = row + dr, col + dc
        
        hit_wall = False
        
        # Check bounds and wall collision
        if (0 <= new_row < self.config['height'] and
            0 <= new_col < self.config['width'] and
            self.grid[new_row, new_col] == 0):
            self.agent_pos = (new_row, new_col)
        else:
            hit_wall = True  # Stay in place
        
        self.steps += 1
        done = (self.agent_pos == self.goal_pos)
        truncated = (self.steps >= self.max_steps)
        
        reward = self.compute_reward(
            (row, col), self.agent_pos, done, hit_wall, truncated
        )
        
        obs = self._encode_state(self.agent_pos)
        info = {'steps': self.steps, 'hit_wall': hit_wall}
        
        return obs, reward, done, truncated, info
    
    def _encode_state(self, pos):
        return pos[0] * self.config['width'] + pos[1]
    
    def set_goal(self, new_goal):
        """Allow dynamic goal change from UI."""
        self.goal_pos = new_goal
        self.grid[new_goal] = 0
    
    def render(self):
        if self.render_mode == 'human':
            # handled by visualization module
            pass
        elif self.render_mode == 'rgb_array':
            return self._grid_to_rgb()
```

---

## 6. Algorithms Explained — Teach Me

---

### 6.1 Q-Learning

#### 🧠 Intuition

Imagine you are a tourist in a city you've never visited. Every intersection you visit, you maintain a little notebook. For each intersection + direction combination, you write a score: "if I go RIGHT here, how good is this decision in the long run?"

At first, all scores are 0 (you know nothing). Each time you try a direction and see what happens, you update your score. Over time, your notebook becomes accurate, and you can always just pick the direction with the highest score. That notebook is the **Q-table**.

#### 📐 The Bellman Equation

The central update rule of Q-Learning:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') - Q(s, a)]
                                    a'
```

Let's break this down piece by piece:

| Symbol | Name | Meaning |
|--------|------|---------|
| `Q(s, a)` | Q-value | Expected future reward from state `s` taking action `a` |
| `α` (alpha) | Learning rate | How fast we update. 0 = no learning, 1 = overwrite completely |
| `r` | Reward | Immediate reward received after taking action `a` |
| `γ` (gamma) | Discount factor | How much we care about future rewards (0=greedy, 1=far-sighted) |
| `max Q(s', a')` | Best future value | Best Q-value achievable from the next state |
| `[...]` | TD Error | The "surprise" — how wrong our current estimate was |

**The Magic:** Q-Learning doesn't need a model of the environment. It learns purely from experience.

**Example step-by-step:**

```
State: Agent at (3,3)
Action: Move RIGHT → ends up at (3,4)
Reward: -0.01 (step cost)

Before update: Q((3,3), RIGHT) = 0.5
Best Q from (3,4): max Q((3,4), *) = 2.0
gamma = 0.99, alpha = 0.1

TD Target = -0.01 + 0.99 * 2.0 = 1.97
TD Error  = 1.97 - 0.5 = 1.47

New Q((3,3), RIGHT) = 0.5 + 0.1 * 1.47 = 0.647
```

The Q-value increased! The agent now "knows" that going RIGHT from (3,3) is better than it thought.

#### 📜 Pseudocode

```
Initialize Q-table: Q[s][a] = 0 for all s, a

For each episode:
    state = env.reset()
    
    While not done:
        # Exploration vs Exploitation
        if random() < epsilon:
            action = random action          # Explore
        else:
            action = argmax Q[state]        # Exploit best known
        
        next_state, reward, done = env.step(action)
        
        # Bellman update
        best_future = max(Q[next_state])
        Q[state][action] += alpha * (
            reward + gamma * best_future - Q[state][action]
        )
        
        state = next_state
    
    epsilon *= decay_rate  # Explore less over time
```

#### ✅ When to Use Q-Learning

- **Small, discrete state spaces** (tabular Q-table fits in memory)
- When you need **fast training** (no neural network overhead)
- When the environment is **fully observable** (you always know your position)
- **Mazes with small grids** are a perfect use case

#### ⚠️ Limitations

- Q-table size = n_states × n_actions. For a 15×15 maze = 900 × 4 = 3,600 entries. Fine. But for a 1000×1000 grid → 4,000,000 entries. Then we need DQN.
- Doesn't generalize: every state is learned independently (no sharing of information between similar states)

---

### 6.2 Policy Gradient — REINFORCE

#### 🧠 Intuition

Instead of learning a table of values, what if we directly learn a **policy** — a function that maps states to action probabilities?

Think of it like this: Q-Learning learns "what is the value of each action?" Policy Gradient learns "what is the best action to take here, directly?"

The policy is a neural network `π_θ(a | s)` that takes the current state and outputs a probability distribution over actions.

**Training idea:** If an episode was successful (high total reward), make the actions taken more likely in the future. If an episode failed, make those actions less likely.

This is exactly the REINFORCE algorithm (also called Monte Carlo Policy Gradient).

#### 📐 The REINFORCE Objective

We want to maximize the **expected total reward**:

```
J(θ) = E_π[G_t]
```

Where `G_t` is the **discounted return** from timestep t:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T
```

The gradient of this objective (the REINFORCE theorem):

```
∇J(θ) = E_π [ G_t · ∇ log π_θ(a_t | s_t) ]
```

In plain English: "Move the parameters in the direction that increases the probability of actions that led to high returns."

**The baseline trick:** Subtracting a baseline `b` (usually the mean return) reduces variance without changing the expected gradient:

```
∇J(θ) ≈ (G_t - b) · ∇ log π_θ(a_t | s_t)
```

#### 📜 Pseudocode

```
Initialize policy network π_θ with random weights

For each episode:
    trajectory = []
    state = env.reset()
    
    While not done:
        probs = π_θ(state)               # Forward pass → action probs
        action = sample from probs        # Stochastic action selection
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    
    # Compute discounted returns for each timestep
    G = 0
    returns = []
    for (s, a, r) in reversed(trajectory):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Normalize returns (baseline subtraction)
    returns = (returns - mean(returns)) / (std(returns) + 1e-8)
    
    # Compute policy gradient loss and update
    loss = 0
    for (s, a, _), G in zip(trajectory, returns):
        log_prob = log(π_θ(a | s))
        loss -= G * log_prob              # Negative because we maximize
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### ✅ When to Use Policy Gradient

- When the **state space is large** or continuous (neural network handles it)
- When you need **stochastic policies** (can explore more naturally)
- When the environment has **sparse rewards** and you need flexibility
- As a stepping stone toward **Actor-Critic** and **PPO** methods

#### ⚠️ Limitations

- High variance in gradient estimates (can be noisy)
- Requires many episodes to converge (especially without a baseline)
- Needs careful learning rate tuning
- On a small maze, Q-Learning will often converge faster

#### ⚖️ Q-Learning vs Policy Gradient — Side by Side

| Property | Q-Learning | Policy Gradient |
|----------|-----------|-----------------|
| What it learns | Value function Q(s,a) | Policy π(a\|s) directly |
| Update frequency | Every step | Every episode (REINFORCE) |
| Exploration | Epsilon-greedy (explicit) | Built into stochastic policy |
| Memory | Q-table (small mazes) | Neural network weights |
| Convergence | Faster on small mazes | Slower, more variance |
| Scalability | Poor (table grows) | Good (network generalizes) |
| Interpretability | Very high (read table) | Lower (black box) |

---

## 7. Training Pipeline

### 7.1 Episode Loop

```python
# training/trainer.py

class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = Logger()
        self.is_running = False
    
    def train(self, n_episodes):
        self.is_running = True
        
        for episode in range(n_episodes):
            if not self.is_running:
                break
            
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Agent learns from this transition
                self.agent.update(state, action, reward, next_state, done or truncated)
                
                state = next_state
                total_reward += reward
                steps += 1
                done = done or truncated
            
            # Post-episode updates
            self.agent.on_episode_end()       # epsilon decay, trajectory flush
            self.logger.log(episode, total_reward, steps)
            
            if episode % 100 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:6.2f} | "
                      f"Steps: {steps:4d} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")
```

### 7.2 Hyperparameters

```python
# training/hyperparams.py

Q_LEARNING_CONFIG = {
    'alpha': 0.1,          # Learning rate — how fast Q-values update
    'gamma': 0.99,         # Discount factor — care about future rewards
    'epsilon_start': 1.0,  # Start fully exploratory
    'epsilon_end': 0.01,   # End almost fully exploitative
    'epsilon_decay': 0.995, # Multiply epsilon by this each episode
    'n_episodes': {
        'easy': 500,
        'medium': 2000,
        'hard': 5000,
    },
    'max_steps_per_episode': {
        'easy': 100,
        'medium': 300,
        'hard': 600,
    }
}

POLICY_GRADIENT_CONFIG = {
    'learning_rate': 3e-4,   # Adam optimizer learning rate
    'gamma': 0.99,
    'hidden_size': 128,
    'n_episodes': {
        'easy': 1000,
        'medium': 5000,
        'hard': 10000,
    },
    'baseline': True,        # Use mean return as baseline
}
```

### 7.3 Hyperparameter Tuning Guide

| Hyperparameter | Too Low | Too High | Good Starting Point |
|----------------|---------|----------|---------------------|
| `alpha` (Q-LR) | Learns very slowly | Oscillates, unstable | 0.1 |
| `gamma` | Short-sighted, ignores future | Over-values distant rewards | 0.99 |
| `epsilon_start` | Under-explores | — | 1.0 (always start exploring) |
| `epsilon_decay` | Stops exploring too fast | Never stops exploring | 0.995–0.999 |
| `learning_rate` (PG) | Doesn't learn | Diverges | 1e-4 to 3e-4 |

### 7.4 Exploration Strategies

#### Strategy 1: Epsilon-Greedy (Used in Q-Learning)

```
with probability ε  → take a RANDOM action     (explore)
with probability 1-ε → take the BEST known action (exploit)
```

**Linear decay:**
```python
epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / n_episodes)
```

**Exponential decay (preferred):**
```python
epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

#### Strategy 2: Stochastic Policy (Used in Policy Gradient)

The policy network outputs probabilities. Sampling from this distribution IS exploration. No explicit epsilon needed. Early in training, the network outputs nearly uniform probabilities → high exploration. As it learns, probabilities concentrate on good actions → exploitation.

#### Strategy 3: Boltzmann (Softmax) Exploration

```
P(a) = exp(Q(s,a) / T) / Σ exp(Q(s,a') / T)
```

Temperature `T` controls randomness. High T → uniform, Low T → greedy. Smooth version of epsilon-greedy.

### 7.5 Logging

```python
# training/logger.py

class Logger:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.steps = []
        self.success = []
    
    def log(self, episode, reward, steps, success=False):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.success.append(success)
    
    def moving_average(self, window=50):
        rewards = np.array(self.rewards)
        return np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    def success_rate(self, last_n=100):
        return np.mean(self.success[-last_n:])
    
    def export_csv(self, path):
        import pandas as pd
        df = pd.DataFrame({
            'episode': self.episodes,
            'reward': self.rewards,
            'steps': self.steps,
            'success': self.success
        })
        df.to_csv(path, index=False)
```

---

## 8. Visualization System

### 8.1 Maze Renderer

```python
# visualization/maze_renderer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class MazeRenderer:
    
    COLORS = {
        'wall':  '#2C3E50',   # Dark blue-gray
        'free':  '#ECF0F1',   # Light gray
        'agent': '#E74C3C',   # Red
        'goal':  '#2ECC71',   # Green
        'path':  '#F39C12',   # Orange (visited cells)
        'start': '#3498DB',   # Blue
    }
    
    def __init__(self, env, figsize=(8, 8)):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.im = None
        self._setup_plot()
    
    def _setup_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title('RL Maze Solver', fontsize=14, fontweight='bold')
    
    def render(self, agent_pos, visited=None, path=None):
        """Render the current state of the maze."""
        grid = self.env.grid.copy().astype(float)
        h, w = grid.shape
        
        # Build RGB image
        rgb = np.zeros((h, w, 3))
        
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 1:
                    rgb[r, c] = self._hex_to_rgb(self.COLORS['wall'])
                else:
                    rgb[r, c] = self._hex_to_rgb(self.COLORS['free'])
        
        # Color visited cells
        if visited:
            for (r, c) in visited:
                rgb[r, c] = self._hex_to_rgb(self.COLORS['path'])
        
        # Color start
        sr, sc = self.env.start_pos
        rgb[sr, sc] = self._hex_to_rgb(self.COLORS['start'])
        
        # Color goal
        gr, gc = self.env.goal_pos
        rgb[gr, gc] = self._hex_to_rgb(self.COLORS['goal'])
        
        # Color agent
        ar, ac = agent_pos
        rgb[ar, ac] = self._hex_to_rgb(self.COLORS['agent'])
        
        if self.im is None:
            self.im = self.ax.imshow(rgb, interpolation='nearest')
        else:
            self.im.set_data(rgb)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _hex_to_rgb(self, hex_color):
        h = hex_color.lstrip('#')
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
```

### 8.2 Training Progress Plots

```python
# visualization/training_plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(logger, agent_name='Agent', window=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{agent_name} — Training Progress', fontsize=14)
    
    # --- Plot 1: Raw reward per episode ---
    axes[0].plot(logger.rewards, alpha=0.3, color='steelblue', label='Raw')
    ma = logger.moving_average(window)
    axes[0].plot(range(window-1, len(logger.rewards)), ma,
                 color='steelblue', linewidth=2, label=f'MA-{window}')
    axes[0].set_title('Reward per Episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    
    # --- Plot 2: Steps per episode ---
    axes[1].plot(logger.steps, alpha=0.3, color='salmon')
    step_ma = np.convolve(logger.steps, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(logger.steps)), step_ma,
                 color='salmon', linewidth=2)
    axes[1].set_title('Steps per Episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    
    # --- Plot 3: Rolling success rate ---
    success = np.array(logger.success, dtype=float)
    success_ma = np.convolve(success, np.ones(window)/window, mode='valid')
    axes[2].plot(range(window-1, len(success)), success_ma,
                 color='mediumseagreen', linewidth=2)
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f'Success Rate (MA-{window})')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    
    plt.tight_layout()
    plt.show()
```

### 8.3 Q-Value Heatmap

```python
# visualization/policy_heatmap.py

import matplotlib.pyplot as plt
import numpy as np

def plot_q_heatmap(q_table, env):
    """Show max Q-value per cell as a heatmap."""
    h, w = env.config['height'], env.config['width']
    q_grid = np.full((h, w), np.nan)
    
    for r in range(h):
        for c in range(w):
            if env.grid[r, c] == 0:  # Free cell only
                state = r * w + c
                q_grid[r, c] = np.max(q_table[state])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(q_grid, cmap='RdYlGn', aspect='equal')
    plt.colorbar(im, ax=ax, label='Max Q-Value')
    ax.set_title('Q-Value Heatmap (greener = better)')
    plt.show()

def plot_policy_arrows(q_table, env):
    """Show arrows for best action at each cell."""
    h, w = env.config['height'], env.config['width']
    
    # Action → direction vectors
    ARROW = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Draw maze background
    ax.imshow(env.grid, cmap='binary', aspect='equal')
    
    for r in range(h):
        for c in range(w):
            if env.grid[r, c] == 0:
                state = r * w + c
                best_action = np.argmax(q_table[state])
                dy, dx = ARROW[best_action]
                ax.annotate('', xy=(c+dx*0.4, r+dy*0.4), xytext=(c, r),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark goal and start
    gr, gc = env.goal_pos
    sr, sc = env.start_pos
    ax.plot(gc, gr, 'g*', markersize=15, label='Goal')
    ax.plot(sc, sr, 'bs', markersize=12, label='Start')
    ax.legend()
    ax.set_title('Learned Policy — Arrows show best action')
    plt.show()
```

---

## 9. UI Design

### 9.1 UI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Maze Solver UI                        │
│                                                             │
│  ┌──────────────────────┐  ┌─────────────────────────────┐ │
│  │                      │  │  CONTROLS                   │ │
│  │    MAZE DISPLAY      │  │                             │ │
│  │   (matplotlib axes)  │  │  Difficulty: [Easy ▼]       │ │
│  │                      │  │                             │ │
│  │  ┌─┬─┬─┬─┬─┬─┬─┐   │  │  Algorithm: [Q-Learning ▼] │ │
│  │  │ │ │ │ │ │ │ │   │  │                             │ │
│  │  ├─┼─┼─┼─┼─┼─┼─┤   │  │  [Click maze to set goal]  │ │
│  │  │ │ │S│ │ │ │ │   │  │                             │ │
│  │  ├─┼─┼─┼─┼─┼─┼─┤   │  │  Episodes: [2000    ]       │ │
│  │  │ │ │ │ │ │ │G│   │  │                             │ │
│  │  └─┴─┴─┴─┴─┴─┴─┘   │  │  Speed:    [────●────]      │ │
│  │                      │  │                             │ │
│  │  Ep: 1247 | R: 8.3  │  │  [▶ Start] [⏸ Pause] [↺]  │ │
│  └──────────────────────┘  │                             │ │
│                             │  Progress:                  │ │
│  ┌──────────────────────┐  │  ██████░░░░ 62%             │ │
│  │   REWARD CURVE       │  │                             │ │
│  │   (live updating)    │  │  Avg Reward:   7.42         │ │
│  │                      │  │  Success Rate: 84%          │ │
│  └──────────────────────┘  │  Best Reward:  9.91         │ │
│                             └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Implementation with Tkinter

```python
# ui/control_panel.py

import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Maze Solver")
        self.root.geometry("1100x700")
        
        self.trainer = None
        self.training_thread = None
        self.is_paused = False
        
        self._build_ui()
    
    def _build_ui(self):
        # Left panel: maze display
        left = tk.Frame(self.root, bg='white')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Embed matplotlib figure
        self.fig_maze, self.ax_maze = plt.subplots(figsize=(5,5))
        self.canvas_maze = FigureCanvasTkAgg(self.fig_maze, master=left)
        self.canvas_maze.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_maze.mpl_connect('button_press_event', self._on_maze_click)
        
        # Right panel: controls
        right = tk.Frame(self.root, bg='#F0F0F0', width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right.pack_propagate(False)
        
        # Title
        tk.Label(right, text="RL Maze Solver", font=('Arial', 14, 'bold'),
                 bg='#F0F0F0').pack(pady=10)
        
        # Difficulty dropdown
        tk.Label(right, text="Maze Difficulty:", bg='#F0F0F0').pack(anchor='w', padx=10)
        self.difficulty_var = tk.StringVar(value='medium')
        ttk.Combobox(right, textvariable=self.difficulty_var,
                     values=['easy', 'medium', 'hard'], state='readonly').pack(
                     fill=tk.X, padx=10, pady=2)
        
        # Algorithm dropdown
        tk.Label(right, text="Algorithm:", bg='#F0F0F0').pack(anchor='w', padx=10, pady=(8,0))
        self.algo_var = tk.StringVar(value='Q-Learning')
        ttk.Combobox(right, textvariable=self.algo_var,
                     values=['Q-Learning', 'Policy Gradient'], state='readonly').pack(
                     fill=tk.X, padx=10, pady=2)
        
        # Episodes
        tk.Label(right, text="Episodes:", bg='#F0F0F0').pack(anchor='w', padx=10, pady=(8,0))
        self.episodes_var = tk.IntVar(value=2000)
        tk.Entry(right, textvariable=self.episodes_var).pack(fill=tk.X, padx=10, pady=2)
        
        # Instruction
        tk.Label(right, text="Click maze to set goal position",
                 bg='#F0F0F0', fg='gray', font=('Arial', 9, 'italic')).pack(pady=5)
        
        # Speed slider
        tk.Label(right, text="Animation Speed:", bg='#F0F0F0').pack(anchor='w', padx=10, pady=(8,0))
        self.speed_var = tk.DoubleVar(value=0.05)
        tk.Scale(right, from_=0.001, to=0.2, resolution=0.001,
                 orient=tk.HORIZONTAL, variable=self.speed_var,
                 bg='#F0F0F0').pack(fill=tk.X, padx=10)
        
        # Buttons
        btn_frame = tk.Frame(right, bg='#F0F0F0')
        btn_frame.pack(pady=10)
        
        self.btn_start = tk.Button(btn_frame, text="▶ Start",
                                   command=self._start_training,
                                   bg='#2ECC71', fg='white', width=8)
        self.btn_start.grid(row=0, column=0, padx=3)
        
        self.btn_pause = tk.Button(btn_frame, text="⏸ Pause",
                                   command=self._pause_training,
                                   bg='#F39C12', fg='white', width=8)
        self.btn_pause.grid(row=0, column=1, padx=3)
        
        self.btn_reset = tk.Button(btn_frame, text="↺ Reset",
                                   command=self._reset,
                                   bg='#E74C3C', fg='white', width=8)
        self.btn_reset.grid(row=0, column=2, padx=3)
        
        # Stats
        stats_frame = tk.LabelFrame(right, text="Statistics", bg='#F0F0F0')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stat_labels = {}
        for key, label in [('episode', 'Episode'), ('reward', 'Avg Reward'),
                            ('success', 'Success Rate'), ('best', 'Best Reward')]:
            row = tk.Frame(stats_frame, bg='#F0F0F0')
            row.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(row, text=f"{label}:", bg='#F0F0F0', width=14, anchor='w').pack(side=tk.LEFT)
            self.stat_labels[key] = tk.Label(row, text='—', bg='#F0F0F0', font=('Arial', 10, 'bold'))
            self.stat_labels[key].pack(side=tk.LEFT)
    
    def _on_maze_click(self, event):
        """Set goal position by clicking on the maze."""
        if event.inaxes and self.trainer:
            col, row = int(round(event.xdata)), int(round(event.ydata))
            if self.trainer.env.grid[row, col] == 0:
                self.trainer.env.set_goal((row, col))
                print(f"Goal moved to ({row}, {col})")
    
    def _start_training(self):
        self.training_thread = threading.Thread(
            target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _pause_training(self):
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="▶ Resume" if self.is_paused else "⏸ Pause")
    
    def _reset(self):
        if self.trainer:
            self.trainer.is_running = False
        # Re-initialize environment and agent
    
    def _training_loop(self):
        """Run in background thread to avoid blocking UI."""
        # Connect trainer and update stats_labels periodically
        pass
    
    def run(self):
        self.root.mainloop()
```

---

## 10. Algorithm Comparison

### 10.1 Performance Metrics

Measure these after training to compare agents fairly:

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Convergence Episode** | Episode at which success rate > 90% | Monitor success rate rolling window |
| **Final Success Rate** | % of last 100 episodes where goal was reached | `sum(success[-100:]) / 100` |
| **Avg Steps to Goal** | Mean steps in successful episodes | Average over successful episodes |
| **Optimal Gap** | Ratio of avg steps vs A* shortest path | `avg_steps / a_star_length` |
| **Training Time** | Wall clock time to convergence | `time.time()` around training loop |

### 10.2 Expected Results (Illustrative)

| Agent | Easy Maze | Medium Maze | Hard Maze | Notes |
|-------|-----------|-------------|-----------|-------|
| Random | ~1% success | ~0.1% | ~0% | Baseline |
| Q-Learning | 95%+ @ ep.200 | 90%+ @ ep.1000 | 85%+ @ ep.3000 | Fast, stable |
| Policy Gradient | 90%+ @ ep.500 | 80%+ @ ep.2500 | 75%+ @ ep.7000 | Slower, more variance |
| A* (classical) | 100% @ step 1 | 100% @ step 1 | 100% @ step 1 | Optimal, no learning |
| Dijkstra | 100% @ step 1 | 100% @ step 1 | 100% @ step 1 | Optimal, no learning |

### 10.3 Convergence Plot (Conceptual)

```
Success
Rate
100% │                    ____________________  A* (always optimal)
     │                   /
 90% │         _________/ ─────────────────── Q-Learning converged
     │        /
 70% │       /
     │     _/  ─────────────────────────────── Policy Gradient converged
 50% │    /
     │   /
 30% │  /
     │ /
 10% │/
  0% ├──────────────────────────────────────────────────────►
     0    500   1000   2000   3000   4000   5000    Episodes
```

### 10.4 Analysis Points

**Q-Learning Wins On:**
- Convergence speed (learns faster on small tabular problems)
- Stability (less variance in updates)
- Interpretability (Q-table is human-readable)

**Policy Gradient Wins On:**
- Scalability (can scale to much larger state spaces with neural networks)
- Stochastic environments (naturally handles randomness)
- Continuous action spaces (future extension)

**Classical Methods Win On:**
- Optimality guarantee (A* always finds shortest path)
- No training required
- Deterministic behavior

**Key Insight:** On small discrete mazes, classical methods are always the "best" — but they require a complete map of the environment. RL methods learn from *experience only*, which is far more powerful when you don't have a model of the world.

---

## 11. Extensions & Advanced Topics

### 11.1 Classical Pathfinding Comparison

#### A* Algorithm

A* finds the shortest path using a heuristic. In a maze, use **Manhattan distance** to the goal as the heuristic.

```python
# extensions/pathfinding.py
import heapq

def a_star(grid, start, goal):
    """A* pathfinding algorithm."""
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])  # Manhattan
    
    open_set = [(0 + heuristic(start), 0, start, [start])]
    visited = set()
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            return path, g  # path and cost
        
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr,nc] == 0 and (nr,nc) not in visited):
                new_g = g + 1
                heapq.heappush(open_set,
                    (new_g + heuristic((nr,nc)), new_g, (nr,nc), path + [(nr,nc)]))
    
    return None, float('inf')  # No path found

def dijkstra(grid, start, goal):
    """Dijkstra's algorithm (A* without heuristic)."""
    open_set = [(0, start, [start])]
    visited = set()
    
    while open_set:
        cost, current, path = heapq.heappop(open_set)
        if current in visited: continue
        visited.add(current)
        if current == goal: return path, cost
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr,nc] == 0 and (nr,nc) not in visited):
                heapq.heappush(open_set, (cost+1, (nr,nc), path+[(nr,nc)]))
    
    return None, float('inf')
```

**When to use each:**
- **A*:** When a good heuristic exists (grid mazes, GPS navigation). Faster than Dijkstra.
- **Dijkstra:** When no heuristic is available or costs vary between edges.
- **RL:** When the environment is unknown, stochastic, or too complex to model.

### 11.2 Deep Q-Network (DQN)

When the maze is too large for a Q-table, replace it with a neural network.

**Key differences from tabular Q-Learning:**

| Feature | Q-Learning | DQN |
|---------|-----------|-----|
| Q-function | Table | Neural Network |
| State input | Integer index | Full grid as image or features |
| Stability | Stable | Requires tricks |
| Scalability | Poor (large mazes) | Excellent |

**DQN Tricks for Stability:**

1. **Experience Replay:** Store transitions `(s, a, r, s', done)` in a buffer. Sample random mini-batches for training. Breaks temporal correlations.

2. **Target Network:** Keep a frozen copy of the Q-network. Use it to compute `max Q(s', a')` targets. Update it every N steps. Prevents chasing a moving target.

```python
# extensions/dqn_agent.py (skeleton)
import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.q_net = DQNetwork(state_size, action_size)
        self.target_net = DQNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        # ... compute loss and update q_net
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
```

### 11.3 Saving and Loading Models

```python
# utils/model_io.py
import pickle
import torch

def save_q_table(q_table, path):
    """Save Q-table to disk."""
    with open(path, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {path}")

def load_q_table(path):
    """Load Q-table from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_policy_network(model, optimizer, episode, path):
    """Save policy gradient model checkpoint."""
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Policy network saved to {path}")

def load_policy_network(model, optimizer, path):
    """Load policy gradient model from checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode']
```

---

## 12. Learning Notes — Core RL Concepts

---

### 12.1 What is Reinforcement Learning?

**Simple explanation:**

RL is learning by doing. An **agent** (the learner) takes **actions** in an **environment**. The environment responds with a new **state** and a **reward** signal. The agent's goal is to maximize the total reward over time.

Think of training a dog:
- Dog (agent) sits when told (action)
- You give it a treat (reward)
- Dog learns: "sitting = treat"
- Over time, dog reliably sits on command

No one programmed the dog with explicit rules. It learned from experience. That's RL.

```
    ┌──────────┐    action    ┌─────────────┐
    │          │─────────────►│             │
    │  AGENT   │             │ ENVIRONMENT │
    │          │◄────────────│             │
    └──────────┘  state,     └─────────────┘
                  reward
```

---

### 12.2 State (s)

**What it is:** A snapshot of the environment at a moment in time. Everything the agent needs to make a decision.

**In our maze:** State = the agent's (row, col) position.

**Why it matters:** The agent can only act based on what it perceives. If the state is incomplete (partial observability), the agent is effectively blind to important information.

**Example:** In Chess, the state is the full board position. In our maze, the state is just the current cell.

---

### 12.3 Action (a)

**What it is:** Something the agent can do to change the state.

**In our maze:** Actions = {UP, DOWN, LEFT, RIGHT}

**Important:** Actions have consequences the agent must learn from experience. Early in training, the agent doesn't know that going LEFT from (3,3) hits a wall.

---

### 12.4 Reward (r)

**What it is:** A scalar signal from the environment telling the agent how good or bad its last action was.

**In our maze:**
- `+10` for reaching the goal
- `-0.01` for each step (step cost)
- `-0.5` for hitting a wall

**Critical design principle:** The reward should reflect what you actually want, not how you think the agent should achieve it. This is called "reward shaping" and is an active area of research.

**Famous failure:** An RL agent playing a boat racing game learned to spin in circles collecting power-ups instead of racing — because the reward was for collecting points, not winning races. Always design rewards carefully!

---

### 12.5 Policy (π)

**What it is:** The agent's strategy — a mapping from states to actions (or action probabilities).

**Two types:**

**Deterministic policy:** `π(s) = a` — always take action `a` in state `s`
```
π((3,3)) = RIGHT
π((3,4)) = RIGHT
π((3,5)) = DOWN
```

**Stochastic policy:** `π(a|s) = probability` — take action `a` in state `s` with some probability
```
π(RIGHT | (3,3)) = 0.7
π(DOWN  | (3,3)) = 0.2
π(LEFT  | (3,3)) = 0.1
π(UP    | (3,3)) = 0.0
```

**Q-Learning** learns the Q-values, and the policy is derived (always pick argmax Q). **Policy Gradient** directly learns the stochastic policy.

---

### 12.6 Value Function (V and Q)

**State Value Function V(s):** How good is it to be in state `s`?

```
V(s) = expected total reward starting from state s, following policy π
```

**Action-Value Function Q(s,a):** How good is it to take action `a` in state `s`?

```
Q(s, a) = expected total reward after taking action a in state s,
           then following policy π
```

**Relationship:**
```
V(s) = max Q(s, a)    (for optimal deterministic policy)
           a
```

**Example:**
```
V((5,5)) = 10.0     (this IS the goal — huge value)
V((5,4)) = 9.9      (one step from goal)
V((5,3)) = 9.8      (two steps from goal)
V((1,1)) = 8.5      (start — far from goal, but reachable)
```

The value function is a "map" of how close each position is to the goal, accounting for future rewards.

---

### 12.7 Exploration vs Exploitation Dilemma

**The dilemma:** Should the agent try new things (explore) or use what it already knows is good (exploit)?

**Analogy:** You love restaurant A. But should you try restaurant B? If you always go to A, you might miss something even better (B). But if you always try new places, you never enjoy your favorite (A).

**RL Solution — Epsilon-Greedy:**
```
- With probability ε → go to a random restaurant (explore)
- With probability 1-ε → go to your current favorite (exploit)
- Slowly reduce ε over time
```

**Why decay epsilon?**

Early training: Know nothing → explore a lot (high ε)
Late training: Know a lot → mostly exploit, slight exploration (low ε)

This is analogous to how humans behave: children explore everything; adults mostly exploit what they know works.

---

### 12.8 Discount Factor (γ)

**The question:** Is a reward now better than the same reward later?

**γ close to 0 (shortsighted):** Agent only cares about immediate reward. Might miss strategies that involve short-term pain for long-term gain.

**γ close to 1 (far-sighted):** Agent cares almost equally about rewards far in the future.

**Mathematical effect:**

A reward `R` received after `k` steps is worth `γ^k * R` now.

```
γ = 0.99, reward = 10, 5 steps away:
Present value = 0.99^5 * 10 = 9.51  (still valuable)

γ = 0.5, reward = 10, 5 steps away:
Present value = 0.5^5 * 10 = 0.31  (barely valuable)
```

In our maze, use `γ = 0.99` — we want the agent to value reaching the goal even if it takes many steps.

---

### 12.9 The Bellman Optimality Equation

**The core insight of RL:** The value of a state equals the immediate reward plus the discounted value of the best next state.

```
Q*(s, a) = r + γ · max Q*(s', a')
                      a'
```

This is recursive. To know Q(s, a), you need Q(s', a'). But that also depends on Q(s'', a'')...

**Solution:** Start with `Q = 0` everywhere. Apply the update rule repeatedly. Over many episodes, Q-values converge to the true Q* (this is proven mathematically under certain conditions).

This iterative process is called **Temporal Difference (TD) learning** — we update based on the difference between our current estimate and a better estimate.

---

### 12.10 On-Policy vs Off-Policy Learning

**Off-Policy (Q-Learning):** The agent can learn the optimal policy even while following a different (exploratory) policy.

The Bellman update uses `max Q(s', a')` — it assumes the agent will always take the best action in the future, even if it currently takes random actions.

**On-Policy (REINFORCE):** The agent learns from the policy it is currently executing.

If the policy is stochastic, the agent collects trajectories *from that policy* and updates that same policy.

**Which is better?** Depends on the situation:
- Off-policy (Q-Learning): More data efficient (can reuse old data)
- On-policy (Policy Gradient): More stable for complex policies, better for continuous actions

---

## 13. Full Dependency List

```
# requirements.txt

# Core RL & Environment
gymnasium>=0.29.0       # Custom environment framework
numpy>=1.24.0           # Numerical computing (Q-tables, grids)

# Neural Networks (Policy Gradient, DQN)
torch>=2.0.0            # PyTorch for policy network
torchvision>=0.15.0     # Optional: for image-based state inputs

# Visualization
matplotlib>=3.7.0       # Maze rendering and training plots
seaborn>=0.12.0         # Optional: prettier plots

# UI
# tkinter is included with Python standard library (no pip install needed)

# Utilities
pandas>=2.0.0           # CSV export of training logs
scipy>=1.10.0           # Optional: statistical analysis

# Development
jupyter>=1.0.0          # For notebooks
ipykernel>=6.0.0        # Jupyter kernel
```

**Installation:**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify Gymnasium is working
python -c "import gymnasium; print(gymnasium.__version__)"

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## 14. Quick-Start Checklist

Use this checklist to track your progress through the project.

### Week 1

**Day 1 — Environment Foundations**
- [ ] Set up virtual environment and install dependencies
- [ ] Implement `maze_generator.py` with DFS backtracking
- [ ] Define `DIFFICULTY_CONFIGS` in `maze_configs.py`
- [ ] Create `MazeEnv` skeleton with `observation_space` and `action_space`

**Day 2 — Environment Completion**
- [ ] Implement `reset()` and `step()` in `MazeEnv`
- [ ] Implement reward function
- [ ] Write random agent test: run 10 episodes, verify step/reward flow
- [ ] Implement basic `Logger` class

**Day 3 — Q-Learning Agent**
- [ ] Implement `QAgent.__init__` with Q-table initialization
- [ ] Implement `select_action` with epsilon-greedy
- [ ] Implement `update` with Bellman equation
- [ ] Implement `on_episode_end` with epsilon decay

**Day 4 — Q-Learning Training**
- [ ] Implement `Trainer.train()` episode loop
- [ ] Train Q-agent on Easy maze (500 episodes)
- [ ] Verify convergence (reward curve rises)
- [ ] Train on Medium maze (2000 episodes)

**Day 5 — Policy Gradient Agent**
- [ ] Build `PolicyNetwork` MLP in PyTorch
- [ ] Implement `PolicyGradientAgent.__init__`
- [ ] Implement `select_action` with softmax sampling
- [ ] Implement trajectory storage and `update` with REINFORCE

**Day 6 — Policy Gradient Training + Visualization**
- [ ] Train Policy Gradient on Easy maze
- [ ] Implement `MazeRenderer` with matplotlib
- [ ] Implement reward curve plotter
- [ ] Test live visualization during training

**Day 7 — Visualization Completion**
- [ ] Implement Q-value heatmap
- [ ] Implement policy arrow visualization
- [ ] Plot training progress side-by-side for both agents

### Week 2

**Day 1 — UI**
- [ ] Create Tkinter `ControlPanel` skeleton
- [ ] Embed matplotlib in Tkinter window
- [ ] Add difficulty dropdown and algorithm selector
- [ ] Add Start/Stop/Pause/Reset buttons

**Day 2 — UI Interaction**
- [ ] Implement click-to-set-goal on maze canvas
- [ ] Connect UI buttons to trainer (threading)
- [ ] Add live statistics display
- [ ] Test full UI loop: change difficulty → retrain → observe

**Day 3 — Classical Pathfinding**
- [ ] Implement A* in `extensions/pathfinding.py`
- [ ] Implement Dijkstra in `extensions/pathfinding.py`
- [ ] Run both on all three maze difficulties
- [ ] Record optimal path lengths for comparison

**Day 4 — Evaluation**
- [ ] Implement formal evaluation: 100 deterministic test episodes
- [ ] Collect all metrics: success rate, avg steps, convergence episode
- [ ] Run comparison across all agents
- [ ] Generate comparison plots

**Day 5 — Model Saving/Loading**
- [ ] Implement `save_q_table` and `load_q_table`
- [ ] Implement `save_policy_network` and `load_policy_network`
- [ ] Test: train → save → load → evaluate (should maintain performance)
- [ ] Add save/load buttons to UI

**Day 6 — DQN Extension**
- [ ] Implement `ReplayBuffer`
- [ ] Implement `DQNetwork` PyTorch module
- [ ] Implement `DQNAgent` with experience replay + target network
- [ ] Train DQN on Hard maze (compare vs Q-Learning)

**Day 7 — Polish & Documentation**
- [ ] Write `README.md` with setup instructions
- [ ] Add docstrings to all major classes and functions
- [ ] Create Jupyter notebooks for easy exploration
- [ ] Record final training videos / screenshots
- [ ] Write personal learning summary (what did RL teach you?)

---

## Appendix A: Reinforcement Learning Taxonomy

```
Reinforcement Learning
│
├── Model-Based RL (has a model of environment)
│   ├── Dyna-Q
│   └── World Models
│
└── Model-Free RL (learns from experience only) ← WE ARE HERE
    │
    ├── Value-Based (learns value functions)
    │   ├── Q-Learning ← PHASE 3
    │   ├── SARSA
    │   └── DQN (Deep Q-Network) ← EXTENSION
    │
    └── Policy-Based (learns policy directly)
        ├── REINFORCE ← PHASE 4
        ├── Actor-Critic (A2C, A3C)
        └── PPO (Proximal Policy Optimization) ← Future
```

---

## Appendix B: Common Pitfalls & Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Reward too large | Training unstable, Q-values explode | Normalize rewards to [-1, 1] |
| Epsilon decays too fast | Agent stops exploring, gets stuck in local optima | Use slower decay (0.999 instead of 0.99) |
| Gamma too low | Agent ignores goal unless it's 1-2 steps away | Use γ ≥ 0.95 for goal-reaching tasks |
| Learning rate too high | Q-values oscillate, no convergence | Reduce alpha (try 0.01 instead of 0.1) |
| No step cost | Agent wanders randomly, finds goal by luck | Add -0.01 per step |
| Wall penalty too high | Agent freezes, refuses to move | Reduce wall penalty to -0.1 or -0.5 |
| Policy gradient — no baseline | Very slow convergence, high variance | Always subtract mean return as baseline |
| Episode too short | Agent can't reach goal even randomly | Increase max_steps (at least 3× optimal path length) |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Agent** | The learner / decision-maker |
| **Environment** | The world the agent interacts with |
| **State (s)** | A snapshot of the environment |
| **Action (a)** | What the agent can do |
| **Reward (r)** | Feedback signal from environment |
| **Policy (π)** | The agent's strategy: state → action |
| **Value Function V(s)** | Expected return from state s |
| **Q-Function Q(s,a)** | Expected return from state s taking action a |
| **Bellman Equation** | Recursive definition of Q-values |
| **TD Error** | Difference between estimated and target Q-value |
| **Epsilon-Greedy** | Explore randomly with probability ε, exploit otherwise |
| **Discount Factor (γ)** | Weight given to future vs immediate rewards |
| **Episode** | One complete run from start to terminal state |
| **Trajectory** | Sequence of (state, action, reward) in an episode |
| **Return (G)** | Total discounted reward over an episode |
| **Convergence** | When Q-values (or policy) stop changing significantly |
| **Exploration** | Trying new actions to discover better options |
| **Exploitation** | Using known good actions to maximize reward |
| **On-Policy** | Learning from the same policy being executed |
| **Off-Policy** | Learning optimal policy while executing a different one |
| **Experience Replay** | Storing and reusing past transitions for training |
| **Target Network** | Frozen copy of Q-network used for stable updates in DQN |

---

*End of Project Plan — Good luck, and remember: the agent doesn't give up after a few failed episodes. Neither should you! 🧭*

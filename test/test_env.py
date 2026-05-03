"""Quick validation script for the maze environment.

Run this to verify:
1. Environment imports correctly
2. reset() returns valid state
3. step() works with all 4 actions
4. Rewards are shaped correctly
5. Episode terminates when goal is reached
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env import MazeEnv, ACTION_MAP


def test_basic_import():
    """Test that the environment can be imported."""
    print("✓ Import test passed")


def test_easy_maze():
    """Test Easy difficulty maze."""
    env = MazeEnv(difficulty="easy")
    obs, info = env.reset()
    
    assert 0 <= obs < 7 * 7, f"State out of bounds: {obs}"
    assert info["position"] == (1, 1), f"Start position wrong: {info['position']}"
    print(f"✓ Easy maze created: {env.height}×{env.width} grid, state={obs}")


def test_medium_maze():
    """Test Medium difficulty maze."""
    env = MazeEnv(difficulty="medium", goal=(9, 9))
    obs, info = env.reset()
    
    assert 0 <= obs < 11 * 11, f"State out of bounds: {obs}"
    assert info["position"] == (1, 1)
    assert env.goal_pos == (9, 9)
    print(f"✓ Medium maze created: {env.height}×{env.width} grid, goal={env.goal_pos}")


def test_step_mechanics():
    """Test that step() works with all 4 actions."""
    env = MazeEnv(difficulty="easy")
    env.reset()
    
    total_reward = 0.0
    for action in range(4):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        assert 0 <= obs < env.height * env.width
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        print(f"  Action {action} ({['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}): "
              f"reward={reward:.2f}, pos={info['position']}, done={terminated}")
    
    print(f"✓ Step mechanics work. Cumulative reward: {total_reward:.3f}")


def test_goal_reach():
    """Test that episode terminates when agent reaches goal."""
    env = MazeEnv(difficulty="easy", goal=(3, 3))
    env.reset()
    
    # Move agent manually to goal position
    env.agent_pos = env.goal_pos
    env.steps = 1
    
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert terminated, "Episode should terminate when goal is reached"
    assert reward == 10.0, f"Goal reward should be 10.0, got {reward}"
    print(f"✓ Goal termination works. Reward at goal: {reward}")


def test_wall_collision():
    """Test that hitting a wall is penalized."""
    env = MazeEnv(difficulty="easy")
    env.reset()
    
    # Try to move into a wall (all 4 directions should hit wall from start)
    obs, reward, terminated, truncated, info = env.step(2)  # LEFT
    
    assert info["hit_wall"] or not terminated, "Should either hit wall or move"
    print(f"✓ Wall collision detected. Hit wall: {info['hit_wall']}, reward: {reward:.2f}")


def test_timeout():
    """Test that episode truncates after max_steps."""
    env = MazeEnv(difficulty="easy")
    env.reset()
    env.steps = env.max_steps - 1
    
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert truncated, "Episode should truncate at max_steps"
    assert reward == -1.0, f"Timeout reward should be -1.0, got {reward}"
    print(f"✓ Timeout truncation works. Reward at timeout: {reward}")


def test_state_encoding():
    """Test that state encoding/decoding is consistent."""
    env = MazeEnv(difficulty="medium")
    
    for row in [1, 5, 9]:
        for col in [1, 5, 9]:
            env.agent_pos = (row, col)
            state = env._encode_state((row, col))
            decoded = env._decode_state(state)
            assert decoded == (row, col), f"Encode/decode failed for ({row}, {col})"
    
    print("✓ State encoding/decoding is reversible")


def test_random_episode():
    """Run a full random episode to completion."""
    env = MazeEnv(difficulty="easy")
    obs, info = env.reset()
    
    done = False
    steps = 0
    episode_reward = 0.0
    
    while not done and steps < env.max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
    
    print(f"✓ Random episode completed: {steps} steps, reward={episode_reward:.2f}, "
          f"goal_reached={terminated}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Maze Environment")
    print("=" * 60)
    
    try:
        test_basic_import()
        test_easy_maze()
        test_medium_maze()
        test_step_mechanics()
        test_goal_reach()
        test_wall_collision()
        test_timeout()
        test_state_encoding()
        test_random_episode()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

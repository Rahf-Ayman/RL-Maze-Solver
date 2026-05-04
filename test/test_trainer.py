"""Quick smoke test for the training loop with Q-Learning."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import MazeEnv
from agents import QLearningAgent
from training import Trainer, get_q_learning_config, get_episode_config


def test_trainer_smoke():
    """Test that trainer runs for a few episodes without error."""
    print("=" * 60)
    print("Testing Trainer with Q-Learning Agent")
    print("=" * 60)
    
    # Setup
    env = MazeEnv(difficulty="easy")
    config = get_q_learning_config()
    agent = QLearningAgent(env=env, **config)
    trainer = Trainer(env=env, agent=agent, config=config)
    
    print(f"✓ Environment: {env.difficulty.title()} {env.height}×{env.width}")
    print(f"✓ Agent: Q-Learning (alpha={config['alpha']}, gamma={config['gamma']})")
    print(f"✓ Trainer initialized")
    
    # Train for a few episodes
    print("\nTraining for 50 episodes...")
    logger = trainer.train(n_episodes=50, verbose_interval=25)
    
    # Validate logger
    assert len(logger.rewards) == 50, f"Expected 50 rewards, got {len(logger.rewards)}"
    assert len(logger.steps) == 50, f"Expected 50 step counts, got {len(logger.steps)}"
    assert len(logger.success) == 50, f"Expected 50 success flags, got {len(logger.success)}"
    
    print(f"\n✓ Training completed: {len(logger.rewards)} episodes logged")
    print(f"  Avg reward: {sum(logger.rewards) / len(logger.rewards):.2f}")
    print(f"  Avg steps: {sum(logger.steps) / len(logger.steps):.1f}")
    print(f"  Success rate: {logger.success_rate(50):.1%}")
    
    # Test moving average
    ma = logger.moving_average(window=25)
    assert len(ma) == 26, f"Expected 26 MA values, got {len(ma)}"
    print(f"✓ Moving average computed: {len(ma)} values")
    
    # Test CSV export
    csv_path = logger.export_csv("test/training_smoke.csv")
    assert csv_path.exists(), f"CSV not written to {csv_path}"
    print(f"✓ Metrics exported to {csv_path}")
    
    print("=" * 60)
    print("✅ ALL TRAINER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_trainer_smoke()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    """
    Callback to track rewards during training.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = 0
        self._current_length = 0

    def _on_step(self):
        self._current_rewards += self.locals["rewards"][0]
        self._current_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_rewards)
            self.episode_lengths.append(self._current_length)
            self._current_rewards = 0
            self._current_length = 0
        return True


def train_ppo(timesteps=50000):
    """
    Train a PPO agent on CartPole-v1.
    Returns trained model and training metrics.
    """
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )

    callback = TrainingCallback()
    model.learn(total_timesteps=timesteps, callback=callback)

    env.close()
    return model, callback.episode_rewards, callback.episode_lengths


def evaluate(model, n_episodes=20):
    """
    Evaluate trained model over n episodes.
    Returns mean reward and all episode rewards.
    """
    env = gym.make("CartPole-v1")
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards), rewards
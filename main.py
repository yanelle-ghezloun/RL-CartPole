import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from src.train import train_ppo, evaluate

print("Training PPO agent on CartPole-v1...")
print("This will take 2-3 minutes...")

# Train
model, episode_rewards, episode_lengths = train_ppo(timesteps=50000)

print(f"Training done: {len(episode_rewards)} episodes completed")

# Evaluate
mean_reward, eval_rewards = evaluate(model, n_episodes=20)
print(f"Mean reward over 20 evaluation episodes: {mean_reward:.1f} / 500")

# smooth rewards for plot 
def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

smoothed = smooth(episode_rewards, window=20)

def run_episode(model):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)

    states = [obs]
    actions = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _ , terminated, truncated, _ = env.step(action)
        done = terminated or truncated 
        states.append(obs)
        actions.append(action)
    env.close()
    return np.array(states), np.array(actions)

# Run one episode for visualization 
states_ep, actions_ep = run_episode(model)
time = np.arange(len(states_ep)) * 0.02  # dt = 0.02s

# Plot 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reinforcement Learning (PPO) — CartPole-v1', fontsize=14)

# Learning curve
axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Episode reward')
axes[0, 0].plot(range(len(smoothed)), smoothed, color='blue',
                linewidth=2, label='Smoothed (window=20)')
axes[0, 0].axhline(y=500, color='green', linestyle='--',
                   linewidth=1.5, label='Max reward (500)')
axes[0, 0].set_title('Learning Curve')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Evaluation
axes[0, 1].bar(range(len(eval_rewards)), eval_rewards,
               color='steelblue', alpha=0.7)
axes[0, 1].axhline(y=mean_reward, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_reward:.1f}')
axes[0, 1].axhline(y=500, color='green', linestyle='--',
                   linewidth=1.5, label='Max (500)')
axes[0, 1].set_title('Evaluation — 20 Episodes')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Total Reward')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Pole angle during episode
axes[1, 0].plot(time, np.degrees(states_ep[:, 2]),
                'r-', linewidth=2, label='Pole angle')
axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Pole Angle — Trained Agent Episode')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Cart position during episode
axes[1, 1].plot(time, states_ep[:, 0],
                'b-', linewidth=2, label='Cart position')
axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Cart Position — Trained Agent Episode')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Position (m)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/rl_result.png', dpi=150)
plt.show()

print("Done : results/rl_result.png saved")

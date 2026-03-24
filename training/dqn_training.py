"""
DQN Training Script - Sista Health RL
========================================
Trains DQN using Stable Baselines 3 and runs
10 hyperparameter experiments automatically.
Saves models, reward curves, and results table.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SistaHealthEnv

os.makedirs("models/dqn", exist_ok=True)
os.makedirs("results", exist_ok=True)


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.current_rewards.append(reward)
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True


def evaluate_model(model, n_episodes=20):
    env = SistaHealthEnv()
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards), np.std(total_rewards)


EXPERIMENTS = [
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 5e-4,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-4,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.90, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 128, "buffer_size": 10000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 50000, "exploration_fraction": 0.3},
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.1},
    {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000, "exploration_fraction": 0.5},
]

TIMESTEPS = 50000


def run_experiments():
    results = []
    all_callbacks = []

    print("=" * 60)
    print("   DQN Hyperparameter Experiments - Sista Health RL")
    print("=" * 60)

    for i, params in enumerate(EXPERIMENTS):
        print(f"\n[Run {i+1}/10] {params}")

        env = Monitor(SistaHealthEnv())
        callback = RewardCallback()

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            exploration_fraction=params["exploration_fraction"],
            exploration_final_eps=0.05,
            verbose=0,
        )

        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        mean_reward, std_reward = evaluate_model(model)
        print(f"   -> Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        if i == 0 or mean_reward > max(r["mean_reward"] for r in results):
            model.save("models/dqn/best_dqn_model")
            print("   -> Saved as best model!")

        results.append({
            "Run": i + 1,
            "Learning Rate": params["learning_rate"],
            "Gamma": params["gamma"],
            "Batch Size": params["batch_size"],
            "Buffer Size": params["buffer_size"],
            "Exploration Fraction": params["exploration_fraction"],
            "mean_reward": mean_reward,
            "Mean Reward": round(mean_reward, 2),
            "Std Reward": round(std_reward, 2),
        })

        all_callbacks.append(callback)
        env.close()

    return results, all_callbacks


def plot_results(results, all_callbacks):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Hyperparameter Experiments - Sista Health", fontsize=14, fontweight="bold")

    runs   = [r["Run"] for r in results]
    means  = [r["Mean Reward"] for r in results]
    stds   = [r["Std Reward"] for r in results]
    colors = ["green" if m > 0 else "red" for m in means]

    ax = axes[0, 0]
    bars = ax.bar(runs, means, color=colors, alpha=0.8, yerr=stds, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Mean Reward per Run")
    ax.set_xlabel("Run #")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_xticks(runs)
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    ax = axes[0, 1]
    lrs = [r["Learning Rate"] for r in results]
    ax.scatter(lrs, means, color="steelblue", s=80, zorder=5)
    ax.set_xscale("log")
    ax.set_title("Learning Rate vs Mean Reward")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Mean Reward")
    for i, (x, y) in enumerate(zip(lrs, means)):
        ax.annotate(f"R{i+1}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax = axes[1, 0]
    gammas = [r["Gamma"] for r in results]
    ax.scatter(gammas, means, color="darkorange", s=80, zorder=5)
    ax.set_title("Gamma vs Mean Reward")
    ax.set_xlabel("Gamma (discount factor)")
    ax.set_ylabel("Mean Reward")

    ax = axes[1, 1]
    sorted_results = sorted(enumerate(results), key=lambda x: x[1]["Mean Reward"], reverse=True)
    top3 = sorted_results[:3]
    for rank, (idx, r) in enumerate(top3):
        cb = all_callbacks[idx]
        rewards = cb.episode_rewards
        if len(rewards) > 5:
            smoothed = np.convolve(rewards, np.ones(5) / 5, mode="valid")
            ax.plot(smoothed, label=f"Run {r['Run']} (LR={r['Learning Rate']})", linewidth=2)
    ax.set_title("Training Curves (Top 3 Runs)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/dqn_experiments.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/dqn_experiments.png")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    best_cb = all_callbacks[best_idx]
    if best_cb.episode_rewards:
        smoothed = np.convolve(
            best_cb.episode_rewards, np.ones(10) / 10, mode="valid"
        ) if len(best_cb.episode_rewards) > 10 else best_cb.episode_rewards
        ax2.plot(smoothed, color="steelblue", linewidth=2, label="Smoothed Reward")
        ax2.fill_between(range(len(smoothed)), smoothed, alpha=0.2, color="steelblue")
    ax2.set_title("DQN Best Run - Reward Curve")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Reward")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/dqn_reward_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/dqn_reward_curve.png")


def save_table(results):
    df = pd.DataFrame(results).drop(columns=["mean_reward"])
    df.to_csv("results/dqn_results.csv", index=False)
    print("Saved: results/dqn_results.csv")
    print("\n" + "=" * 70)
    print("DQN RESULTS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    results, callbacks = run_experiments()
    plot_results(results, callbacks)
    df = save_table(results)
    best = df.loc[df["Mean Reward"].idxmax()]
    print(f"\nBest Run: #{int(best['Run'])} - Mean Reward: {best['Mean Reward']}")
    print(f"Learning Rate: {best['Learning Rate']}, Gamma: {best['Gamma']}, Batch: {best['Batch Size']}")
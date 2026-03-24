"""
Policy Gradient Training — Sista Health RL
============================================
Trains PPO and REINFORCE using Stable Baselines 3.
Runs 10 hyperparameter experiments each automatically.
Saves models, reward curves, entropy curves, and results tables.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SistaHealthEnv

os.makedirs("models/pg", exist_ok=True)
os.makedirs("results", exist_ok=True)

TIMESTEPS = 50000


# ── Callback: Logs rewards + entropy ─────────────────────────────────────────
class PGCallback(BaseCallback):
 def __init__(self, verbose=0):
 super().__init__(verbose)
 self.episode_rewards = []
 self.current_rewards = []
 self.entropy_log = []

 def _on_step(self):
 reward = self.locals["rewards"][0]
 self.current_rewards.append(reward)
 done = self.locals["dones"][0]
 if done:
 self.episode_rewards.append(sum(self.current_rewards))
 self.current_rewards = []

 # Log entropy if available
 if hasattr(self.model, "policy") and hasattr(self.model.policy, "action_dist"):
 try:
 dist = self.model.policy.action_dist
 entropy = dist.entropy().mean().item()
 self.entropy_log.append(entropy)
 except Exception:
 pass
 return True


# ── Evaluate ──────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# PPO EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

PPO_EXPERIMENTS = [
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.90, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.01, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.05, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.0, "clip_range": 0.2},
 {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.3},
]


def run_ppo_experiments():
 results = []
 callbacks = []

 print("=" * 60)
 print(" PPO Hyperparameter Experiments — Sista Health RL")
 print("=" * 60)

 for i, params in enumerate(PPO_EXPERIMENTS):
 print(f"\n[PPO Run {i+1}/10] {params}")

 env = Monitor(SistaHealthEnv())
 callback = PGCallback()

 model = PPO(
 "MlpPolicy",
 env,
 learning_rate=params["learning_rate"],
 gamma=params["gamma"],
 n_steps=params["n_steps"],
 ent_coef=params["ent_coef"],
 clip_range=params["clip_range"],
 verbose=0,
 )

 model.learn(total_timesteps=TIMESTEPS, callback=callback)
 mean_reward, std_reward = evaluate_model(model)
 print(f" → Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

 if i == 0 or mean_reward > max(r["mean_reward"] for r in results):
 model.save("models/pg/best_ppo_model")
 print(" → Saved as best PPO model!")

 results.append({
 "Run": i + 1,
 "Learning Rate": params["learning_rate"],
 "Gamma": params["gamma"],
 "N Steps": params["n_steps"],
 "Entropy Coef": params["ent_coef"],
 "Clip Range": params["clip_range"],
 "mean_reward": mean_reward,
 "Mean Reward": round(mean_reward, 2),
 "Std Reward": round(std_reward, 2),
 })
 callbacks.append(callback)
 env.close()

 return results, callbacks


# ══════════════════════════════════════════════════════════════════════════════
# REINFORCE EXPERIMENTS
# SB3 does not have a native REINFORCE, so we use A2C with no critic
# (value_coef=0), which approximates vanilla policy gradient / REINFORCE
# ══════════════════════════════════════════════════════════════════════════════

REINFORCE_EXPERIMENTS = [
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.95, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.90, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.05, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.0, "vf_coef": 0.0},
 {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.25},
]


def run_reinforce_experiments():
 results = []
 callbacks = []

 print("\n" + "=" * 60)
 print(" REINFORCE Hyperparameter Experiments — Sista Health RL")
 print("=" * 60)

 for i, params in enumerate(REINFORCE_EXPERIMENTS):
 print(f"\n[REINFORCE Run {i+1}/10] {params}")

 env = Monitor(SistaHealthEnv())
 callback = PGCallback()

 # A2C with vf_coef=0 ≈ REINFORCE (pure policy gradient, no critic)
 model = A2C(
 "MlpPolicy",
 env,
 learning_rate=params["learning_rate"],
 gamma=params["gamma"],
 n_steps=params["n_steps"],
 ent_coef=params["ent_coef"],
 vf_coef=params["vf_coef"],
 verbose=0,
 )

 model.learn(total_timesteps=TIMESTEPS, callback=callback)
 mean_reward, std_reward = evaluate_model(model)
 print(f" → Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

 if i == 0 or mean_reward > max(r["mean_reward"] for r in results):
 model.save("models/pg/best_reinforce_model")
 print(" → Saved as best REINFORCE model!")

 results.append({
 "Run": i + 1,
 "Learning Rate": params["learning_rate"],
 "Gamma": params["gamma"],
 "N Steps": params["n_steps"],
 "Entropy Coef": params["ent_coef"],
 "VF Coef": params["vf_coef"],
 "mean_reward": mean_reward,
 "Mean Reward": round(mean_reward, 2),
 "Std Reward": round(std_reward, 2),
 })
 callbacks.append(callback)
 env.close()

 return results, callbacks


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_ppo_results(results, callbacks):
 fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 fig.suptitle("PPO Hyperparameter Experiments — Sista Health", fontsize=14, fontweight="bold")

 runs = [r["Run"] for r in results]
 means = [r["Mean Reward"] for r in results]
 colors = ["green" if m > 0 else "red" for m in means]

 # Bar chart
 ax = axes[0, 0]
 ax.bar(runs, means, color=colors, alpha=0.8)
 ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
 ax.set_title("Mean Reward per Run (PPO)")
 ax.set_xlabel("Run #")
 ax.set_ylabel("Mean Episode Reward")

 # Entropy coef effect
 ax = axes[0, 1]
 ent_coefs = [r["Entropy Coef"] for r in results]
 ax.scatter(ent_coefs, means, color="darkorange", s=80)
 ax.set_title("Entropy Coef vs Mean Reward")
 ax.set_xlabel("Entropy Coefficient")
 ax.set_ylabel("Mean Reward")

 # N_steps effect
 ax = axes[1, 0]
 n_steps = [r["N Steps"] for r in results]
 ax.scatter(n_steps, means, color="purple", s=80)
 ax.set_title("N Steps vs Mean Reward")
 ax.set_xlabel("N Steps")
 ax.set_ylabel("Mean Reward")

 # Entropy curves for top 3
 ax = axes[1, 1]
 sorted_idx = sorted(range(len(results)), key=lambda i: results[i]["Mean Reward"], reverse=True)
 for rank, idx in enumerate(sorted_idx[:3]):
 cb = callbacks[idx]
 if cb.entropy_log:
 ax.plot(cb.entropy_log, label=f"Run {results[idx]['Run']}", linewidth=1.5)
 ax.set_title("Entropy Curves (Top 3 PPO Runs)")
 ax.set_xlabel("Step")
 ax.set_ylabel("Policy Entropy")
 ax.legend(fontsize=8)

 plt.tight_layout()
 plt.savefig("results/ppo_experiments.png", dpi=150, bbox_inches="tight")
 plt.show()
 print(" Saved: results/ppo_experiments.png")

 # Reward curve for best PPO
 best_idx = np.argmax([r["Mean Reward"] for r in results])
 fig2, ax2 = plt.subplots(figsize=(10, 4))
 rewards = callbacks[best_idx].episode_rewards
 if len(rewards) > 10:
 smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
 ax2.plot(smoothed, color="darkorange", linewidth=2)
 ax2.fill_between(range(len(smoothed)), smoothed, alpha=0.2, color="darkorange")
 ax2.set_title("PPO Best Run — Cumulative Reward Curve")
 ax2.set_xlabel("Episode")
 ax2.set_ylabel("Episode Reward")
 ax2.grid(alpha=0.3)
 plt.tight_layout()
 plt.savefig("results/ppo_reward_curve.png", dpi=150, bbox_inches="tight")
 plt.show()
 print(" Saved: results/ppo_reward_curve.png")


def plot_reinforce_results(results, callbacks):
 fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 fig.suptitle("REINFORCE Hyperparameter Experiments — Sista Health", fontsize=14, fontweight="bold")

 runs = [r["Run"] for r in results]
 means = [r["Mean Reward"] for r in results]
 colors = ["green" if m > 0 else "red" for m in means]

 ax = axes[0, 0]
 ax.bar(runs, means, color=colors, alpha=0.8)
 ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
 ax.set_title("Mean Reward per Run (REINFORCE)")
 ax.set_xlabel("Run #")
 ax.set_ylabel("Mean Episode Reward")

 ax = axes[0, 1]
 lrs = [r["Learning Rate"] for r in results]
 ax.scatter(lrs, means, color="steelblue", s=80)
 ax.set_xscale("log")
 ax.set_title("Learning Rate vs Mean Reward")
 ax.set_xlabel("Learning Rate")
 ax.set_ylabel("Mean Reward")

 ax = axes[1, 0]
 ent_coefs = [r["Entropy Coef"] for r in results]
 ax.scatter(ent_coefs, means, color="green", s=80)
 ax.set_title("Entropy Coef vs Mean Reward")
 ax.set_xlabel("Entropy Coefficient")
 ax.set_ylabel("Mean Reward")

 # Training stability (variance across runs)
 ax = axes[1, 1]
 stds = [r["Std Reward"] for r in results]
 ax.bar(runs, stds, color="salmon", alpha=0.8)
 ax.set_title("Reward Std Dev per Run (Stability)")
 ax.set_xlabel("Run #")
 ax.set_ylabel("Std Deviation")

 plt.tight_layout()
 plt.savefig("results/reinforce_experiments.png", dpi=150, bbox_inches="tight")
 plt.show()
 print(" Saved: results/reinforce_experiments.png")

 # Reward curve
 best_idx = np.argmax([r["Mean Reward"] for r in results])
 fig2, ax2 = plt.subplots(figsize=(10, 4))
 rewards = callbacks[best_idx].episode_rewards
 if len(rewards) > 10:
 smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
 ax2.plot(smoothed, color="steelblue", linewidth=2)
 ax2.fill_between(range(len(smoothed)), smoothed, alpha=0.2, color="steelblue")
 ax2.set_title("REINFORCE Best Run — Cumulative Reward Curve")
 ax2.set_xlabel("Episode")
 ax2.set_ylabel("Episode Reward")
 ax2.grid(alpha=0.3)
 plt.tight_layout()
 plt.savefig("results/reinforce_reward_curve.png", dpi=150, bbox_inches="tight")
 plt.show()
 print(" Saved: results/reinforce_reward_curve.png")


def plot_comparison(dqn_results, ppo_results, rf_results):
 """All 3 algorithms compared side by side."""
 fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 fig.suptitle("Algorithm Comparison — Sista Health RL", fontsize=14, fontweight="bold")

 algo_data = [
 ("DQN", dqn_results, "steelblue"),
 ("PPO", ppo_results, "darkorange"),
 ("REINFORCE", rf_results, "green"),
 ]

 for ax, (name, results, color) in zip(axes, algo_data):
 means = [r["Mean Reward"] for r in results]
 runs = [r["Run"] for r in results]
 ax.plot(runs, means, color=color, marker="o", linewidth=2)
 ax.axhline(np.mean(means), color=color, linestyle="--", alpha=0.5,
 label=f"Avg: {np.mean(means):.1f}")
 ax.set_title(name)
 ax.set_xlabel("Run #")
 ax.set_ylabel("Mean Reward")
 ax.legend()
 ax.grid(alpha=0.3)

 plt.tight_layout()
 plt.savefig("results/algorithm_comparison.png", dpi=150, bbox_inches="tight")
 plt.show()
 print(" Saved: results/algorithm_comparison.png")


def save_tables(ppo_results, rf_results):
 ppo_df = pd.DataFrame(ppo_results).drop(columns=["mean_reward"])
 rf_df = pd.DataFrame(rf_results).drop(columns=["mean_reward"])

 ppo_df.to_csv("results/ppo_results.csv", index=False)
 rf_df.to_csv("results/reinforce_results.csv", index=False)

 print("\n Saved: results/ppo_results.csv")
 print(" Saved: results/reinforce_results.csv")
 print("\n── PPO Results ─────────────────────────────────────────")
 print(ppo_df.to_string(index=False))
 print("\n── REINFORCE Results ───────────────────────────────────")
 print(rf_df.to_string(index=False))
 return ppo_df, rf_df


if __name__ == "__main__":
 # Load DQN results if available for comparison plot
 dqn_dummy = [{"Run": i+1, "Mean Reward": 0} for i in range(10)]
 try:
 dqn_df = pd.read_csv("results/dqn_results.csv")
 dqn_dummy = dqn_df.to_dict("records")
 except FileNotFoundError:
 print(" DQN results not found. Run dqn_training.py first for comparison plot.")

 ppo_results, ppo_cbs = run_ppo_experiments()
 rf_results, rf_cbs = run_reinforce_experiments()

 plot_ppo_results(ppo_results, ppo_cbs)
 plot_reinforce_results(rf_results, rf_cbs)
 plot_comparison(dqn_dummy, ppo_results, rf_results)
 ppo_df, rf_df = save_tables(ppo_results, rf_results)

 best_ppo = ppo_df.loc[ppo_df["Mean Reward"].idxmax()]
 best_rf = rf_df.loc[rf_df["Mean Reward"].idxmax()]
 print(f"\n Best PPO Run: #{int(best_ppo['Run'])} — {best_ppo['Mean Reward']}")
 print(f" Best REINFORCE Run: #{int(best_rf['Run'])} — {best_rf['Mean Reward']}")

"""
main.py - Sista Health RL
===========================
Loads the best-performing model and runs it.

Usage:
    python main.py --algo ppo       # run best PPO (default)
    python main.py --algo dqn       # run best DQN
    python main.py --algo reinforce # run best REINFORCE
    python main.py --episodes 5     # number of episodes
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import SistaHealthEnv

MODEL_PATHS = {
    "ppo":       "models/pg/ppo/best_ppo_model.zip",
    "dqn":       "models/dqn/best_dqn_model.zip",
    "reinforce": "models/pg/reinforce/best_reinforce_model.zip",
}


def load_model(algo):
    if algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(MODEL_PATHS["dqn"])
    elif algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(MODEL_PATHS["ppo"])
    elif algo == "reinforce":
        from stable_baselines3 import A2C
        model = A2C.load(MODEL_PATHS["reinforce"])
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    print(f"Loaded {algo.upper()} model from {MODEL_PATHS[algo]}")
    return model


def run_episode(model, env, algo, episode_num):
    obs, info = env.reset()
    ep_reward = 0
    step = 0
    done = False

    print(f"\n{'='*55}")
    print(f"  Episode {episode_num} - Algorithm: {algo.upper()}")
    print(f"{'='*55}")
    print(f"  User: {info['language']} | {info['domain']} | {info['topic']}")
    print(f"  Urgency: {info['urgency']} | Literacy: {info['literacy']}")
    print(f"{'─'*55}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        action_name = env.ACTIONS[action]
        sign = "GOOD" if reward > 0 else ("OK" if reward == 0 else "BAD")

        print(f"  Step {step+1:2d} | Action: {action_name:20s} | "
              f"Reward: {reward:+.0f}  [{sign}]")

        ep_reward += reward
        step += 1
        done = terminated or truncated
        time.sleep(0.4)

    print(f"{'─'*55}")
    print(f"  Episode Reward: {ep_reward:.1f}")
    return ep_reward


def main():
    parser = argparse.ArgumentParser(description="Sista Health RL - Run Best Agent")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "reinforce"],
                        help="Algorithm to run (default: ppo)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to simulate (default: 5)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable pygame GUI (terminal only)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("   SISTA HEALTH - RL RESPONSE AGENT")
    print("   Voice-Enabled Multilingual Maternal Health")
    print("="*55)
    print(f"\n  Algorithm : {args.algo.upper()}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  GUI       : {'Disabled' if args.no_gui else 'Enabled'}")

    try:
        model = load_model(args.algo)
    except Exception as e:
        print(f"\nCould not load model: {e}")
        print("Have you run the training scripts yet?")
        print("Run: python training/dqn_training.py")
        print("Run: python training/pg_training.py")
        sys.exit(1)

    render_mode = None if args.no_gui else "human"
    env = SistaHealthEnv(render_mode=render_mode)
    all_rewards = []

    try:
        for ep in range(1, args.episodes + 1):
            ep_reward = run_episode(model, env, args.algo, ep)
            all_rewards.append(ep_reward)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        env.close()

    if all_rewards:
        print(f"\n{'='*55}")
        print(f"  SUMMARY ({args.algo.upper()})")
        print(f"{'='*55}")
        print(f"  Episodes run   : {len(all_rewards)}")
        print(f"  Mean reward    : {np.mean(all_rewards):.2f}")
        print(f"  Best episode   : {max(all_rewards):.2f}")
        print(f"  Worst episode  : {min(all_rewards):.2f}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
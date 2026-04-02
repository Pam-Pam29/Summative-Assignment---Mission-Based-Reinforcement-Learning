"""
random_agent.py - Sista Health RL
====================================
Demonstrates the Sista Health environment with a random agent.
No trained model is used - actions are sampled randomly.
This file shows the environment components and visualization
without any training involved.

Usage:
    python random_agent.py
    python random_agent.py --episodes 3
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import SistaHealthEnv


def run_random_episode(env, episode_num):
    obs, info = env.reset()
    ep_reward = 0
    step      = 0
    done      = False

    print(f"\n{'='*60}")
    print(f"  EPISODE {episode_num} - RANDOM AGENT (No Training)")
    print(f"{'='*60}")
    print(f"  User Profile:")
    print(f"    Language : {info['language']}")
    print(f"    Domain   : {info['domain']}")
    print(f"    Topic    : {info['topic']}")
    print(f"    Literacy : {info['literacy']}")   # FIX 1: removed info['urgency']
    print(f"{'─'*60}")
    print(f"  {'Step':<6} {'Action':<22} {'Reward':<10} {'Feedback'}")
    print(f"{'─'*60}")

    while not done:
        action = env.action_space.sample()  # RANDOM - no model
        obs, reward, terminated, truncated, info = env.step(action)

        action_name = env.ACTIONS[action]
        sign        = "GOOD" if reward > 0 else ("OK" if reward == 0 else "BAD")

        print(f"  {step+1:<6} {action_name:<22} {reward:+.0f}  [{sign:<4}]  {env.last_feedback}")

        ep_reward += reward
        step      += 1
        done       = terminated or truncated

    print(f"{'─'*60}")
    print(f"  Total Episode Reward (random): {ep_reward:.1f}")
    return ep_reward


def main():
    parser = argparse.ArgumentParser(
        description="Sista Health - Random Agent Demo"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run (default: 3)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SISTA HEALTH - RANDOM AGENT DEMONSTRATION")
    print("  Environment: SistaHealthEnv")
    print("  Agent: Random (no training, no model)")
    print("="*60)
    print("\n  Environment Overview:")
    # FIX 2: removed urgency from observation list
    print("  - Observation: language, domain, topic, literacy, step")
    # FIX 3: action 2 is Resource Link, not Emergency Referral
    print("  - Actions:     Text Response | Voice Note | Resource Link | Clarify")
    # FIX 5: no emergency termination, exactly 10 steps
    print("  - Episodes:    Max 10 steps")
    print("  - Purpose:     Shows environment dynamics without learning")

    env = SistaHealthEnv()
    all_rewards = []

    for ep in range(1, args.episodes + 1):
        ep_reward = run_random_episode(env, ep)
        all_rewards.append(ep_reward)

    env.close()

    print(f"\n{'='*60}")
    print(f"  RANDOM AGENT SUMMARY ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean reward  : {np.mean(all_rewards):.2f}")
    print(f"  Best episode : {max(all_rewards):.1f}")
    print(f"  Worst episode: {min(all_rewards):.1f}")
    print(f"{'─'*60}")
    print(f"  NOTE: Random agent scores near 0 because it has no")
    print(f"  policy. Trained agents score 103-114 on this environment.")  # FIX 4: matches actual results
    print(f"  Run: python main.py --algo ppo to see trained behaviour.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
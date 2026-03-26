# Sista Health — Mission-Based Reinforcement Learning

**Victoria Fakunle | ALU 2026**

A reinforcement-learning project that trains agents (DQN, PPO, and REINFORCE) to choose the optimal response strategy for **Sista Health**, a maternal and sexual-health assistant targeting women in West Africa.

---

## Project Overview

The agent must select the best communication channel for each user interaction:

| Action | Description |
|--------|-------------|
| `Text Response` | Written reply — best for high-literacy users |
| `Voice Note` | Audio message — best for low-literacy / Pidgin users |
| `Resource Link` | External resource — best for informational topics |
| `Clarify` | Ask a clarifying question — best for sensitive or ambiguous topics |

Rewards are shaped by how well the chosen action matches the user's language, literacy level, and health topic. The best possible action per step earns up to **+16 points**; a poor match can score **−4**.

---

## Repository Structure

```
project_root/
├── environment/
│   └── custom_env.py          # Custom Gymnasium environment (SistaHealthEnv)
├── training/
│   ├── dqn_training.py        # DQN: 10 experiments → saves best model
│   └── pg_training.py         # PPO + REINFORCE: 10 experiments each
├── models/
│   ├── dqn/                   # Saved DQN model weights
│   └── pg/
│       ├── ppo/               # Saved PPO model weights
│       └── reinforce/         # Saved REINFORCE (A2C) model weights
├── results/                   # Auto-generated plots and CSV files
├── main.py                    # Entry point — loads best model and runs demo
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train DQN

```bash
python training/dqn_training.py
```

### 3 — Train PPO & REINFORCE

```bash
python training/pg_training.py
```

### 4 — Run the best trained agent

```bash
# Auto-selects whichever model exists (defaults to PPO — best performing)
python main.py

# Force a specific algorithm
python main.py --algo ppo
python main.py --algo dqn
python main.py --algo reinforce

# Adjust number of evaluation episodes
python main.py --episodes 100

# Print step-by-step actions
python main.py --render
```

---

## Environment: `SistaHealthEnv`

| Component | Details |
|-----------|---------|
| **Observation** | `[language, domain, topic, literacy, step]` — 5 continuous features |
| **Language** | English (0), Yoruba (1), Pidgin (2) |
| **Domain** | Sexual Health (0), Maternal Health (1) |
| **Topic** | 9 topics (FGM Complications → Postpartum Care) |
| **Literacy** | Low (0), Medium (1), High (2) |
| **Action space** | Discrete(4) |
| **Episode length** | 10 steps |

---

## Algorithms

### DQN (Deep Q-Network)
- Off-policy, value-based
- Replay buffer + target network
- 10 experiments varying: learning rate, gamma, batch size, buffer size, exploration fraction

### PPO (Proximal Policy Optimisation) ✅ Best overall
- On-policy, actor-critic
- Clipped surrogate objective
- 10 experiments varying: learning rate, gamma, n_steps, entropy coefficient, clip range, GAE lambda

### REINFORCE (via A2C with `vf_coef=0`)
- On-policy, policy-gradient
- Monte-Carlo returns, optional value baseline
- 10 experiments varying: learning rate, gamma, n_steps, entropy coefficient, value function coefficient

---

## Results

All plots and CSV files are saved to `results/` after training.

### DQN Evaluation (50 episodes)

| Metric | Value |
|--------|-------|
| Mean reward | **109.40** |
| Std deviation | 32.27 |
| Min / Max | 70.0 / 160.0 |

### Scenario Breakdown — PPO Best Model (100 episodes)

| Scenario | Mean Reward | n |
|----------|-------------|---|
| English \| Maternal Health \| High | 100.0 | 3 |
| English \| Maternal Health \| Low | 140.0 | 8 |
| English \| Maternal Health \| Medium | 70.0 | 6 |
| English \| Sexual Health \| High | 100.0 | 5 |
| English \| Sexual Health \| Low | 140.0 | 8 |
| English \| Sexual Health \| Medium | 70.0 | 3 |
| Pidgin \| Maternal Health \| High | 100.0 | 8 |
| **Pidgin \| Maternal Health \| Low** | **160.0** | 5 |
| Pidgin \| Maternal Health \| Medium | 90.0 | 5 |
| Pidgin \| Sexual Health \| High | 100.0 | 5 |
| **Pidgin \| Sexual Health \| Low** | **160.0** | 2 |
| Pidgin \| Sexual Health \| Medium | 90.0 | 5 |
| Yoruba \| Maternal Health \| High | 100.0 | 8 |
| Yoruba \| Maternal Health \| Low | 140.0 | 7 |
| Yoruba \| Maternal Health \| Medium | 70.0 | 7 |
| Yoruba \| Sexual Health \| High | 100.0 | 6 |
| Yoruba \| Sexual Health \| Low | 140.0 | 7 |
| Yoruba \| Sexual Health \| Medium | 70.0 | 2 |

### Dominant Action by User Profile

| Profile | Action | Confidence |
|---------|--------|------------|
| English \| High literacy | Text Response | 100% |
| English \| Medium literacy | Voice Note | 100% |
| English \| Low literacy | Voice Note | 100% |
| Yoruba \| High literacy | Text Response | 100% |
| Yoruba \| Medium literacy | Voice Note | 100% |
| Yoruba \| Low literacy | Voice Note | 100% |
| Pidgin \| High literacy | Text Response | 100% |
| Pidgin \| Medium literacy | Voice Note | 100% |
| Pidgin \| Low literacy | Voice Note | 100% |

### Output Files

| File | Description |
|------|-------------|
| `dqn_experiments.png` | DQN hyperparameter comparison |
| `dqn_reward_curve.png` | Episode & cumulative reward for best DQN run |
| `dqn_objective_curve.png` | TD loss + Q-value proxy |
| `dqn_generalization.png` | Reward curves for all 10 DQN runs |
| `dqn_results.csv` | DQN results table |
| `ppo_experiments.png` | PPO hyperparameter comparison |
| `ppo_reward_curve.png` | Episode & cumulative reward for best PPO run |
| `ppo_entropy_curve.png` | Policy entropy + rolling std proxy |
| `ppo_results.csv` | PPO results table |
| `reinforce_experiments.png` | REINFORCE hyperparameter comparison |
| `reinforce_reward_curve.png` | REINFORCE reward curves |
| `reinforce_entropy_curve.png` | REINFORCE entropy curves |
| `reinforce_results.csv` | REINFORCE results table |
| `algorithm_comparison.png` | Cross-algorithm cumulative reward & convergence |
| `graph1_cumulative_rewards.png` | Combined cumulative reward chart (all three) |

---

## Key Findings

- **Pidgin + Low literacy** is the highest-reward scenario, consistently scoring **160/160** — the agent correctly selects Voice Note every step for a perfect episode.
- **Voice Note** is the dominant action for all low- and medium-literacy users across every language (100% confidence), confirming the reward shaping works as intended.
- **Text Response** is optimal only for high-literacy users across all three languages, also at 100% confidence.
- **Clarify** and **Resource Link** never emerged as dominant actions — the agent effectively reduced the policy to a 2-action decision driven purely by literacy level.
- Pidgin medium-literacy users score **90** vs 70 for English/Yoruba medium users, reflecting the +9 Pidgin bonus for Voice Note at medium literacy.
- **PPO** achieved the best overall performance and is the recommended model for deployment.

---

## Citation / Attribution

Built as part of the ALU 2026 summative assignment on mission-based reinforcement learning for African health-tech applications.

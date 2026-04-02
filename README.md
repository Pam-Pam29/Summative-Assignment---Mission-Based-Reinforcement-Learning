# 🏥 Sista Health — Mission-Based Reinforcement Learning

**Victoria Fakunle | ALU 2026**

> Training DQN, PPO, and REINFORCE agents to optimise response strategies for the **Sista Health** maternal and sexual-health assistant targeting women in West Africa.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Environment](#environment)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [DQN Results](#dqn-results)
6. [PPO Results](#ppo-results)
7. [REINFORCE Results](#reinforce-results)
8. [Algorithm Comparison](#algorithm-comparison)
9. [Generalisation Analysis](#generalisation-analysis)
10. [Key Findings](#key-findings)

---

## Project Overview

Sista Health is a maternal and sexual-health assistant designed for West African women. This project frames response-strategy selection as a reinforcement-learning problem: given a user's language, literacy level, and health topic, an agent must choose the most appropriate communication format.

Three algorithms were trained and evaluated — **DQN**, **PPO**, and **REINFORCE** — each with 10 hyperparameter configurations at 100,000 timesteps per run, totalling 30 training runs.

---

## Environment

**Class:** `SistaHealthEnv` (custom Gymnasium environment)

| Component | Details |
|-----------|---------|
| Observation space | `Box(5,)` — `[language, domain, topic, literacy, step]` |
| Action space | `Discrete(4)` |
| Episode length | 10 steps |
| Max reward per step | +16 (Voice Note + Pidgin + Low literacy) |
| Min reward per step | −4 (Text Response for Low literacy user) |

### State Features

| Feature | Values |
|---------|--------|
| Language | English (0), Yoruba (1), Pidgin (2) |
| Domain | Sexual Health (0), Maternal Health (1) |
| Topic | FGM Complications, VVF Causes, Cultural Barriers, Early Marriage, TBA Dangers, Contraception, STIs and HIV, Antenatal Care, Postpartum Care |
| Literacy | Low (0), Medium (1), High (2) |
| Step | 0 – 9 |

### Reward Structure

| Action | Condition | Reward |
|--------|-----------|--------|
| Text Response | High literacy | +10 |
| Text Response | Medium literacy | +5 |
| Text Response | Low literacy | −4 |
| Voice Note | Low literacy | +14 |
| Voice Note | Low literacy + Pidgin | **+16** |
| Voice Note | Medium literacy | +7 |
| Voice Note | Medium literacy + Pidgin | +9 |
| Voice Note | High literacy | +2 |
| Resource Link | Relevant topic + literacy ≥ 1 | +12 |
| Resource Link | Relevant topic + low literacy | +4 |
| Resource Link | Irrelevant topic | −2 |
| Clarify | Sensitive topic | +8 |
| Clarify | Start of session | +6 |
| Clarify | Mid-session, unnecessary | −3 |

---

## Repository Structure

```
project_root/
├── environment/
│   └── custom_env.py          # SistaHealthEnv (Gymnasium)
├── training/
│   ├── dqn_training.py        # DQN: 10 experiments → saves best model
│   └── pg_training.py         # PPO + REINFORCE: 10 experiments each
├── models/
│   ├── dqn/                   # best_dqn_model.zip  (Run 3)
│   └── pg/
│       ├── ppo/               # best_ppo_model.zip  (Run 1)
│       └── reinforce/         # best_reinforce_model.zip  (Run 6)
├── results/                   # All plots and CSVs
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train DQN (saves to models/dqn/)
python training/dqn_training.py

# 3. Train PPO + REINFORCE (saves to models/pg/)
python training/pg_training.py

# 4. Run the best agent (auto-selects best available model)
python main.py

# Optional flags
python main.py --algo ppo          # force algorithm
python main.py --episodes 100      # number of eval episodes
python main.py --render            # print step-by-step output
```

---

## DQN Results

**Best run:** Run 3 — Mean Reward **112.33 ± 30.62**
**Hyperparameters:** LR = 5e-3 · γ = 0.99 · Batch = 256 · Buffer = 50,000 · Explore fraction = 0.15 · Final ε = 0.02

### All 10 Runs

| Run | Learning Rate | γ | Batch | Buffer | Explore | Mean Reward | Std |
|-----|-------------|---|-------|--------|---------|-------------|-----|
| 1 | 1e-3 | 0.990 | 64 | 50,000 | 0.30 | 107.63 | 32.71 |
| 2 | 1e-4 | 0.990 | 64 | 100,000 | 0.30 | 30.67 | 60.82 |
| **3 ★** | **5e-3** | **0.990** | **256** | **50,000** | **0.15** | **112.33** | **30.62** |
| 4 | 1e-3 | 0.850 | 32 | 50,000 | 0.50 | 112.33 | 33.13 |
| 5 | 5e-4 | 0.995 | 256 | 100,000 | 0.20 | 105.67 | 22.61 |
| 6 | 1e-3 | 0.990 | 64 | 100,000 | 0.30 | 109.67 | 31.12 |
| 7 | 2e-3 | 0.970 | 32 | 50,000 | 0.10 | 108.67 | 32.01 |
| 8 | 5e-4 | 0.900 | 128 | 50,000 | 0.60 | 109.67 | 29.72 |
| 9 | 5e-4 | 0.990 | 128 | 100,000 | 0.25 | 100.07 | 30.44 |
| 10 | 2e-4 | 0.995 | 256 | 100,000 | 0.20 | 54.90 | 45.53 |

### Training Dynamics (Best Run #3)

- **TD Loss** peaked at ~14 around step 12,000, then decayed smoothly to near zero by ~40,000 steps and remained stable for the rest of training.
- **Q-value proxy** rose steadily from ~25 to ~115 over the first 2,000 episodes, then plateaued in the 105–125 range.
- **Cumulative reward** reached ~1,000,000 over 10,000 episodes — a clean upward curve confirming consistent improvement.
- **Convergence:** 9 of 10 runs exceeded the ≥40 threshold. Only Run 2 (LR = 1e-4) failed to converge reliably, collapsing to 30.67 with a std of 60.82.

### Hyperparameter Insights (DQN)

- **Learning rate** had the largest impact of any hyperparameter. LR = 5e-3 and 2e-3 both scored above 108; LR = 1e-4 collapsed to 30.67. Very low LRs (1e-4, 2e-4) severely underperformed.
- **Gamma:** Both γ = 0.85 and γ = 0.99 reached 112+ — discount factor had less influence than learning rate, provided it was not too extreme.
- **Batch size:** Larger batches (256) performed well, but small batches (32) also reached 112 with the right LR, suggesting batch size is not the primary lever here.
- **Buffer size:** 50,000 was sufficient for the best runs; larger buffers (100,000) did not consistently improve performance.
- **Exploration fraction:** Lower exploration (0.10–0.15) yielded slightly better final performance, but even high exploration (0.60) still reached ~110, indicating the environment is learnable despite noisy early exploration.

---

## PPO Results

**Best run:** Run 1 — Mean Reward **114.33 ± 28.60**
**Hyperparameters:** LR = 3e-4 · γ = 0.99 · n_steps = 2048 · ent_coef = 0.01 · clip = 0.20 · GAE λ = 0.95

### All 10 Runs

| Run | Learning Rate | γ | n_steps | ent_coef | clip | GAE λ | Mean Reward | Std |
|-----|-------------|---|---------|----------|------|-------|-------------|-----|
| **1 ★** | **3e-4** | **0.990** | **2048** | **0.010** | **0.20** | **0.95** | **114.33** | **28.60** |
| 2 | 1e-4 | 0.990 | 512 | 0.005 | 0.15 | 0.92 | 103.80 | 34.22 |
| 3 | 1e-3 | 0.990 | 4096 | 0.050 | 0.20 | 0.98 | 103.40 | 33.44 |
| 4 | 3e-4 | 0.850 | 512 | 0.010 | 0.20 | 0.90 | 104.67 | 31.91 |
| 5 | 3e-4 | 0.995 | 2048 | 0.010 | 0.30 | 0.98 | 107.67 | 28.01 |
| 6 | 5e-4 | 0.990 | 1024 | 0.000 | 0.10 | 0.95 | 102.07 | 28.28 |
| 7 | 2e-4 | 0.900 | 4096 | 0.050 | 0.25 | 0.92 | 106.13 | 27.10 |
| 8 | 3e-4 | 0.970 | 1024 | 0.020 | 0.15 | 0.95 | 107.33 | 29.66 |
| 9 | 5e-5 | 0.990 | 4096 | 0.010 | 0.20 | 0.99 | 106.80 | 31.06 |
| 10 | 2e-4 | 0.995 | 2048 | 0.020 | 0.25 | 0.98 | 105.40 | 34.69 |

### Training Dynamics (Best Run #1)

- **Policy entropy** started at ~1.4 (near-uniform distribution) and decayed continuously to ~0.1 by the end of training — a healthy sign of policy specialisation.
- **Entropy proxy (rolling std)** stabilised in the 25–40 range after an initial ramp-up, reflecting maintained but bounded exploration throughout training.
- **Cumulative reward** reached ~1,000,000 over 10,000 episodes, matching DQN's trajectory.
- The smoothed reward crossed ~100 within ~2,000 episodes and held steadily above that mark for the rest of training — more stable convergence than DQN or REINFORCE.

### Hyperparameter Insights (PPO)

- **PPO was the most stable algorithm overall**, with only a 12.26-point spread across all 10 runs (102.07–114.33), versus DQN's 81.66-point spread.
- **Learning rate:** The baseline 3e-4 was optimal. High LR = 1e-3 (Run 3) underperformed despite using a long rollout (4096 steps).
- **Entropy coefficient:** Zero entropy (Run 6) produced the lowest score (102.07). A small amount of entropy regularisation (0.01) was beneficial.
- **Clip range:** Standard 0.20 outperformed tighter (0.10) and wider (0.30) ranges at equivalent LRs.
- **n_steps:** 2048 was the sweet spot. Both shorter (512) and longer (4096) rollouts slightly underperformed.
- **Gamma:** γ = 0.99 dominated; lower gamma (0.85, 0.90) reduced performance, confirming the value of long-horizon planning in this 10-step episodic environment.

---

## REINFORCE Results

**Best run:** Run 6 — Mean Reward **103.47 ± 29.85**
**Hyperparameters:** LR = 3e-4 · γ = 0.99 · n_steps = 30 · ent_coef = 0.02 · vf_coef = 0.50 · max_grad_norm = 0.5

### All 10 Runs

| Run | Learning Rate | γ | n_steps | ent_coef | vf_coef | Mean Reward | Std |
|-----|-------------|---|---------|----------|---------|-------------|-----|
| 1 | 7e-4 | 0.990 | 20 | 0.010 | 0.00 | 72.33 | 48.49 |
| 2 | 1e-4 | 0.990 | 50 | 0.000 | 0.00 | 77.33 | 48.78 |
| 3 | 1e-3 | 0.990 | 10 | 0.050 | 0.00 | 79.67 | 54.07 |
| 4 | 7e-4 | 0.850 | 50 | 0.010 | 0.00 | 86.33 | 45.72 |
| 5 | 5e-4 | 0.995 | 10 | 0.010 | 0.25 | 100.93 | 26.26 |
| **6 ★** | **3e-4** | **0.990** | **30** | **0.020** | **0.50** | **103.47** | **29.85** |
| 7 | 1e-3 | 0.970 | 50 | 0.010 | 0.10 | 84.00 | 50.57 |
| 8 | 7e-4 | 0.900 | 20 | 0.050 | 0.00 | 77.00 | 52.23 |
| 9 | 5e-5 | 0.990 | 50 | 0.005 | 0.10 | 102.67 | 48.85 |
| 10 | 5e-4 | 0.995 | 50 | 0.020 | 0.10 | 97.67 | 26.67 |

### Training Dynamics (Best Run #6)

- **Policy entropy** declined from ~1.4 to near zero — a more gradual and noisier decay than PPO, reflecting REINFORCE's higher-variance policy gradient updates.
- **Entropy proxy** stabilised around 25–40 with more oscillation than PPO throughout training.
- **Cumulative reward** reached ~850,000 over 10,000 episodes — below DQN and PPO, consistent with the lower mean reward.
- The smoothed reward crossed ~100 only around episodes 5,000–6,000, roughly twice as slow to converge as the other two algorithms.

### Hyperparameter Insights (REINFORCE)

- **Value function baseline (vf_coef) was the single most important factor.** Runs without a baseline (vf_coef = 0.00: Runs 1–4, 8) averaged ~78 mean reward; runs with a baseline (vf_coef ≥ 0.10: Runs 5, 6, 7, 9, 10) averaged ~98 — a ~20-point improvement from one parameter change.
- **High learning rate was harmful without a baseline.** Runs 3 and 7 (LR = 1e-3) both scored below 85 and had standard deviations above 50, confirming instability.
- **n_steps:** Medium rollouts (30–50 steps) outperformed very short ones (10 steps) when paired with a baseline.
- **Gamma:** γ = 0.995 worked well in Runs 5 and 10 when combined with a strong baseline; γ = 0.85 or 0.90 was harmful.
- **REINFORCE had the highest average standard deviation (43.86) across all runs**, compared to PPO (30.61) and DQN (34.89), confirming the algorithm's inherently higher variance.

---

## Algorithm Comparison

### Summary Table

| Metric | DQN | PPO | REINFORCE |
|--------|-----|-----|-----------|
| **Best mean reward** | 112.33 | **114.33** | 103.47 |
| **Best run** | Run 3 | Run 1 | Run 6 |
| **Avg std dev (all runs)** | 34.89 | **30.61** | 43.86 |
| **Worst run mean reward** | 30.67 | 102.07 | 72.33 |
| **Run-to-run spread** | 81.66 | **12.26** | 31.14 |
| **100-ep eval mean** | 107.1 | **113.1** | 109.8 |
| **100-ep cumulative reward** | 10,710 | **11,310** | 10,982 |

### Convergence (100-episode evaluation)

- **PPO** achieved the highest smoothed mean reward across 100 evaluation episodes (mean = 113.1) and the highest cumulative total (11,310).
- **REINFORCE** came second in 100-episode evaluation (mean = 109.8, cumulative = 10,982), slightly outperforming DQN despite a lower best-run score — its best model generalised more consistently over many episodes.
- **DQN** finished third in 100-episode evaluation (mean = 107.1, cumulative = 10,710), likely due to higher episode-to-episode variance.
- All three algorithms produced broadly overlapping convergence curves once warmed up, confirming the environment is solvable by all three approaches.

### Stability

PPO was the most stable algorithm by a significant margin — a run-to-run spread of just 12.26 points versus 81.66 for DQN and 31.14 for REINFORCE. This makes PPO the recommended choice for deployment: hyperparameter sensitivity is low and performance is consistent across configurations.

---

## Generalisation Analysis

### DQN — Reward Heatmap (by Language × Literacy)

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 106 | 111 | 101 |
| Medium | 106 | **115** | **116** |
| High | 98 | 86 | 110 |

- Strongest profiles: Medium Pidgin (116) and Medium Yoruba (115) — where Voice Note + Pidgin rewards dominate.
- Weakest profile: High-literacy Yoruba (86) — the agent occasionally chose Voice Note over the optimal Text Response.

### PPO — Reward Heatmap (by Language × Literacy)

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 108 | 111 | 108 |
| Medium | 109 | 110 | **117** |
| High | 98 | 92 | 114 |

- Strongest profile: Medium Pidgin (117), correctly exploiting Voice Note + Pidgin.
- All cells above 90 — PPO has the most uniform coverage across user profiles.
- Weakest profile: High-literacy Yoruba (92).

### REINFORCE — Reward Heatmap (by Language × Literacy)

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 91 | 89 | 105 |
| Medium | 89 | 104 | 96 |
| High | **111** | **111** | 106 |

- REINFORCE shows an inverted pattern compared to the other two algorithms: **strongest on high-literacy users** (111 for English and Yoruba), weakest on low/medium-literacy users.
- Low-literacy Yoruba was its weakest cell (89), suggesting it did not fully learn the Voice Note + Pidgin pairing for this segment.
- The high-literacy strength likely reflects a partially Text-Response-heavy policy, which happens to be optimal for high-literacy users but suboptimal for others.

---

## Key Findings

1. **PPO is the recommended algorithm** — it achieved the highest best-run reward (114.33), the highest 100-episode evaluation mean (113.1), and the lowest run-to-run variance (12.26 points). It is the most robust to hyperparameter choice.

2. **Voice Note + Pidgin is the highest-value action (+16)**, and all three algorithms learned to exploit it for low/medium-literacy Pidgin speakers, validating the reward structure.

3. **A value function baseline is essential for REINFORCE.** Without it (vf_coef = 0), mean rewards hovered at 72–86. With a moderate baseline (vf_coef ≥ 0.25), performance jumped to 100+, a ~20-point improvement from a single parameter change.

4. **DQN is the most sensitive to learning rate.** The gap between LR = 5e-3 (112.33) and LR = 1e-4 (30.67) is 81.66 points — the largest single-hyperparameter sensitivity observed across all 30 runs.

5. **High-literacy Yoruba is the hardest profile for all three algorithms**, scoring 86 (DQN), 92 (PPO), and 111 (REINFORCE). The optimal action (Text Response) competes with the high-reward Voice + Pidgin combination in the agent's value estimates.

6. **DQN converges fastest.** The TD loss curve stabilises by ~40,000 steps and the Q-value proxy plateaus clearly. Both PPO and REINFORCE show smoother but slower convergence due to on-policy update mechanics.

---

## Results Files

| File | Description |
|------|-------------|
| `dqn_results.csv` | All 10 DQN runs with hyperparameters and rewards |
| `ppo_results.csv` | All 10 PPO runs |
| `reinforce_results.csv` | All 10 REINFORCE runs |
| `dqn_experiments.png` | DQN hyperparameter scatter plots |
| `dqn_reward_curve.png` | Episode + cumulative reward — Best Run #3 |
| `dqn_objective_curve.png` | TD loss + Q-value proxy — Best Run #3 |
| `dqn_generalization.png` | DQN reward heatmap by user profile |
| `dqn_convergence.png` | DQN convergence analysis across all 10 runs |
| `ppo_experiments.png` | PPO hyperparameter scatter plots |
| `ppo_reward_curve.png` | Episode + cumulative reward — Best Run #1 |
| `ppo_entropy_curve.png` | PPO policy entropy decay |
| `reinforce_experiments.png` | REINFORCE hyperparameter scatter plots |
| `reinforce_reward_curve.png` | Episode + cumulative reward — Best Run #6 |
| `reinforce_entropy_curve.png` | REINFORCE policy entropy decay |
| `pg_generalization.png` | PPO + REINFORCE reward heatmaps side-by-side |
| `algorithm_comparison.png` | Cross-algorithm bar charts, box plots, and convergence |
| `graph1_cumulative_rewards.png` | 100-episode cumulative reward for all three algorithms |

---

## Dependencies

```
gymnasium>=0.29.0
stable-baselines3[extra]>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.2.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

*Sista Health RL · Victoria Fakunle · ALU 2026*

#  Sista Health — Mission-Based Reinforcement Learning


> Framing maternal and sexual health response strategy as a reinforcement-learning problem: given a West African woman's language, literacy level, and health query, train an agent to select the most appropriate communication format — voice note, text, emergency referral, or clarification.

---

##  Table of Contents

1. [Project Overview](#-project-overview)
2. [Environment Design](#-environment-design)
3. [Repository Structure](#-repository-structure)
4. [Quick Start](#-quick-start)
5. [DQN Results](#-dqn-results)
6. [PPO Results](#-ppo-results)
7. [REINFORCE Results](#-reinforce-results)
8. [Algorithm Comparison](#-algorithm-comparison)
9. [Generalisation Analysis](#-generalisation-analysis)
10. [Key Findings](#-key-findings)
11. [Dependencies](#-dependencies)

---

##  Project Overview

Sista Health is a voice-enabled multilingual maternal and sexual health assistant for Nigerian women, delivered via WhatsApp. This capstone project applies reinforcement learning to simulate an agent that selects the appropriate **response modality** based on user context — choosing between text responses, voice notes, emergency referrals, or clarification questions depending on what will best serve each user.

**Three algorithms were trained and evaluated:**

| Algorithm | Type | Best Mean Reward | Stability (Spread) |
|-----------|------|------------------|--------------------|
| **PPO**  | On-policy (Actor-Critic) | **114.33** | **12.26 pts** |
| DQN | Off-policy (Value-based) | 112.33 | 81.66 pts |
| REINFORCE | On-policy (Policy Gradient) | 103.47 | 31.14 pts |

Each algorithm was trained for **100,000 timesteps** across **10 hyperparameter configurations**, totalling **30 training runs**.

A random agent baseline was also evaluated for comparison — all three trained algorithms significantly outperformed it.

---

##  Environment Design

**Class:** `SistaHealthEnv` (Custom Gymnasium Environment)

### Observation & Action Space

| Component | Details |
|-----------|---------|
| Observation space | `Box(6,)` — `[language, domain, topic, urgency, literacy, step]` |
| Action space | `Discrete(4)` — Text / Voice Note / Resource Link / Clarify |
| Episode length | 10 steps |
| Max reward per step | +16 (Voice Note + Pidgin + Low literacy) |
| Min reward per step | −4 (Text Response for Low-literacy user) |

### State Features

| Feature | Values |
|---------|--------|
| Language | English (0), Yoruba (1), Pidgin (2) |
| Domain | Sexual Health (0), Maternal Health (1) |
| Topic | FGM Complications, VVF Causes, Cultural Barriers, Early Marriage, TBA Dangers, Contraception, STIs and HIV, Antenatal Care, Postpartum Care |
| Literacy | Low (0), Medium (1), High (2) |
| Step | 0–9 (position within episode) |

### Reward Structure

The reward function is designed to encode domain expertise: voice notes are best for low-literacy users, text for high-literacy users, and Pidgin speakers receive a bonus when matched with voice notes.

| Action | Condition | Reward |
|--------|-----------|--------|
| Text Response | High literacy | +10 |
| Text Response | Medium literacy | +5 |
| Text Response | Low literacy | −4 |
| Voice Note | Low literacy | +14 |
| **Voice Note** | **Low literacy + Pidgin** | **+16** |
| Voice Note | Medium literacy | +7 |
| Voice Note | Medium literacy + Pidgin | +9 |
| Voice Note | High literacy | +2 |
| Resource Link | Relevant topic + literacy ≥ 1 | +12 |
| Resource Link | Relevant topic + low literacy | +4 |
| Resource Link | Irrelevant topic | −2 |
| Clarify | Sensitive topic | +8 |
| Clarify | Start of session | +6 |
| Clarify | Mid-session, unnecessary | −3 |

The highest possible reward (+16) occurs when a Pidgin-speaking, low-literacy user receives a voice note — reflecting the real-world priority of reaching the most linguistically and educationally marginalised users in the most accessible format.

---

##  Repository Structure

```
sista_health_rl/
├── environment/
│   └── custom_env.py          # SistaHealthEnv (Gymnasium)
├── training/
│   ├── dqn_training.py        # DQN: 10 experiments → saves best model
│   └── pg_training.py         # PPO + REINFORCE: 10 experiments each
├── models/
│   ├── dqn/
│   │   └── best_dqn_model.zip     # Best DQN model (Run 3)
│   └── pg/
│       ├── ppo/
│       │   └── best_ppo_model.zip     # Best PPO model (Run 1)
│       └── reinforce/
│           └── best_reinforce_model.zip   # Best REINFORCE model (Run 6)
├── results/                   # All plots and CSVs
│   ├── dqn_results.csv
│   ├── ppo_results.csv
│   ├── reinforce_results.csv
│   └── *.png                  # All visualisation outputs
├── main.py                    # Entry point — runs best available agent
├── random_agent.py            # Baseline random agent
├── api.py                     # FastAPI REST endpoint
├── frontend_demo.html         # Browser-based demo
├── rendering.py               # WhatsApp-style pygame visualisation
├── requirements.txt
└── README.md
```

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train DQN (saves best model to models/dqn/)
python training/dqn_training.py

# 3. Train PPO + REINFORCE (saves best models to models/pg/)
python training/pg_training.py

# 4. Run the best agent (auto-selects best available model)
python main.py

# Optional flags
python main.py --algo ppo          # force a specific algorithm
python main.py --algo dqn
python main.py --algo reinforce
python main.py --episodes 100      # number of evaluation episodes
python main.py --render            # print step-by-step output

# 5. Launch the REST API
uvicorn api:app --reload

# 6. Open the browser demo
open frontend_demo.html
```

> **Note:** `main.py` loads a pre-trained model from `models/`. Run the training scripts first, or place your own `.zip` model files in the appropriate directories before running the entry point.

---

##  DQN Results

**Best run:** Run 3 — Mean Reward **112.33 ± 30.62**

**Best hyperparameters:** LR = 5e-3 · γ = 0.99 · Batch = 256 · Buffer = 50,000 · Explore fraction = 0.15 · Final ε = 0.02

### All 10 Runs

| Run | Learning Rate | γ | Batch | Buffer | Explore | **Mean Reward** | Std |
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

### Hyperparameter Scatter Plots

![DQN Hyperparameter Comparison](results/dqn_experiments.png)

The scatter plots confirm that **learning rate is the dominant hyperparameter for DQN** — a gap of 81.66 points separates the best (LR = 5e-3, reward = 112.33) from the worst (LR = 1e-4, reward = 30.67). Gamma, batch size, buffer size, and exploration fraction each show far weaker individual effects.

### Training Dynamics (Best Run #3)

![DQN Reward Curve](results/dqn_reward_curve.png)

![DQN Objective Curve](results/dqn_objective_curve.png)

- **TD Loss** peaked at ~14 around step 12,000, then decayed smoothly to near zero by ~40,000 steps.
- **Q-value proxy** rose steadily from ~25 to ~115 over the first 2,000 episodes, then plateaued in the 105–125 range — a clear sign of convergence.
- **Cumulative reward** reached ~1,000,000 over 10,000 episodes in a clean upward curve confirming consistent improvement.

### Convergence Analysis

![DQN Convergence](results/dqn_convergence.png)

9 of 10 DQN runs exceeded the ≥40 convergence threshold. Only Run 2 (LR = 1e-4) failed to converge reliably, collapsing to 30.67 with a standard deviation of 60.82.

### Hyperparameter Insights

- **Learning rate** had the largest single impact of any hyperparameter across all 30 runs. LR = 5e-3 and 2e-3 both scored above 108; LR = 1e-4 and 2e-4 both collapsed below 55.
- **Gamma:** Both γ = 0.85 and γ = 0.99 reached 112+ — discount factor had less influence than learning rate, provided it was not extreme.
- **Batch size:** Larger batches (256) performed well, but small batches (32) also reached 112 with the right LR.
- **Buffer size:** 50,000 was sufficient; larger buffers (100,000) did not consistently improve performance.
- **Exploration fraction:** Lower exploration (0.10–0.15) yielded slightly better final performance, but even high exploration (0.60) still reached ~110.

---

##  PPO Results

**Best run:** Run 1 — Mean Reward **114.33 ± 28.60**

**Best hyperparameters:** LR = 3e-4 · γ = 0.99 · n_steps = 2048 · ent_coef = 0.01 · clip = 0.20 · GAE λ = 0.95

### All 10 Runs

| Run | Learning Rate | γ | n_steps | ent_coef | clip | GAE λ | **Mean Reward** | Std |
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

### Hyperparameter Scatter Plots

![PPO Hyperparameter Comparison](results/ppo_experiments.png)

PPO's scatter plots show a remarkably tight cluster — all runs sit between 102 and 114, confirming the algorithm's low sensitivity to hyperparameter choice. The clearest signal is gamma: γ = 0.99 consistently outperforms lower values.

### Training Dynamics (Best Run #1)

![PPO Reward Curve](results/ppo_reward_curve.png)

![PPO Entropy Curves](results/ppo_entropy_curve.png)

- **Policy entropy** started at ~1.4 (near-uniform) and decayed to ~0.1 by end of training — healthy policy specialisation.
- **Entropy proxy (rolling std)** stabilised in the 25–40 range after an initial ramp, reflecting maintained but bounded exploration.
- **Cumulative reward** reached ~1,000,000 over 10,000 episodes.
- The smoothed reward crossed ~100 within ~2,000 episodes and held steadily above it — more stable convergence than DQN or REINFORCE.

### Hyperparameter Insights

- **PPO was the most stable algorithm overall**, with only a 12.26-point spread across all 10 runs (102.07–114.33), versus DQN's 81.66-point spread.
- **Learning rate:** The baseline 3e-4 was optimal. High LR = 1e-3 underperformed despite a long rollout (4096 steps).
- **Entropy coefficient:** Zero entropy (Run 6) produced the lowest score (102.07). A small amount of entropy regularisation (0.01) was beneficial.
- **Clip range:** Standard 0.20 outperformed tighter (0.10) and wider (0.30) ranges at equivalent LRs.
- **n_steps:** 2048 was the sweet spot. Both shorter (512) and longer (4096) rollouts slightly underperformed.
- **Gamma:** γ = 0.99 dominated; lower gamma (0.85, 0.90) reduced performance, confirming the value of long-horizon planning in this 10-step episodic environment.

---

##  REINFORCE Results

**Best run:** Run 6 — Mean Reward **103.47 ± 29.85**

**Best hyperparameters:** LR = 3e-4 · γ = 0.99 · n_steps = 30 · ent_coef = 0.02 · vf_coef = 0.50 · max_grad_norm = 0.5

### All 10 Runs

| Run | Learning Rate | γ | n_steps | ent_coef | vf_coef | **Mean Reward** | Std |
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

### Hyperparameter Scatter Plots

![REINFORCE Hyperparameter Comparison](results/reinforce_experiments.png)

The VF Coef scatter plot tells the clearest story in the entire experiment: all runs without a value function baseline (vf_coef = 0.00) sit at the bottom of the chart (72–86), while all runs with a baseline (vf_coef ≥ 0.10) sit at the top (97–103).

### Training Dynamics (Best Run #6)

![REINFORCE Reward Curve](results/reinforce_reward_curve.png)

![REINFORCE Entropy Curves](results/reinforce_entropy_curve.png)

- **Policy entropy** declined from ~1.4 to near zero — a noisier and slower decay than PPO, reflecting REINFORCE's higher-variance updates.
- **Entropy proxy** stabilised around 25–40 with more oscillation than PPO throughout training.
- **Cumulative reward** reached ~850,000 over 10,000 episodes — below DQN and PPO.
- The smoothed reward crossed ~100 only around episodes 5,000–6,000 — roughly twice as slow to converge as the other two algorithms.

### Hyperparameter Insights

- **The value function baseline (vf_coef) was the single most impactful parameter.** Runs without a baseline (Runs 1–4, 8) averaged ~78; runs with a baseline (Runs 5, 6, 7, 9, 10) averaged ~98 — a ~20-point improvement from one parameter change.
- **High learning rate without a baseline caused collapse.** Runs 3 and 7 (LR = 1e-3) both scored below 85 with standard deviations above 50.
- **n_steps:** Medium rollouts (30–50 steps) outperformed very short ones (10 steps) when paired with a baseline.
- **Gamma:** γ = 0.995 worked well in Runs 5 and 10 when combined with a strong baseline; lower gamma (0.85, 0.90) consistently harmed performance.
- **REINFORCE had the highest average standard deviation (43.86)** across all runs, compared to PPO (30.61) and DQN (34.89).

---

##  Algorithm Comparison

### Summary Table

| Metric | DQN | PPO | REINFORCE |
|--------|-----|-----|-----------|
| Best mean reward | 112.33 | **114.33** | 103.47 |
| Best run | Run 3 | Run 1 | Run 6 |
| Avg std dev (all runs) | 34.89 | **30.61** | 43.86 |
| Worst run mean reward | 30.67 | **102.07** | 72.33 |
| Run-to-run spread | 81.66 | **12.26** | 31.14 |
| 100-ep eval mean | 107.1 | **113.1** | 109.8 |
| 100-ep cumulative reward | 10,710 | **11,310** | 10,982 |

### Visualisations

![Algorithm Comparison](results/algorithm_comparison.png)

The four-panel comparison shows:
- **(Top left) Mean Reward per Run:** PPO's green line stays tightly clustered; DQN's blue line shows the dramatic Run 2 collapse to ~30; REINFORCE's gold line shows a wide band of variance across runs.
- **(Top right) Best Run Reward:** PPO edges ahead with 114.3 vs DQN's 112.3 and REINFORCE's 103.5.
- **(Bottom left) Avg Std Dev:** PPO is measurably the most stable algorithm (lower bar = better).
- **(Bottom right) Reward Distribution:** PPO's box plot is extremely compact — tiny IQR, no outliers on the downside. DQN shows two severe low outliers (Runs 2 and 10).

![100-Episode Cumulative Reward + Convergence](results/graph1_cumulative_rewards.png)

The 100-episode evaluation confirms PPO as the top performer (Final: 11,310), followed by REINFORCE (10,982) and DQN (10,710). The smoothed convergence curves (bottom panel) show all three algorithms entering a broadly similar performance band after warm-up, confirming the environment is solvable by all three approaches — PPO simply does so more reliably.

### Why PPO Wins

PPO's advantage in this environment comes from three structural properties:

1. **Clipped surrogate objective** prevents catastrophic policy updates — no single bad gradient step can collapse performance the way a bad TD-learning update can collapse DQN.
2. **On-policy updates with GAE** provide a lower-variance estimate of advantage than REINFORCE's Monte Carlo returns, particularly valuable in a short 10-step episodic environment.
3. **Entropy regularisation** keeps the policy from prematurely collapsing onto a single action, allowing it to maintain coverage across all nine user profile combinations.

---

##  Generalisation Analysis

Generalisation was tested by holding out evaluation episodes and stratifying results by language × literacy profile. A well-generalised agent should score consistently across all nine cells.

### DQN — Reward Heatmap

![DQN Generalisation](results/dqn_generalization.png)

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 106 | 111 | 101 |
| Medium | 106 | **115** | **116** |
| High | 98 | 86 | 110 |

DQN's weakest cell is **High-literacy Yoruba (86)** — the agent occasionally chose Voice Note over Text Response, likely because Yoruba co-activates the voice note value function. All other cells are ≥100.

### PPO & REINFORCE — Reward Heatmaps

![PG Generalisation](results/pg_generalization.png)

**PPO:**

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 108 | 111 | 108 |
| Medium | 109 | 110 | **117** |
| High | 98 | 92 | 114 |

PPO has the most uniform coverage — all cells above 92, with Medium Pidgin reaching 117. Weakest: High-literacy Yoruba (92), the same failure mode as DQN.

**REINFORCE:**

| Literacy | English | Yoruba | Pidgin |
|----------|---------|--------|--------|
| Low | 91 | 89 | 105 |
| Medium | 89 | 104 | 96 |
| High | **111** | **111** | 106 |

REINFORCE shows a striking **inverted pattern** compared to the other two algorithms — it excels on high-literacy users (111 for English and Yoruba) but underperforms on low and medium-literacy users. This suggests its policy learned a partially Text-Response-heavy strategy, which happens to be optimal for high-literacy users but suboptimal for others. Low-literacy Yoruba (89) is its weakest cell.

### Cross-Algorithm Generalisation Summary

- **PPO generalises most uniformly** — smallest gap between best and worst cell (117 − 92 = 25 points).
- **DQN generalises similarly well** but with a slightly larger spread (116 − 86 = 30 points).
- **REINFORCE shows the most uneven generalisation** — strongest on high-literacy users, weakest on low-literacy non-Pidgin speakers.
- **High-literacy Yoruba is the hardest profile for all three algorithms** — the agent must choose Text Response (+10) over Voice Note (+2), but Yoruba's association with voice in the reward landscape creates a persistent confound.

---

##  Key Findings

**1. PPO is the recommended algorithm for deployment.**
It achieved the highest best-run reward (114.33), the highest 100-episode evaluation mean (113.1), the lowest average standard deviation (30.61), and the narrowest run-to-run spread (12.26 points). Its hyperparameter sensitivity is low — even its worst run scored 102.07.

**2. Voice Note + Pidgin is the highest-value action (+16), and all algorithms learn to exploit it.**
All three agents successfully identified that Pidgin-speaking, low-literacy users should receive voice notes — validating the reward structure as learnable and the environment as well-specified.

**3. A value function baseline is non-negotiable for REINFORCE.**
Without a baseline (vf_coef = 0.00), mean rewards ranged from 72–86 with standard deviations above 45. With a moderate baseline (vf_coef ≥ 0.25), performance jumped to 100+ and variance dropped dramatically. This single hyperparameter choice explained ~20 points of performance difference — the largest marginal gain from any single change across the entire experiment.

**4. DQN is the most learning-rate-sensitive algorithm.**
The gap between LR = 5e-3 (112.33) and LR = 1e-4 (30.67) is 81.66 points — the largest single-hyperparameter sensitivity across all 30 runs. DQN practitioners should treat learning rate as the primary tuning target before any other hyperparameter.

**5. High-literacy Yoruba is the hardest user profile for all three algorithms.**
Scores were 86 (DQN), 92 (PPO), and 111 (REINFORCE). The optimal action — Text Response (+10) — must be preferred over Voice Note (+2), but Yoruba co-activates learned voice note preferences. REINFORCE's accidentally Text-Response-heavy policy happened to serve this cell well; PPO and DQN overfit to voice note rewards.

**6. DQN converges fastest in terms of training timesteps.**
The TD loss curve stabilises by ~40,000 steps and the Q-value proxy plateaus clearly. PPO and REINFORCE show smoother but slower convergence due to on-policy update mechanics — PPO crossed the 100-point threshold at ~2,000 episodes, REINFORCE at ~5,000–6,000 episodes.

---

##  Results Files

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

##  Dependencies

```
gymnasium>=0.29.0
stable-baselines3[extra]>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.2.0
fastapi>=0.110.0
uvicorn>=0.29.0
pygame>=2.5.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

##  Implementation Notes

- **Model persistence:** Models are saved to `models/` after training. In notebook environments (Colab/Kaggle), mount Google Drive before training to prevent loss on runtime expiration.
- **Filename case sensitivity:** Ensure `api.py` (not `Api.py`) — case differences silently break imports across environments.
- **`uvicorn.run()` configuration:** Use `uvicorn.run(app, reload=False)` when passing a direct app object; `reload=True` with a direct object causes errors.
- **Syntax validation:** If copying Python files between environments, validate with `ast.parse()` before running to catch indentation errors early.

---

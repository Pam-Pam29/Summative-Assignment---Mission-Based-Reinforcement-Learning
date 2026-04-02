# Sista Health — Mission-Based Reinforcement Learning

> An RL agent that selects the best communication format (voice note, text, resource link, or clarification) for Nigerian women's health queries on WhatsApp, based on language, literacy level, and session context.

**Student:** Victoria Fakunle
**GitHub:** https://github.com/Pam-Pam29/victoria_fakunle_rl_summative
**Video:** https://drive.google.com/file/d/1NGtggZ01Uhmkxi1v0lr1AXehNlpMj8Gw/view?usp=drivesdk

---

## Repository Structure

```
├── environment/
│   ├── custom_env.py       # SistaHealthEnv (Gymnasium)
│   └── rendering.py        # WhatsApp-style pygame visualisation
├── training/
│   ├── dqn_training.py     # DQN: 10 hyperparameter runs
│   └── pg_training.py      # PPO + REINFORCE: 10 runs each
├── models/
│   ├── dqn/                # Saved DQN models
│   └── pg/                 # Saved PPO + REINFORCE models
├── results/                # All plots and CSVs
├── main.py                 # Entry point — runs best available agent
├── random_agent.py         # Random agent baseline demo
├── api.py                  # FastAPI REST endpoint
├── frontend_demo.html      # Browser-based WhatsApp demo
├── requirements.txt
└── README.md
```

---

## Quick Start

> **Python 3.10 or 3.11 required.** numpy 1.26.4 and torch 2.0+ are not compatible with Python 3.9 or 3.12.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train DQN
python training/dqn_training.py

# 3. Train PPO + REINFORCE
python training/pg_training.py

# 4. Run the best agent (auto-selects best available model)
python main.py

# Optional flags
python main.py --algo ppo          # force a specific algorithm
python main.py --algo dqn
python main.py --algo reinforce
python main.py --episodes 100
python main.py --render            # step-by-step terminal output

# 5. Run random agent baseline
python random_agent.py

# 6. Launch REST API
uvicorn api:app --reload

# 7. Open browser demo
open frontend_demo.html
```

---

## Environment

| Component | Details |
|-----------|---------|
| Observation space | `Box(5,)` — language, domain, topic, literacy, step |
| Action space | `Discrete(4)` — Text / Voice Note / Resource Link / Clarify |
| Episode length | 10 steps |
| Max reward/step | +16 (Voice Note + Pidgin + Low literacy) |
| Min reward/step | −4 (Text Response for Low-literacy user) |

---

## Results Summary

| Algorithm | Best Mean Reward | Run-to-Run Spread | 100-ep Eval Mean |
|-----------|-----------------|-------------------|-----------------|
| **PPO** | **114.33** | **12.26 pts** | **113.1** |
| DQN | 112.33 | 81.66 pts | 107.1 |
| REINFORCE | 103.47 | 31.14 pts | 109.8 |

**PPO is the best-performing and most stable algorithm.** Even its worst run scored 102.07 — above REINFORCE's best run.

---

## Dependencies

```
gymnasium==0.29.1
stable-baselines3==2.3.2
pygame==2.5.2
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
torch>=2.0.0
fastapi==0.110.0
uvicorn==0.29.0
scipy>=1.11.0
httpx>=0.27.0
python-multipart>=0.0.9
```

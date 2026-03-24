# fakunle_victoria_rl_summative

## Sista Health — Mission-Based Reinforcement Learning

**Victoria Fakunle | ALU Software Engineering (ML) | 2026**

---

### 🌍 Project Overview

This project trains a Reinforcement Learning agent to optimise response strategies for **Sista Health** — a voice-enabled multilingual sexual and reproductive health assistant for Nigerian women.

The RL agent learns to decide:
- Whether to respond via **text**, **voice note**, or **emergency referral**
- Matching the response modality to the user's language, literacy level, and urgency

### 🤖 Algorithms Compared
| Algorithm | Type |
|-----------|------|
| DQN | Value-Based |
| PPO | Policy Gradient |
| REINFORCE | Policy Gradient |

---

### 📁 Project Structure

```
fakunle_victoria_rl_summative/
├── environment/
│   ├── custom_env.py       # Custom Gymnasium environment
│   └── rendering.py        # Pygame WhatsApp-style visualization
├── training/
│   ├── dqn_training.py     # DQN + 10 hyperparameter experiments
│   └── pg_training.py      # PPO + REINFORCE + 20 experiments
├── models/
│   ├── dqn/                # Saved DQN models
│   └── pg/                 # Saved PPO + REINFORCE models
├── results/                # Graphs and CSV tables (auto-generated)
├── main.py                 # Run best model with visualization
├── requirements.txt
└── README.md
```

---

### 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fakunle_victoria_rl_summative
cd fakunle_victoria_rl_summative

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models
python training/dqn_training.py
python training/pg_training.py

# 4. Run best agent with visualization
python main.py --algo ppo --episodes 5
```

---

### 🎯 Environment Design

| Component | Details |
|-----------|---------|
| **Observation** | language, domain, topic, urgency, literacy, session_step |
| **Actions** | Text response, Voice note, Emergency referral, Clarify |
| **Reward** | +2 to +10 for correct decisions, -2 to -10 for errors |
| **Terminal** | 10 steps OR emergency triggered |

---

### 📊 Results

All results, graphs, and hyperparameter tables are saved in `/results/` after training.

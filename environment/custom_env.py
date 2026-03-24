import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SistaHealthEnv(gym.Env):
 """
 Sista Health RL Environment
 ============================
 Simulates a Nigerian woman sending an SRH/maternal health query to Sista Health.
 The agent must decide the best response strategy to maximize health outcomes.

 Observation Space (6 features):
 - language: 0=English, 1=Yoruba, 2=Pidgin
 - domain: 0=sexual_health, 1=maternal_health
 - topic: 0-8 (9 health topics)
 - urgency: 0=normal, 1=emergency
 - literacy: 0=low, 1=medium, 2=high
 - session_step: 0-9

 Action Space (4 discrete actions):
 - 0: Send text response
 - 1: Send voice note
 - 2: Trigger emergency referral
 - 3: Ask clarifying question

 Reward Structure:
 - Correct modality for literacy level: +2
 - User comprehends response: +3
 - Care-seeking intention triggered: +5
 - Emergency correctly escalated: +10
 - Emergency missed (normal response): -10
 - Wrong modality for literacy: -2
 - Unnecessary emergency flag: -3
 - Clarification when urgent: -2
 """

 metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

 # Topic names for rendering
 TOPICS = [
 "FGM Complications",
 "VVF Causes & Stigma",
 "Cultural/Religious Barriers",
 "Early Marriage Health",
 "TBA Dangers",
 "Contraception",
 "STIs & HIV",
 "Antenatal Care",
 "Postpartum Care",
 ]

 LANGUAGES = ["English", "Yoruba", "Pidgin"]
 DOMAINS = ["Sexual Health", "Maternal Health"]
 ACTIONS = ["Text Response", "Voice Note", "Emergency Referral", "Clarify"]

 def __init__(self, render_mode=None):
 super().__init__()

 # Observation space: 6 features
 self.observation_space = spaces.Box(
 low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
 high=np.array([2, 1, 8, 1, 2, 9], dtype=np.float32),
 dtype=np.float32,
 )

 # Action space: 4 discrete actions
 self.action_space = spaces.Discrete(4)

 self.render_mode = render_mode
 self.state = None
 self.step_count = 0
 self.episode_reward = 0
 self.last_action = None
 self.last_reward = 0
 self.last_feedback = ""

 # For rendering
 self.screen = None
 self.clock = None

 def _get_obs(self):
 return self.state.astype(np.float32)

 def _get_info(self):
 return {
 "language": self.LANGUAGES[int(self.state[0])],
 "domain": self.DOMAINS[int(self.state[1])],
 "topic": self.TOPICS[int(self.state[2])],
 "urgency": "Emergency" if self.state[3] == 1 else "Normal",
 "literacy": ["Low", "Medium", "High"][int(self.state[4])],
 "step": int(self.state[5]),
 "episode_reward": self.episode_reward,
 }

 def reset(self, seed=None, options=None):
 super().reset(seed=seed)

 # Random user profile
 language = self.np_random.integers(0, 3)
 domain = self.np_random.integers(0, 2)
 topic = self.np_random.integers(0, 9)
 # Emergency is rare (20% chance)
 urgency = 1 if self.np_random.random() < 0.2 else 0
 literacy = self.np_random.integers(0, 3)
 session_step = 0

 self.state = np.array(
 [language, domain, topic, urgency, literacy, session_step],
 dtype=np.float32,
 )
 self.step_count = 0
 self.episode_reward = 0
 self.last_action = None
 self.last_reward = 0
 self.last_feedback = "New user session started"

 return self._get_obs(), self._get_info()

 def step(self, action):
 assert self.action_space.contains(action), f"Invalid action: {action}"

 language = int(self.state[0])
 urgency = int(self.state[3])
 literacy = int(self.state[4])

 reward = 0
 terminated = False
 feedback = ""

 # --- Reward Logic ---

 # Action 2 = Emergency Referral
 if action == 2:
 if urgency == 1:
 reward += 10
 feedback = " Emergency correctly escalated! +10"
 terminated = True # Episode ends on emergency
 else:
 reward -= 3
 feedback = " Unnecessary emergency flag. -3"

 # Action 0 = Text Response
 elif action == 0:
 if urgency == 1:
 reward -= 10
 feedback = " Emergency missed! Should escalate. -10"
 elif literacy == 2:
 # High literacy → text is great
 reward += 2 + 3
 feedback = " Text works well for high literacy user. +5"
 elif literacy == 1:
 reward += 2
 feedback = " Text is okay for medium literacy. +2"
 else:
 # Low literacy → text is wrong modality
 reward -= 2
 feedback = " Text is poor for low literacy user. -2"

 # Action 1 = Voice Note
 elif action == 1:
 if urgency == 1:
 reward -= 10
 feedback = " Emergency missed! Should escalate. -10"
 elif literacy == 0:
 # Low literacy → voice is perfect
 reward += 2 + 3 + 5
 feedback = " Voice note perfect for low literacy! +10"
 elif literacy == 1:
 reward += 2 + 3
 feedback = " Voice note good for medium literacy. +5"
 else:
 # High literacy → voice is okay but not optimal
 reward += 1
 feedback = "Voice note okay for high literacy user. +1"

 # Action 3 = Ask Clarifying Question
 elif action == 3:
 if urgency == 1:
 reward -= 2
 feedback = " Don't clarify during emergency! -2"
 else:
 reward += 1
 feedback = "Clarification asked. +1"

 # Pidgin users get bonus for voice (preferred modality)
 if action == 1 and language == 2 and urgency == 0:
 reward += 1
 feedback += " (Pidgin voice bonus +1)"

 # Update state
 self.state[5] = min(self.state[5] + 1, 9)
 self.step_count += 1
 self.last_action = action
 self.last_reward = reward
 self.last_feedback = feedback
 self.episode_reward += reward

 # Terminal conditions
 if self.step_count >= 10:
 terminated = True

 truncated = False

 if self.render_mode == "human":
 self._render_frame()

 return self._get_obs(), reward, terminated, truncated, self._get_info()

 def render(self):
 if self.render_mode == "rgb_array":
 return self._render_frame()
 elif self.render_mode == "human":
 self._render_frame()

 def _render_frame(self):
 try:
 import pygame
 except ImportError:
 print("pygame not installed. Run: pip install pygame")
 return

 from environment.rendering import render_frame
 return render_frame(self)

 def close(self):
 if self.screen is not None:
 import pygame
 pygame.display.quit()
 pygame.quit()
 self.screen = None

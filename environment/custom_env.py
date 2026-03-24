import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SistaHealthEnv(gym.Env):
    """
    Sista Health Custom RL Environment
    ====================================
    Simulates a Nigerian woman sending an SRH/maternal health query.
    The agent learns the optimal response strategy.

    Observation Space (6 features):
        language:     0=English, 1=Yoruba, 2=Pidgin
        domain:       0=Sexual Health, 1=Maternal Health
        topic:        0-8 (9 health topics)
        urgency:      0=Normal, 1=Emergency
        literacy:     0=Low, 1=Medium, 2=High
        session_step: 0-9

    Action Space (4 discrete actions):
        0 = Text Response
        1 = Voice Note
        2 = Emergency Referral
        3 = Ask Clarifying Question
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    TOPICS = [
        "FGM Complications",
        "VVF Causes",
        "Cultural Barriers",
        "Early Marriage",
        "TBA Dangers",
        "Contraception",
        "STIs and HIV",
        "Antenatal Care",
        "Postpartum Care",
    ]
    LANGUAGES = ["English", "Yoruba", "Pidgin"]
    DOMAINS   = ["Sexual Health", "Maternal Health"]
    ACTIONS   = ["Text Response", "Voice Note", "Emergency Referral", "Clarify"]

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 1, 8, 1, 2, 9], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space   = spaces.Discrete(4)
        self.render_mode    = render_mode
        self.state          = None
        self.step_count     = 0
        self.episode_reward = 0
        self.last_action    = None
        self.last_reward    = 0
        self.last_feedback  = ""
        self.screen         = None
        self.clock          = None

    def _get_obs(self):
        return self.state.astype(np.float32)

    def _get_info(self):
        return {
            "language":       self.LANGUAGES[int(self.state[0])],
            "domain":         self.DOMAINS[int(self.state[1])],
            "topic":          self.TOPICS[int(self.state[2])],
            "urgency":        "Emergency" if self.state[3] == 1 else "Normal",
            "literacy":       ["Low", "Medium", "High"][int(self.state[4])],
            "step":           int(self.state[5]),
            "episode_reward": self.episode_reward,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [
                self.np_random.integers(0, 3),
                self.np_random.integers(0, 2),
                self.np_random.integers(0, 9),
                1 if self.np_random.random() < 0.2 else 0,
                self.np_random.integers(0, 3),
                0,
            ],
            dtype=np.float32,
        )
        self.step_count     = 0
        self.episode_reward = 0
        self.last_action    = None
        self.last_reward    = 0
        self.last_feedback  = "New user session started"
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        urgency  = int(self.state[3])
        literacy = int(self.state[4])
        language = int(self.state[0])
        reward   = 0
        terminated = False
        feedback   = ""

        if action == 2:
            if urgency == 1:
                reward    += 10
                feedback   = "Emergency correctly escalated. +10"
                terminated = True
            else:
                reward  -= 3
                feedback = "Unnecessary emergency flag. -3"

        elif action == 0:
            if urgency == 1:
                reward  -= 10
                feedback = "Emergency missed. -10"
            elif literacy == 2:
                reward  += 5
                feedback = "Text great for high literacy. +5"
            elif literacy == 1:
                reward  += 2
                feedback = "Text ok for medium literacy. +2"
            else:
                reward  -= 2
                feedback = "Text poor for low literacy. -2"

        elif action == 1:
            if urgency == 1:
                reward  -= 10
                feedback = "Emergency missed. -10"
            elif literacy == 0:
                reward  += 10
                feedback = "Voice perfect for low literacy. +10"
            elif literacy == 1:
                reward  += 5
                feedback = "Voice good for medium literacy. +5"
            else:
                reward  += 1
                feedback = "Voice ok for high literacy. +1"
            if language == 2:
                reward  += 1
                feedback += " Pidgin bonus +1"

        elif action == 3:
            if urgency == 1:
                reward  -= 2
                feedback = "Do not clarify during emergency. -2"
            else:
                reward  += 1
                feedback = "Clarification asked. +1"

        self.state[5]       = min(self.state[5] + 1, 9)
        self.step_count    += 1
        self.last_action    = action
        self.last_reward    = reward
        self.last_feedback  = feedback
        self.episode_reward += reward

        if self.step_count >= 10:
            terminated = True

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
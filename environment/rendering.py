"""
Sista Health - Pygame Visualization
=====================================
Renders a WhatsApp-style interface showing:
- Current user profile (left panel)
- Agent chosen action (center)
- Live reward tracker (right panel)
- Feedback message (bottom)
"""

import numpy as np

# Colors
BG_COLOR       = (18, 18, 18)
PANEL_COLOR    = (30, 30, 30)
WHATSAPP_GREEN = (37, 211, 102)
WHATSAPP_DARK  = (7, 94, 84)
WHITE          = (255, 255, 255)
LIGHT_GRAY     = (200, 200, 200)
GRAY           = (120, 120, 120)
RED            = (220, 53, 69)
ORANGE         = (255, 165, 0)
YELLOW         = (255, 193, 7)
BLUE           = (13, 110, 253)
PURPLE         = (111, 66, 193)

ACTION_COLORS = {
    0: BLUE,
    1: WHATSAPP_GREEN,
    2: RED,
    3: ORANGE,
}

ACTION_LABELS = {
    0: "[TXT]",
    1: "[VOI]",
    2: "[EMR]",
    3: "[CLR]",
}

WINDOW_W = 900
WINDOW_H = 600


def render_frame(env):
    import pygame

    if env.screen is None:
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Sista Health - RL Agent")
        env.clock = pygame.time.Clock()

    try:
        font_large  = pygame.font.SysFont("Arial", 22, bold=True)
        font_medium = pygame.font.SysFont("Arial", 17)
        font_small  = pygame.font.SysFont("Arial", 14)
        font_title  = pygame.font.SysFont("Arial", 28, bold=True)
    except Exception:
        font_large  = pygame.font.Font(None, 26)
        font_medium = pygame.font.Font(None, 20)
        font_small  = pygame.font.Font(None, 16)
        font_title  = pygame.font.Font(None, 32)

    surface = env.screen
    surface.fill(BG_COLOR)

    # Title Bar
    pygame.draw.rect(surface, WHATSAPP_DARK, (0, 0, WINDOW_W, 55))
    title = font_title.render("Sista Health - RL Response Agent", True, WHITE)
    surface.blit(title, (20, 14))

    # Left Panel: User Profile
    pygame.draw.rect(surface, PANEL_COLOR, (15, 70, 260, 450), border_radius=12)
    pygame.draw.rect(surface, WHATSAPP_GREEN, (15, 70, 260, 40), border_radius=12)
    lbl = font_large.render("User Profile", True, WHITE)
    surface.blit(lbl, (25, 80))

    if env.state is not None:
        info = env._get_info()
        profile_items = [
            ("Language", info["language"],  WHATSAPP_GREEN),
            ("Domain",   info["domain"],    BLUE),
            ("Topic",    info["topic"],     PURPLE),
            ("Urgency",  info["urgency"],   RED if info["urgency"] == "Emergency" else LIGHT_GRAY),
            ("Literacy", info["literacy"],  YELLOW),
            ("Step",     f"{info['step']} / 9", GRAY),
        ]
        y = 125
        for label, value, color in profile_items:
            key_surf = font_small.render(label.upper(), True, GRAY)
            val_surf = font_medium.render(str(value), True, color)
            surface.blit(key_surf, (28, y))
            surface.blit(val_surf, (28, y + 16))
            pygame.draw.line(surface, (50, 50, 50), (28, y + 36), (255, y + 36))
            y += 50

        # Episode reward box
        pygame.draw.rect(surface, (40, 40, 40), (20, 445, 250, 65), border_radius=8)
        ep_lbl = font_small.render("EPISODE REWARD", True, GRAY)
        ep_val = font_large.render(
            f"{env.episode_reward:.1f}",
            True,
            WHATSAPP_GREEN if env.episode_reward >= 0 else RED
        )
        surface.blit(ep_lbl, (30, 452))
        surface.blit(ep_val, (30, 470))

    # Center Panel: Agent Decision
    pygame.draw.rect(surface, PANEL_COLOR, (295, 70, 310, 450), border_radius=12)
    pygame.draw.rect(surface, WHATSAPP_DARK, (295, 70, 310, 40), border_radius=12)
    albl = font_large.render("Agent Decision", True, WHITE)
    surface.blit(albl, (305, 80))

    if env.last_action is not None:
        action_name  = env.ACTIONS[env.last_action]
        action_color = ACTION_COLORS[env.last_action]
        action_label = ACTION_LABELS[env.last_action]

        # Big action bubble
        pygame.draw.rect(surface, action_color, (315, 125, 270, 80), border_radius=16)
        act_surf = font_large.render(f"{action_label} {action_name}", True, WHITE)
        surface.blit(act_surf, (330, 155))

        # All actions list
        y = 225
        for i, aname in enumerate(env.ACTIONS):
            col  = ACTION_COLORS[i]
            bold = (i == env.last_action)
            bg   = (50, 50, 50) if bold else (35, 35, 35)
            pygame.draw.rect(surface, bg, (315, y, 270, 38), border_radius=8)
            if bold:
                pygame.draw.rect(surface, col, (315, y, 6, 38), border_radius=4)
            txt = font_medium.render(
                f"  {ACTION_LABELS[i]}  {aname}",
                True,
                col if bold else GRAY
            )
            surface.blit(txt, (320, y + 10))
            y += 46

        # Step reward
        rew_color = WHATSAPP_GREEN if env.last_reward >= 0 else RED
        rew_surf  = font_large.render(
            f"Step Reward: {env.last_reward:+.0f}", True, rew_color
        )
        surface.blit(rew_surf, (315, 420))

    else:
        waiting = font_medium.render("Waiting for first action...", True, GRAY)
        surface.blit(waiting, (315, 270))

    # Right Panel: Reward Tracker
    pygame.draw.rect(surface, PANEL_COLOR, (625, 70, 260, 450), border_radius=12)
    pygame.draw.rect(surface, PURPLE, (625, 70, 260, 40), border_radius=12)
    rlbl = font_large.render("Reward Tracker", True, WHITE)
    surface.blit(rlbl, (635, 80))

    # Reward bar chart (steps 0-9)
    bar_w   = 20
    bar_gap = 5
    base_y  = 380
    max_bar = 120

    for i in range(10):
        x = 635 + i * (bar_w + bar_gap)
        if env.state is not None and i < int(env.state[5]):
            h   = min(abs(env.last_reward) * 10, max_bar) if i == int(env.state[5]) - 1 else 20
            col = WHATSAPP_GREEN if env.last_reward >= 0 else RED
        else:
            h   = 5
            col = (60, 60, 60)
        pygame.draw.rect(surface, col, (x, base_y - h, bar_w, h), border_radius=4)
        step_lbl = font_small.render(str(i), True, GRAY)
        surface.blit(step_lbl, (x + 5, base_y + 5))

    axis_lbl = font_small.render("Steps", True, GRAY)
    surface.blit(axis_lbl, (635, base_y + 20))

    # Legend
    legend_items = [
        (WHATSAPP_GREEN, "Positive reward"),
        (RED,            "Negative reward"),
        (GRAY,           "Pending step"),
    ]
    y = 395
    for col, txt in legend_items:
        pygame.draw.circle(surface, col, (645, y + 6), 6)
        surface.blit(font_small.render(txt, True, LIGHT_GRAY), (658, y))
        y += 20

    # Bottom: Feedback Message
    pygame.draw.rect(surface, PANEL_COLOR, (15, 530, 870, 55), border_radius=10)
    feedback_txt = env.last_feedback if env.last_feedback else "Agent initialising..."
    if len(feedback_txt) > 90:
        feedback_txt = feedback_txt[:87] + "..."
    fb_surf = font_medium.render(f"  {feedback_txt}", True, WHATSAPP_GREEN)
    surface.blit(fb_surf, (25, 547))

    pygame.event.pump()
    pygame.display.flip()
    env.clock.tick(env.metadata["render_fps"])

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
    )
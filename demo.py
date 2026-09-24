# demo.py
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import Tuple
import numpy as np

# Fallback to dummy video driver if headless Linux without display server
if "DISPLAY" not in os.environ and sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from nevorl import NevoRLCompiler

# =====================================================================
# 1. 30x24 ASCII CIRCUIT ('X' = Open Track, '#' = Wall, 'S' = Start, 'G' = Goal)
# =====================================================================
ASCII_CIRCUIT = """
##############################
#SXXXXXXXX#XXXXXXXXXXXXXXXXXX#
#XXXXXXXXX#XXXXXXXXXXXXXXXXXX#
#XXXXXXXXX#XXXX##########XXXX#
#XXXX#####XXXXX#XXXXXXXX#XXXX#
#XXXX#XXXXXXXXX#XXXXXXXX#XXXX#
#XXXX#XXXXXXXXX#XXX##XXX#XXXX#
#XXXX#XXXX######XXX##XXX#XXXX#
#XXXX#XXXX#XXXXXXXX##XXXXXXXX#
#XXXX#XXXX#XXXXXXXXXXXXXXXXXX#
#XXXXXXXXX#XXXX##########XXXX#
#XXXXXXXXX#XXXX#XXXXXXXX#XXXX#
######XXXX######XXXXXXXX######
#XXXXXXXXXXXXXX#XXXX#####XXXX#
#XXXXXXXXXXXXXX#XXXX#####X##X#
#XXXX##########XXXXX#####X#XX#
#XXXX#XX#################X#X##
#XXXX#XX####XXXXXXXX#####X#XX#
#XXXX#XX#####XXXXXXXX###XX##X#
#XXXX#XX###############XXX#XX#
#XXXX#XX#XXXXXXX#######XXX#XX#
#XXXXXXX###XXXXX#######X####X#
#XXXXXXXXXXXXXXXXXXXXXXX#XXXG#
##############################
"""

def parse_ascii_track(ascii_map: str) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    lines = [row.strip() for row in ascii_map.strip().splitlines() if row.strip()]
    H, W = len(lines), len(lines[0])
    grid = np.zeros((H, W), dtype=np.uint8)
    start_pos = (1.5, 1.5)
    goal_pos = (W - 2.5, H - 2.5)

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == "#":
                grid[y, x] = 1
            elif char == "S":
                start_pos = (x + 0.5, y + 0.5)
            elif char == "G":
                goal_pos = (x + 0.5, y + 0.5)

    return grid, start_pos, goal_pos


# =====================================================================
# 2. NEVORL ENVIRONMENT
# =====================================================================
def build_environment_source(start_pt: Tuple[float, float], goal_pt: Tuple[float, float]) -> str:
    return f"""
env CyberArena {{
    state {{
        pos: vec2;
        vel: vec2;
        target: vec2;
        heading: float;
        ang_vel: float;
        z: float;
        vz: float;
        crashed: float;
        steps: int;
    }}

    action continuous(3); // act[0]=steer, act[1]=throttle, act[2]=jump (>0.0)

    observation {{
        let rays = raycast_fan(state.pos, state.heading, 2.094, 5, 6.5);
        let bfs_dir = bfs_vector(state.pos, state.target);
        let path_dist = bfs_dist(state.pos, state.target);

        // Transform corridor direction and velocity into car-local frame
        let ego_dir = rotate(bfs_dir, -state.heading);
        let ego_vel = rotate(state.vel, -state.heading);

        // Output dim = 14:
        // [0..4]: rays (-60°, -30°, 0°, +30°, +60°), [5]: heading, [6]: path_dist, [7..8]: bfs_dir, [9]: z
        // [10]: ego_dir.x (fwd alignment), [11]: ego_dir.y (lateral steer error), [12]: ego_vel.x, [13]: ego_vel.y
        return [rays, state.heading, path_dist, bfs_dir.x, bfs_dir.y, state.z, ego_dir.x, ego_dir.y, ego_vel.x, ego_vel.y];
    }}

    reset {{
        state.pos = vec2({start_pt[0]:.2f}, {start_pt[1]:.2f});
        state.vel = vec2(0.0, 0.0);
        state.target = vec2({goal_pt[0]:.2f}, {goal_pt[1]:.2f});
        state.heading = 0.0;
        state.ang_vel = 0.0;
        state.z = 0.0;
        state.vz = 0.0;
        state.crashed = 0.0;
        state.steps = 0;
    }}

    step(act) {{
        let wants_jump = act[2] > 0.0;
        let on_ground = state.z <= 0.02;
        let do_jump = on_ground and wants_jump;

        state.vz = where(do_jump, 0.36, state.vz - 0.032);
        state.z = clamp(state.z + state.vz, 0.0, 3.0);
        state.vz = where(state.z <= 0.0, 0.0, state.vz);

        // High traction (0.82), agile steering (0.42), compact radius (0.28)
        let hit = kinematics_car(act[0], act[1], 0.82, 0.88, 0.42, 0.45, 0.05, 0.28, state.z);

        let touching_wall = is_wall(state.pos, 0.28);
        let landed_on_wall = (state.z <= 0.08) and touching_wall;

        state.crashed = where(hit or landed_on_wall, 1.0, 0.0);
        state.steps = state.steps + 1;
    }}

    reward {{
        let path_dist = bfs_dist(state.pos, state.target);
        let bfs_dir = bfs_vector(state.pos, state.target);

        // Velocity along shortest path corridor in WORLD frame
        let corridor_vel = state.vel.x * bfs_dir.x + state.vel.y * bfs_dir.y;

        // Alignment with path direction
        let fwd_x = cos(state.heading);
        let fwd_y = sin(state.heading);
        let alignment = fwd_x * bfs_dir.x + fwd_y * bfs_dir.y;

        let reached_goal = (path_dist < 1.8) and (state.z <= 0.08);
        let goal_bonus = where(reached_goal, 5000.0, 0.0);
        let crash_tax = where(state.crashed > 0.5, 60.0, 0.0);

        return corridor_vel * 15.0 + alignment * 2.5 - 0.2 + goal_bonus - crash_tax;
    }}

    terminal {{
        let path_dist = bfs_dist(state.pos, state.target);
        return ((path_dist < 1.8) and (state.z <= 0.08)) or (state.crashed > 0.5) or (state.steps >= 340);
    }}
}}
"""

# =====================================================================
# 3. HIGH-SPEED NEUROEVOLUTION ENGINE
# =====================================================================
class FastNeuroEvolution:
    def __init__(self, pop_size=1024, in_dim=14, out_dim=3):
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Layer 1: 14 -> 32
        self.W1 = np.random.randn(pop_size, in_dim, 32).astype(np.float32) * 0.02
        self.b1 = np.zeros((pop_size, 32), dtype=np.float32)

        # Layer 2: 32 -> 16
        self.W2 = np.random.randn(pop_size, 32, 16).astype(np.float32) * 0.02
        self.b2 = np.zeros((pop_size, 16), dtype=np.float32)

        # Layer 3: 16 -> 3
        self.W3 = np.random.randn(pop_size, 16, out_dim).astype(np.float32) * 0.02
        self.b3 = np.zeros((pop_size, out_dim), dtype=np.float32)

        # --- SEED HIGH-PERFORMANCE STEERING & APEX BRAKING PRIOR ---
        # 1. Channel corridor lateral error (ego_dir.y, idx 11) to steer (act[0])
        self.W1[:, 11, 0] = 2.4
        # Wall repulsion: right rays minus left rays (positive = steer right away from left wall)
        self.W1[:, 0, 0] = -0.4   # left-most ray
        self.W1[:, 1, 0] = -0.7   # left diagonal ray
        self.W1[:, 3, 0] = 0.7    # right diagonal ray
        self.W1[:, 4, 0] = 0.4    # right-most ray
        self.W2[:, 0, 0] = 1.8
        self.W3[:, 0, 0] = 1.5

        # 2. Dynamic Speed Control: brake into sharp corners, full throttle on straights
        self.W1[:, 10, 1] = 1.6   # ego_dir.x (fwd corridor alignment)
        self.W1[:, 2, 1] = 1.4    # center ray clearance
        self.b1[:, 1] = -0.6
        self.W2[:, 1, 1] = 1.5
        self.W3[:, 1, 1] = 1.2
        self.b3[:, 1] = 0.7       # baseline forward thrust

        # 3. Jump thruster: grounded by default
        self.b3[:, 2] = -1.2

        # Small exploratory variance across the population
        self.W1 += np.random.randn(*self.W1.shape).astype(np.float32) * 0.05
        self.W2 += np.random.randn(*self.W2.shape).astype(np.float32) * 0.05
        self.W3 += np.random.randn(*self.W3.shape).astype(np.float32) * 0.05

        self.generation = 0
        self.last_hidden = np.zeros((pop_size, 16), dtype=np.float32)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        N = obs.shape[0]
        w1, b1 = self.W1[:N], self.b1[:N]
        w2, b2 = self.W2[:N], self.b2[:N]
        w3, b3 = self.W3[:N], self.b3[:N]

        h1 = np.tanh(np.matmul(obs[:, None, :], w1).squeeze(1) + b1)
        h2 = np.tanh(np.matmul(h1[:, None, :], w2).squeeze(1) + b2)
        self.last_hidden = h2

        out = np.tanh(np.matmul(h2[:, None, :], w3).squeeze(1) + b3)
        return out

    def evolve(self, fitness: np.ndarray):
        ranks = np.argsort(fitness)[::-1]
        self.generation += 1

        num_elites = 16
        elites_w1 = self.W1[ranks[:num_elites]].copy()
        elites_b1 = self.b1[ranks[:num_elites]].copy()
        elites_w2 = self.W2[ranks[:num_elites]].copy()
        elites_b2 = self.b2[ranks[:num_elites]].copy()
        elites_w3 = self.W3[ranks[:num_elites]].copy()
        elites_b3 = self.b3[ranks[:num_elites]].copy()

        # Truncation selection from top 10%
        top_pool_size = max(num_elites, int(self.pop_size * 0.10))
        top_indices = ranks[:top_pool_size]
        weights = 1.0 / np.sqrt(np.arange(1, top_pool_size + 1))
        probs = weights / np.sum(weights)

        chosen = np.random.choice(top_indices, size=self.pop_size, p=probs)

        new_W1 = self.W1[chosen].copy()
        new_b1 = self.b1[chosen].copy()
        new_W2 = self.W2[chosen].copy()
        new_b2 = self.b2[chosen].copy()
        new_W3 = self.W3[chosen].copy()
        new_b3 = self.b3[chosen].copy()

        # Multi-scale fine-tuning mutations (70% micro, 25% medium, 5% macro)
        sigmas = np.random.choice([0.02, 0.06, 0.15], size=(self.pop_size, 1, 1), p=[0.70, 0.25, 0.05]).astype(np.float32)
        p_mut = 0.15

        m1 = np.random.rand(*new_W1.shape) < p_mut
        new_W1 += (m1 * np.random.randn(*new_W1.shape)).astype(np.float32) * sigmas
        new_b1 += ((np.random.rand(*new_b1.shape) < p_mut) * np.random.randn(*new_b1.shape)).astype(np.float32) * sigmas.squeeze(-1)

        m2 = np.random.rand(*new_W2.shape) < p_mut
        new_W2 += (m2 * np.random.randn(*new_W2.shape)).astype(np.float32) * sigmas
        new_b2 += ((np.random.rand(*new_b2.shape) < p_mut) * np.random.randn(*new_b2.shape)).astype(np.float32) * sigmas.squeeze(-1)

        m3 = np.random.rand(*new_W3.shape) < p_mut
        new_W3 += (m3 * np.random.randn(*new_W3.shape)).astype(np.float32) * sigmas
        new_b3 += ((np.random.rand(*new_b3.shape) < p_mut) * np.random.randn(*new_b3.shape)).astype(np.float32) * sigmas.squeeze(-1)

        # Strictly preserve top elites untouched
        new_W1[:num_elites] = elites_w1
        new_b1[:num_elites] = elites_b1
        new_W2[:num_elites] = elites_w2
        new_b2[:num_elites] = elites_b2
        new_W3[:num_elites] = elites_w3
        new_b3[:num_elites] = elites_b3

        self.W1, self.b1 = new_W1, new_b1
        self.W2, self.b2 = new_W2, new_b2
        self.W3, self.b3 = new_W3, new_b3

    def train_epoch(self, envs, generations=60, rollout_steps=340, verbose=True):
        t0 = time.perf_counter()
        top_fit = -9999.0

        for gen in range(generations):
            obs, _ = envs.reset()
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            alive = np.ones(self.pop_size, dtype=bool)
            completed = np.zeros(self.pop_size, dtype=bool)
            steps_to_goal = np.full(self.pop_size, rollout_steps, dtype=np.int32)

            init_dist = obs[:, 6].copy()
            min_dist = init_dist.copy()

            for step_idx in range(rollout_steps):
                act = self.forward(obs)
                noise = np.random.randn(*act.shape).astype(np.float32) * 0.015
                act_noisy = np.clip(act + noise, -1.0, 1.0)

                obs, rewards, term, trunc, _ = envs.step(act_noisy)

                # Record goal achievement
                cur_dist = obs[:, 6]
                reached = alive & (cur_dist < 1.8)
                newly_completed = reached & (~completed)
                completed |= reached
                steps_to_goal = np.where(newly_completed, step_idx + 1, steps_to_goal)

                fitness += np.where(alive, rewards, 0.0)
                min_dist = np.where(alive & (cur_dist < min_dist), cur_dist, min_dist)

                alive &= ~(term | trunc)
                if not np.any(alive):
                    break

            # Distance-to-goal progress bonus
            net_progress = np.maximum(0.0, init_dist - min_dist)
            fitness += net_progress * 80.0

            # Huge reward for solving the maze + bonus for speed
            finish_bonus = np.where(completed, 8000.0 + (rollout_steps - steps_to_goal) * 20.0, 0.0)
            fitness += finish_bonus

            top_fit = float(np.max(fitness))
            min_rem_dist = float(np.min(min_dist))
            num_solved = int(np.sum(completed))

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Closest: {min_rem_dist:4.1f}u | Solved: {num_solved:3d}/{self.pop_size}")

            self.evolve(fitness)

        elapsed = time.perf_counter() - t0
        sps = (self.pop_size * rollout_steps * generations) / max(elapsed, 1e-5)
        return elapsed, sps, top_fit

    def evolve_more(self, env_cls, maze, generations=10):
        envs = env_cls(num_envs=self.pop_size, grid_map=maze)
        self.train_epoch(envs, generations=generations, rollout_steps=340, verbose=True)


# =====================================================================
# 4. HIGH-PRODUCTION 3D JUMP VISUALIZER & VIDEO RECORDER
# =====================================================================
class CyberVisualizer:
    def __init__(self, env_cls, maze, ga: FastNeuroEvolution, start_pt, goal_pt):
        pygame.init()
        pygame.font.init()

        self.cell_size = 26
        self.cols, self.rows = maze.shape[1], maze.shape[0]
        self.arena_w = self.cols * self.cell_size
        self.arena_h = self.rows * self.cell_size
        self.hud_h = 100

        self.screen_w = self.arena_w + 240
        self.screen_h = self.arena_h + self.hud_h
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NevoRL Autonomous Complex Labyrinth [3D Airtime + HUD]")
        self.clock = pygame.time.Clock()

        font_names = ["Consolas", "dejavusansmono", "monospace", "courier"]
        self.font_main = pygame.font.SysFont(font_names, 14, bold=True)
        self.font_big = pygame.font.SysFont(font_names, 18, bold=True)
        self.font_tiny = pygame.font.SysFont(font_names, 11)

        self.maze = maze
        self.ga = ga
        self.env_cls = env_cls
        self.start_pt = start_pt
        self.goal_pt = goal_pt

        self.swarm_size = 40
        self.env = env_cls(num_envs=self.swarm_size, grid_map=maze)
        self.obs, _ = self.env.reset()

        self.particles: list[list[float]] = []
        self.show_swarm = True
        self.show_lasers = True
        self.paused = False

    def spawn_drift_smoke(self, x: float, y: float, heading: float, speed: float):
        for _ in range(2):
            bx = x - math.cos(heading) * 12 + np.random.uniform(-3, 3)
            by = y - math.sin(heading) * 12 + np.random.uniform(-3, 3)
            vx = -math.cos(heading) * speed * 4 + np.random.uniform(-1, 1)
            vy = -math.sin(heading) * speed * 4 + np.random.uniform(-1, 1)
            self.particles.append([bx, by, vx, vy, 1.0, 1.0])

    def update_particles(self):
        survivors = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 0.05
            if p[4] > 0:
                survivors.append(p)
        self.particles = survivors

    def draw_particles(self, surface):
        for p in self.particles:
            alpha = int((p[4] / p[5]) * 140)
            radius = int((1.0 - (p[4] / p[5])) * 8 + 3)
            smoke_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(smoke_surf, (200, 220, 255, alpha), (radius, radius), radius)
            surface.blit(smoke_surf, (int(p[0] - radius), int(p[1] - radius)))

    def draw_hud(self, action, speed, slip_deg, path_dist, altitude):
        hud_rect = pygame.Rect(0, self.arena_h, self.screen.get_width(), self.hud_h)
        pygame.draw.rect(self.screen, (11, 15, 23), hud_rect)
        pygame.draw.line(self.screen, (30, 41, 59), (0, self.arena_h), (self.screen.get_width(), self.arena_h), 2)

        txt_gen = self.font_big.render(f"GEN {self.ga.generation:03d}", True, (56, 189, 248))
        txt_spd = self.font_main.render(f"SPEED     : {speed:4.2f} u/f", True, (241, 245, 249))
        alt_color = (192, 132, 252) if altitude > 0.05 else (148, 163, 184)
        txt_alt = self.font_main.render(f"ALTITUDE  : {altitude:4.2f} m", True, alt_color)
        txt_dist = self.font_main.render(f"PATH DIST : {path_dist:4.1f} units", True, (52, 211, 153))

        self.screen.blit(txt_gen, (20, self.arena_h + 12))
        self.screen.blit(txt_spd, (20, self.arena_h + 38))
        self.screen.blit(txt_alt, (20, self.arena_h + 56))
        self.screen.blit(txt_dist, (20, self.arena_h + 74))

        steer_val = float(action[0, 0])
        gas_val = float(action[0, 1])
        jump_val = float(action[0, 2])
        pygame.draw.line(self.screen, (30, 41, 59), (220, self.arena_h + 10), (220, self.arena_h + 90), 1)

        txt_act = self.font_big.render("AI ACTUATORS", True, (148, 163, 184))
        self.screen.blit(txt_act, (235, self.arena_h + 12))

        # Steer indicator
        pygame.draw.rect(self.screen, (20, 26, 38), (235, self.arena_h + 38, 110, 12))
        center_x = 235 + 55
        steer_bar_w = int(steer_val * 53)
        bar_color = (244, 63, 94) if steer_val < 0 else (56, 189, 248)
        pygame.draw.rect(self.screen, bar_color, (center_x if steer_val > 0 else center_x + steer_bar_w, self.arena_h + 38, abs(steer_bar_w), 12))
        self.screen.blit(self.font_tiny.render(f"STEER [{steer_val:+.2f}]", True, (203, 213, 225)), (355, self.arena_h + 38))

        # Gas indicator
        pygame.draw.rect(self.screen, (20, 26, 38), (235, self.arena_h + 56, 110, 12))
        gas_bar_w = int(max(0.0, gas_val) * 110)
        pygame.draw.rect(self.screen, (34, 197, 94), (235, self.arena_h + 56, gas_bar_w, 12))
        self.screen.blit(self.font_tiny.render(f"GAS   [{gas_val:.2f}]", True, (203, 213, 225)), (355, self.arena_h + 56))

        # Jump thruster indicator
        is_firing = jump_val > 0.0
        jump_color = (192, 132, 252) if is_firing else (71, 85, 105)
        jump_state = "FIRING" if is_firing else "GROUND"
        self.screen.blit(self.font_main.render(f"JUMP THRUSTER: [{jump_state}]", True, jump_color), (235, self.arena_h + 74))

        # Shortcuts Guide
        pygame.draw.line(self.screen, (30, 41, 59), (480, self.arena_h + 10), (480, self.arena_h + 90), 1)
        self.screen.blit(self.font_main.render("AUTONOMOUS CONTROL", True, (168, 85, 247)), (500, self.arena_h + 12))
        keys = [
            "[SPACE] Pause / Resume",
            "[G]     Toggle Ghost Swarm",
            "[E]     Evolve +10 Generations",
            "[R]     Reset Lap"
        ]
        for i, k in enumerate(keys):
            self.screen.blit(self.font_tiny.render(k, True, (100, 116, 139)), (500, self.arena_h + 32 + i * 15))

    def draw_side_panel(self, obs, hidden_act):
        panel_x = self.arena_w
        panel_w = 240
        pygame.draw.rect(self.screen, (15, 23, 42), (panel_x, 0, panel_w, self.arena_h))
        pygame.draw.line(self.screen, (30, 41, 59), (panel_x, 0), (panel_x, self.arena_h), 2)

        txt = self.font_big.render("NEURAL MONITOR", True, (56, 189, 248))
        self.screen.blit(txt, (panel_x + 15, 16))

        lbl_rays = self.font_main.render("RAYCAST DISTANCES", True, (148, 163, 184))
        self.screen.blit(lbl_rays, (panel_x + 15, 52))

        angles_lbl = ["-60°", "-30°", "  0°", "+30°", "+60°"]
        for i in range(5):
            r_val = float(obs[0, i])
            by = 75 + i * 18
            self.screen.blit(self.font_tiny.render(angles_lbl[i], True, (148, 163, 184)), (panel_x + 15, by))
            pygame.draw.rect(self.screen, (20, 26, 38), (panel_x + 55, by + 1, 140, 10))
            bar_c = (244, 63, 94) if r_val < 0.3 else ((250, 204, 21) if r_val < 0.7 else (52, 211, 153))
            pygame.draw.rect(self.screen, bar_c, (panel_x + 55, by + 1, int(r_val * 140), 10))

        lbl_h = self.font_main.render("HIDDEN LAYER (16-Tanh)", True, (148, 163, 184))
        self.screen.blit(lbl_h, (panel_x + 15, 185))

        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                act_val = float(hidden_act[0, idx]) if hidden_act.shape[0] > 0 else 0.0
                intensity = min(255, int(abs(act_val) * 220))
                color = (intensity, int(intensity * 0.8), 255) if act_val > 0 else (255, int(intensity * 0.5), intensity)
                if abs(act_val) < 0.05:
                    color = (30, 41, 59)

                rx = panel_x + 25 + col * 46
                ry = 210 + row * 26
                pygame.draw.rect(self.screen, color, (rx, ry, 36, 18), border_radius=3)
                txt_val = self.font_tiny.render(f"{act_val:+.1f}", True, (241, 245, 249) if intensity > 60 else (71, 85, 105))
                self.screen.blit(txt_val, (rx + 4, ry + 2))

        lbl_gps = self.font_main.render("BFS PATH CORRIDOR", True, (148, 163, 184))
        self.screen.blit(lbl_gps, (panel_x + 15, 335))
        compass_cx, compass_cy = panel_x + 115, 410
        pygame.draw.circle(self.screen, (20, 26, 38), (compass_cx, compass_cy), 45)
        pygame.draw.circle(self.screen, (56, 189, 248), (compass_cx, compass_cy), 45, 1)

        vx, vy = float(obs[0, 7]), float(obs[0, 8])
        end_x = compass_cx + int(vx * 40)
        end_y = compass_cy + int(vy * 40)
        pygame.draw.line(self.screen, (0, 240, 255), (compass_cx, compass_cy), (end_x, end_y), 3)
        pygame.draw.circle(self.screen, (0, 240, 255), (end_x, end_y), 5)

    def run(self, video_path: str | None = None, max_frames: int | None = None):
        running = True
        alpha_surf = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)

        video_writer = None
        if video_path:
            import imageio
            print(f"[*] Initializing video recorder for: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

        frame_count = 0

        while running:
            if max_frames is not None and frame_count >= max_frames:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_g:
                        self.show_swarm = not self.show_swarm
                    elif event.key == pygame.K_l:
                        self.show_lasers = not self.show_lasers
                    elif event.key == pygame.K_r:
                        self.obs, _ = self.env.reset()
                    elif event.key == pygame.K_e:
                        print(f"[*] Training +10 Generations live in background...")
                        self.ga.evolve_more(self.env_cls, self.maze, generations=10)
                        self.obs, _ = self.env.reset()

            if not self.paused:
                actions = self.ga.forward(self.obs)
                self.obs, rewards, term, trunc, _ = self.env.step(actions)

                if term[0] or trunc[0]:
                    self.obs, _ = self.env.reset()

            c_pos = self.env.state["pos"][0]
            c_vel = self.env.state["vel"][0]
            c_head = float(self.env.state["heading"][0])
            c_target = self.env.state["target"][0]
            c_alt = float(self.env.state["z"][0])

            speed = float(np.linalg.norm(c_vel))
            vel_h = float(np.arctan2(c_vel[1], c_vel[0])) if speed > 0.05 else c_head
            slip_deg = abs(float(np.degrees((vel_h - c_head + np.pi) % (2 * np.pi) - np.pi)))

            if speed > 0.12 and slip_deg > 14.0 and c_alt <= 0.05:
                self.spawn_drift_smoke(c_pos[0] * self.cell_size, c_pos[1] * self.cell_size, c_head, speed)

            self.update_particles()
            self.screen.fill((13, 17, 23))

            # 1. Maze Arena
            for y in range(self.rows):
                for x in range(self.cols):
                    if self.maze[y, x] == 1:
                        rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        pygame.draw.rect(self.screen, (30, 41, 59), rect)
                        pygame.draw.rect(self.screen, (15, 23, 42), rect, 1)

            # 2. Target Goal
            gx, gy = int(c_target[0] * self.cell_size), int(c_target[1] * self.cell_size)
            pulse = int(math.sin(time.time() * 6.0) * 3)
            pygame.draw.circle(self.screen, (16, 185, 129), (gx, gy), 16 + pulse, 2)
            pygame.draw.circle(self.screen, (52, 211, 153), (gx, gy), 10)

            # 3. Tire Drift Smoke
            self.draw_particles(self.screen)

            # 4. Ghost Swarm
            if self.show_swarm:
                alpha_surf.fill((0, 0, 0, 0))
                for i in range(1, self.swarm_size):
                    sp = self.env.state["pos"][i]
                    sz = float(self.env.state["z"][i])
                    sx, sy = int(sp[0] * self.cell_size), int(sp[1] * self.cell_size) - int(sz * 20)
                    pygame.draw.circle(alpha_surf, (236, 72, 153, 90), (sx, sy), 5)
                self.screen.blit(alpha_surf, (0, 0))

            # 5. Raycasts
            ax, ay = int(c_pos[0] * self.cell_size), int(c_pos[1] * self.cell_size)
            if self.show_lasers:
                angles = c_head + np.linspace(-2.094 / 2, 2.094 / 2, 5)
                for i, a in enumerate(angles):
                    ray_len = float(self.obs[0, i]) * 6.5 * self.cell_size
                    ex = int(c_pos[0] * self.cell_size + math.cos(a) * ray_len)
                    ey = int(c_pos[1] * self.cell_size + math.sin(a) * ray_len)
                    laser_c = (244, 63, 94) if float(self.obs[0, i]) < 0.35 else (251, 113, 133)
                    pygame.draw.line(self.screen, laser_c, (ax, ay), (ex, ey), 1)
                    pygame.draw.circle(self.screen, laser_c, (ex, ey), 3)

            # 6. Car Body
            lift_y = int(c_alt * 24)
            scale = 1.0 + c_alt * 0.18

            shadow_surf = pygame.Surface((32, 20), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0, 0, 0, 120), (0, 0, 32, 20))
            self.screen.blit(shadow_surf, (ax - 16, ay - 10))

            car_len, car_wid = int(16 * scale), int(10 * scale)
            ca, sa = math.cos(c_head), math.sin(c_head)
            car_center_y = ay - lift_y

            corners = [
                (ax + ca * car_len - sa * car_wid, car_center_y + sa * car_len + ca * car_wid),
                (ax - ca * car_len - sa * car_wid, car_center_y - sa * car_len + ca * car_wid),
                (ax - ca * car_len + sa * car_wid, car_center_y - sa * car_len - ca * car_wid),
                (ax + ca * car_len + sa * car_wid, car_center_y + sa * car_len - ca * car_wid),
            ]
            car_color = (192, 132, 252) if c_alt > 0.05 else (0, 229, 255)
            pygame.draw.polygon(self.screen, car_color, corners)
            pygame.draw.polygon(self.screen, (255, 255, 255), corners, 2)

            hx = int(ax + ca * (car_len + 4))
            hy = int(car_center_y + sa * (car_len + 4))
            pygame.draw.line(self.screen, (255, 255, 255), (ax, car_center_y), (hx, hy), 3)

            if speed > 0.05:
                vx_end = int(ax + c_vel[0] * 38)
                vy_end = int(car_center_y + c_vel[1] * 38)
                pygame.draw.line(self.screen, (250, 204, 21), (ax, car_center_y), (vx_end, vy_end), 2)

            self.draw_hud(actions, speed, slip_deg, float(self.obs[0, 6]), c_alt)
            self.draw_side_panel(self.obs, self.ga.last_hidden)

            pygame.display.flip()

            if video_writer is not None:
                frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
                video_writer.append_data(frame)

            frame_count += 1
            if video_writer is None:
                self.clock.tick(60)

        if video_writer is not None:
            video_writer.close()
            print(f"[+] Video successfully written to: {video_path} ({frame_count} frames)")

        pygame.quit()


# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NevoRL Autonomous Complex Labyrinth [3D Airtime + HUD]")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to specified path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600 = 10s)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    parser.add_argument("--pop-size", type=int, default=1024, help="Evolution population size (default: 1024)")
    args = parser.parse_args()

    print("1. Parsing Monospace Equal-Width ASCII Labyrinth (30x24)...")
    maze, start_pt, goal_pt = parse_ascii_track(ASCII_CIRCUIT)
    print(f"   Track Dimensions : {maze.shape[1]}x{maze.shape[0]} cells")
    print(f"   Start Spawn Point: {start_pt}")
    print(f"   Goal Target Point: {goal_pt}")

    print("\n2. Compiling NevoRL 3D Jumping Environment...")
    env_src = build_environment_source(start_pt, goal_pt)
    compiler = NevoRLCompiler()
    CyberArenaCls = compiler.compile_source(env_src)

    print(f"\n3. Evolving {args.pop_size:,} agents ({args.generations} Generations)...")
    train_envs = CyberArenaCls(num_envs=args.pop_size, grid_map=maze)
    ga = FastNeuroEvolution(pop_size=args.pop_size, in_dim=14, out_dim=3)

    elapsed, sps, top_fit = ga.train_epoch(train_envs, generations=args.generations, rollout_steps=340, verbose=True)
    print(f"\n   Done in {elapsed:.2f}s! ({sps:,.0f} agent-steps/sec)")
    print(f"   Champion Fitness: {top_fit:.1f}")

    print("\n4. Launching CyberVisualizer...")
    viz = CyberVisualizer(CyberArenaCls, maze, ga, start_pt, goal_pt)
    viz.run(video_path=args.video, max_frames=args.frames if args.video else None)


if __name__ == "__main__":
    main()

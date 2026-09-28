# demo.py
from __future__ import annotations
import sys
sys.path.append(r'H:\py')
import os
import sys
import time
import math
import random
import argparse
from typing import Tuple, List, Optional
import numpy as np

# Headless SDL video driver fallback for CI / headless servers
if "DISPLAY" not in os.environ and sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from nevorl import NevoRLCompiler

# =====================================================================
# 1. PROCEDURAL LABYRINTH & ASCII CIRCUIT PARSER
# =====================================================================
ASCII_CIRCUIT = """
##############################
#XXXXXXXX#XXXXXXXXXXXXXXXXXX##
#X############################
#X##########################X#
#XXXX#######################X#
#XXXX#######################X#
#XXXX#########################
#XXXX#######################X#
#XXXX#######################X#
#XXXX#########################
#XXXX#######################X#
#XXXX#######################X#
##############################
##############################
#X############################
#X############################
#X############################
#X##########################X#
#X##########################X#
#X##########################X#
#X##########################X#
#X##########################X#
#S##########################G#
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


def generate_procedural_maze(width: int = 30, height: int = 24, seed: Optional[int] = None) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.ones((height, width), dtype=np.uint8)

    for x in range(1, width - 1):
        grid[1, x] = 0
        grid[height - 2, x] = 0
    for y in range(1, height - 1):
        grid[y, 1] = 0
        grid[y, width - 2] = 0

    stack = [(3, 3)]
    grid[3, 3] = 0

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = cx + dx, cy + dy
            if 2 <= nx < width - 2 and 2 <= ny < height - 2 and grid[ny, nx] == 1:
                neighbors.append((nx, ny, cx + dx // 2, cy + dy // 2))

        if neighbors:
            nx, ny, wx, wy = random.choice(neighbors)
            grid[wy, wx] = 0
            grid[ny, nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    for _ in range(int(width * height * 0.06)):
        rx = random.randint(2, width - 3)
        ry = random.randint(2, height - 3)
        grid[ry, rx] = 0

    start_pos = (1.5, 1.5)
    goal_pos = (width - 2.5, height - 2.5)
    grid[int(start_pos[1]), int(start_pos[0])] = 0
    grid[int(goal_pos[1]), int(goal_pos[0])] = 0

    return grid, start_pos, goal_pos


# =====================================================================
# 2. NEVORL ENVIRONMENT (Exploration & Safe-Flight Physics)
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

        // Pure Bird's-Eye View: Euclidean vector to Goal Beacon (no maze graph / BFS)
        let to_target = state.target - state.pos;
        let target_dist = length(to_target);
        let target_dir = to_target / (target_dist + 0.0001);

        let ego_target = rotate(target_dir, -state.heading);
        let ego_vel = rotate(state.vel, -state.heading);

        let speed = length(state.vel);
        let can_jump = where((state.z <= 0.02) and (speed > 0.18), 1.0, 0.0);

        // Dim = 15:
        // [0..4]: rays, [5]: heading, [6]: target_dist, [7..8]: target_dir, [9]: z
        // [10]: ego_target.x, [11]: ego_target.y, [12]: ego_vel.x, [13]: ego_vel.y, [14]: can_jump
        return [rays, state.heading, target_dist, target_dir.x, target_dir.y, state.z, ego_target.x, ego_target.y, ego_vel.x, ego_vel.y, can_jump];
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
        let speed = length(state.vel);
        let high_speed = speed > 0.18;
        let do_jump = on_ground and wants_jump and high_speed;

        // Launch kinematics: vz=0.40, gravity=-0.028
        state.vz = where(do_jump, 0.40, state.vz - 0.028);
        state.z = clamp(state.z + state.vz, 0.0, 3.5);
        state.vz = where(state.z <= 0.0, 0.0, state.vz);

        let hit = kinematics_car(act[0], act[1], 0.85, 0.88, 0.45, 0.40, 0.05, 0.28, state.z);

        let touching_wall = is_wall(state.pos, 0.28);
        let landed_on_wall = (state.z <= 0.08) and touching_wall;

        state.crashed = where(hit or landed_on_wall, 1.0, 0.0);
        state.steps = state.steps + 1;
    }}

    reward {{
        let to_target = state.target - state.pos;
        let target_dist = length(to_target);
        let target_dir = to_target / (target_dist + 0.0001);
        let speed = length(state.vel);

        // Continuous progress incentives: forward velocity toward beacon
        let beacon_vel = state.vel.x * target_dir.x + state.vel.y * target_dir.y;
        let progress_reward = beacon_vel * 14.0;
        let speed_bonus = speed * 8.0;

        // Alive reward for clean ground driving
        let alive_bonus = where(state.crashed < 0.5, 0.8, 0.0);

        let reached_goal = (target_dist < 1.8) and (state.z <= 0.08);
        let goal_bonus = where(reached_goal, 10000.0, 0.0);

        // Strict penalty for wall collisions and bad landings (no free airtime points!)
        let crash_tax = where(state.crashed > 0.5, 260.0, 0.0);

        return progress_reward + speed_bonus + alive_bonus + goal_bonus - crash_tax;
    }}

    terminal {{
        let to_target = state.target - state.pos;
        let target_dist = length(to_target);
        return ((target_dist < 1.8) and (state.z <= 0.08)) or (state.crashed > 0.5) or (state.steps >= 340);
    }}
}}
"""

# =====================================================================
# 3. ANTITHETIC EVOLUTION STRATEGY (Exploration & Safe Jumps)
# =====================================================================
class FastNeuroEvolution:
    def __init__(self, pop_size=1024, in_dim=16, out_dim=3):
        assert pop_size % 2 == 0, "Population size must be even for antithetic sampling"
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.generation = 0

        # Master Policy Weights
        self.mW1 = np.random.randn(in_dim, 32).astype(np.float32) * 0.02
        self.mb1 = np.zeros(32, dtype=np.float32)
        self.mW2 = np.random.randn(32, 16).astype(np.float32) * 0.02
        self.mb2 = np.zeros(16, dtype=np.float32)
        self.mW3 = np.random.randn(16, out_dim).astype(np.float32) * 0.02
        self.mb3 = np.zeros(out_dim, dtype=np.float32)

        self.vW1 = np.zeros_like(self.mW1)
        self.vb1 = np.zeros_like(self.mb1)
        self.vW2 = np.zeros_like(self.mW2)
        self.vb2 = np.zeros_like(self.mb2)
        self.vW3 = np.zeros_like(self.mW3)
        self.vb3 = np.zeros_like(self.mb3)

        # 1. Steering Prior: Strong wall avoidance + beacon seeking
        self.mW1[11, 0] = 2.0     # ego_target.y -> steer towards beacon
        self.mW1[0, 0] = -2.2     # Strong left outer ray repulsion
        self.mW1[1, 0] = -3.4     # Strong left inner ray repulsion
        self.mW1[3, 0] = 3.4      # Strong right inner ray repulsion
        self.mW1[4, 0] = 2.2      # Strong right outer ray repulsion
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.6

        # 2. Racing Throttle: High drive when forward path is clear
        self.mW1[2, 1] = 2.4      # center clearance boosts throttle
        self.mW1[10, 1] = 1.2     # forward target component
        self.mW2[1, 1] = 1.6
        self.mW3[1, 1] = 1.4
        self.mb3[1] = 0.95        # baseline forward throttle

        # 3. Controlled Wall Launch Shortcut:
        # Require forward momentum and clear launch opportunity, suppress point-blank crashes
        self.mW1[15, 2] = 2.4     # safe jump opportunity signal
        self.mW1[14, 2] = 1.6     # can_jump
        self.mW1[2, 2] = -1.8     # Suppress jump if center wall is <0.20 (avoids point-blank suicides)
        self.mb1[2] = -0.2
        self.mW2[2, 2] = 1.8
        self.mW3[2, 2] = 1.6
        self.mb3[2] = -0.65       # Negative baseline bias: jumps must be calculated, not spam

        self.W1 = np.zeros((pop_size, in_dim, 32), dtype=np.float32)
        self.b1 = np.zeros((pop_size, 32), dtype=np.float32)
        self.W2 = np.zeros((pop_size, 32, 16), dtype=np.float32)
        self.b2 = np.zeros((pop_size, 16), dtype=np.float32)
        self.W3 = np.zeros((pop_size, 16, out_dim), dtype=np.float32)
        self.b3 = np.zeros((pop_size, out_dim), dtype=np.float32)

        self.sigma = 0.08
        self.last_hidden = np.zeros((pop_size, 16), dtype=np.float32)
        self._sample_antithetic_population()

    def _sample_antithetic_population(self):
        half = self.pop_size // 2
        self.eps_W1 = np.random.randn(half, self.in_dim, 32).astype(np.float32)
        self.eps_b1 = np.random.randn(half, 32).astype(np.float32)
        self.eps_W2 = np.random.randn(half, 32, 16).astype(np.float32)
        self.eps_b2 = np.random.randn(half, 16).astype(np.float32)
        self.eps_W3 = np.random.randn(half, 16, self.out_dim).astype(np.float32)
        self.eps_b3 = np.random.randn(half, self.out_dim).astype(np.float32)

        self.W1[:half] = self.mW1 + self.sigma * self.eps_W1
        self.W1[half:] = self.mW1 - self.sigma * self.eps_W1
        self.b1[:half] = self.mb1 + self.sigma * self.eps_b1
        self.b1[half:] = self.mb1 - self.sigma * self.eps_b1

        self.W2[:half] = self.mW2 + self.sigma * self.eps_W2
        self.W2[half:] = self.mW2 - self.sigma * self.eps_W2
        self.b2[:half] = self.mb2 + self.sigma * self.eps_b2
        self.b2[half:] = self.mb2 - self.sigma * self.eps_b2

        self.W3[:half] = self.mW3 + self.sigma * self.eps_W3
        self.W3[half:] = self.mW3 - self.sigma * self.eps_W3
        self.b3[:half] = self.mb3 + self.sigma * self.eps_b3
        self.b3[half:] = self.mb3 - self.sigma * self.eps_b3

        self.W1[0] = self.mW1
        self.b1[0] = self.mb1
        self.W2[0] = self.mW2
        self.b2[0] = self.mb2
        self.W3[0] = self.mW3
        self.b3[0] = self.mb3

    def forward(self, obs: np.ndarray) -> np.ndarray:
        N = obs.shape[0]

        # Calculate calculated jump opportunity with approach speed and clear runway
        center_ray = obs[:, 2]
        ego_target_x = obs[:, 10]
        ego_vel_x = obs[:, 12]
        can_jump = obs[:, 14]

        # Trigger readiness only when speed is high, heading roughly toward target, and barrier is detected
        jump_opp = (
            (ego_target_x > 0.15) &
            (ego_vel_x > 0.22) &
            (center_ray < 0.65) &
            (center_ray > 0.18) &
            (can_jump > 0.5)
        ).astype(np.float32)[:, None]

        obs_16 = np.hstack([obs, jump_opp])

        w1, b1 = self.W1[:N], self.b1[:N]
        w2, b2 = self.W2[:N], self.b2[:N]
        w3, b3 = self.W3[:N], self.b3[:N]

        h1 = np.tanh(np.matmul(obs_16[:, None, :], w1).squeeze(1) + b1)
        h2 = np.tanh(np.matmul(h1[:, None, :], w2).squeeze(1) + b2)
        self.last_hidden = h2

        raw_out = np.tanh(np.matmul(h2[:, None, :], w3).squeeze(1) + b3)

        steer = raw_out[:, 0:1]
        throttle = np.clip(0.60 + 0.40 * raw_out[:, 1:2], 0.35, 1.0)
        jump = raw_out[:, 2:3]

        return np.hstack([steer, throttle, jump]).astype(np.float32)

    def evolve(self, fitness: np.ndarray):
        self.generation += 1
        half = self.pop_size // 2

        fit_norm = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-6)
        diff = (fit_norm[:half] - fit_norm[half:])[:, None, None]
        diff_b = (fit_norm[:half] - fit_norm[half:])[:, None]

        gW1 = np.mean(diff * self.eps_W1, axis=0)
        gb1 = np.mean(diff_b * self.eps_b1, axis=0)
        gW2 = np.mean(diff * self.eps_W2, axis=0)
        gb2 = np.mean(diff_b * self.eps_b2, axis=0)
        gW3 = np.mean(diff * self.eps_W3, axis=0)
        gb3 = np.mean(diff_b * self.eps_b3, axis=0)

        lr, beta = 0.04, 0.85
        self.vW1 = beta * self.vW1 + lr * gW1
        self.vb1 = beta * self.vb1 + lr * gb1
        self.vW2 = beta * self.vW2 + lr * gW2
        self.vb2 = beta * self.vb2 + lr * gb2
        self.vW3 = beta * self.vW3 + lr * gW3
        self.vb3 = beta * self.vb3 + lr * gb3

        self.mW1 += self.vW1
        self.mb1 += self.vb1
        self.mW2 += self.vW2
        self.mb2 += self.vb2
        self.mW3 += self.vW3
        self.mb3 += self.vb3

        top_idx = int(np.argmax(fitness))
        if fitness[top_idx] > fitness[0]:
            self.mW1 = 0.8 * self.mW1 + 0.2 * self.W1[top_idx]
            self.mb1 = 0.8 * self.mb1 + 0.2 * self.b1[top_idx]
            self.mW2 = 0.8 * self.mW2 + 0.2 * self.W2[top_idx]
            self.mb2 = 0.8 * self.mb2 + 0.2 * self.b2[top_idx]
            self.mW3 = 0.8 * self.mW3 + 0.2 * self.W3[top_idx]
            self.mb3 = 0.8 * self.mb3 + 0.2 * self.b3[top_idx]

        self.sigma = max(0.03, self.sigma * 0.985)
        self._sample_antithetic_population()

    def train_epoch(self, envs, generations=60, rollout_steps=340, verbose=True):
        t0 = time.perf_counter()
        top_fit = -9999.0

        for gen in range(generations):
            obs, _ = envs.reset()
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            alive = np.ones(self.pop_size, dtype=bool)
            completed = np.zeros(self.pop_size, dtype=bool)
            steps_to_goal = np.full(self.pop_size, rollout_steps, dtype=np.int32)
            max_altitude = np.zeros(self.pop_size, dtype=np.float32)

            init_dist = obs[:, 6].copy()
            min_dist = init_dist.copy()
            start_xy = envs.state["pos"].copy()

            # Dynamic jump tracking for safe landing & anti back-and-forth enforcement
            in_air = np.zeros(self.pop_size, dtype=bool)
            takeoff_pos = np.zeros((self.pop_size, 2), dtype=np.float32)
            last_land_pos = start_xy.copy()
            total_jump_bonus = np.zeros(self.pop_size, dtype=np.float32)
            safe_jumps = np.zeros(self.pop_size, dtype=np.int32)
            visited_cells = [set() for _ in range(self.pop_size)]

            for step_idx in range(rollout_steps):
                act = self.forward(obs)
                obs, rewards, term, trunc, _ = envs.step(act)

                cur_pos = envs.state["pos"]
                cur_z = envs.state["z"]
                crashed = envs.state["crashed"] > 0.5
                cur_dist = obs[:, 6]

                max_altitude = np.maximum(max_altitude, cur_z)

                # Track grid cell exploration
                for i in range(self.pop_size):
                    if alive[i]:
                        gx, gy = int(cur_pos[i, 0]), int(cur_pos[i, 1])
                        visited_cells[i].add((gx, gy))

                # 1. Detect Takeoff
                takeoff_now = alive & (~in_air) & (cur_z > 0.08)
                takeoff_pos[takeoff_now] = cur_pos[takeoff_now]
                in_air[takeoff_now] = True

                # 2. Detect Landing
                landing_now = alive & in_air & (cur_z <= 0.04)

                # STRICT CRITERIA: Only reward jumps that land safely and don't kill the agent!
                safe_landing = landing_now & (~crashed)

                if np.any(safe_landing):
                    jump_disp = cur_pos[safe_landing] - takeoff_pos[safe_landing]
                    jump_dist = np.linalg.norm(jump_disp, axis=1)

                    # Anti-Oscillation / Back-and-Forth Detection
                    prog_from_last = cur_pos[safe_landing] - last_land_pos[safe_landing]
                    net_prog = np.linalg.norm(prog_from_last, axis=1)

                    # Also verify forward distance relative to start
                    dist_to_start_now = np.linalg.norm(cur_pos[safe_landing] - start_xy[safe_landing], axis=1)
                    dist_to_start_prev = np.linalg.norm(takeoff_pos[safe_landing] - start_xy[safe_landing], axis=1)
                    backwards_jump = dist_to_start_now < (dist_to_start_prev - 0.5)

                    is_jitter = (net_prog < 1.8) | backwards_jump

                    # Large reward for legitimate forward distance, heavy penalty for jumping back-and-forth
                    payout = np.where(
                        (~is_jitter) & (jump_dist > 1.5),
                        jump_dist * 50.0 + 120.0,
                        -60.0
                    )

                    total_jump_bonus[safe_landing] += payout
                    safe_jumps[safe_landing] += np.where(~is_jitter, 1, 0)
                    last_land_pos[safe_landing] = cur_pos[safe_landing]

                # Reset flight status
                in_air[landing_now] = False
                in_air[crashed] = False

                reached = alive & (cur_dist < 1.8) & (cur_z <= 0.08)
                newly_done = reached & (~completed)
                completed |= reached
                steps_to_goal = np.where(newly_done, step_idx + 1, steps_to_goal)

                fitness += np.where(alive, rewards, 0.0)
                min_dist = np.where(alive & (cur_dist < min_dist), cur_dist, min_dist)

                alive &= ~(term | trunc)
                if not np.any(alive):
                    break

            # 3. Maze Exploration & Goal Reward Assembly
            progress = np.maximum(0.0, init_dist - min_dist)
            fitness += progress * 110.0

            # Cell exploration bonus encourages mapping corridors without any BFS cheating
            explored_count = np.array([len(s) for s in visited_cells], dtype=np.float32)
            fitness += explored_count * 24.0

            # Add verified safe-landing bonuses
            fitness += total_jump_bonus

            # Finish bonus
            finish_bonus = np.where(completed, 12000.0 + (rollout_steps - steps_to_goal) * 40.0, 0.0)
            fitness += finish_bonus

            top_fit = float(np.max(fitness))
            min_rem_dist = float(np.min(min_dist))
            num_solved = int(np.sum(completed))
            total_safe_jumps = int(np.sum(safe_jumps))

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Closest: {min_rem_dist:4.1f}m | Solved: {num_solved:3d}/{self.pop_size} | Safe Jumps: {total_safe_jumps:3d} | Max Alt: {float(np.max(max_altitude)):.2f}m")

            self.evolve(fitness)

        elapsed = time.perf_counter() - t0
        sps = (self.pop_size * rollout_steps * generations) / max(elapsed, 1e-5)
        return elapsed, sps, top_fit

    def evolve_more(self, env_cls, maze, generations=10):
        envs = env_cls(num_envs=self.pop_size, grid_map=maze)
        self.train_epoch(envs, generations=generations, rollout_steps=340, verbose=True)


# =====================================================================
# 4. ROBUST 3D LOOK-AT CAMERA & PERSPECTIVE RENDERER
# =====================================================================
class LookAtCamera3D:
    def __init__(self, screen_w: int, screen_h: int, fov: float = 580.0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fov = fov

        self.pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.R = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.U = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.F = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def update_look_at(self, eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)):
        self.pos = eye.copy()
        self.target = target.copy()

        fwd = target - eye
        norm_fwd = np.linalg.norm(fwd)
        self.F = fwd / (norm_fwd + 1e-7)

        right = np.cross(self.F, world_up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-5:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= norm_r
        self.R = right
        self.U = np.cross(self.R, self.F)

    def project_point(self, x: float, y: float, z: float) -> Optional[Tuple[int, int]]:
        vx = x - self.pos[0]
        vy = y - self.pos[1]
        vz = z - self.pos[2]

        cam_x = vx * self.R[0] + vy * self.R[1] + vz * self.R[2]
        cam_y = vx * self.U[0] + vy * self.U[1] + vz * self.U[2]
        cam_z = vx * self.F[0] + vy * self.F[1] + vz * self.F[2]

        if cam_z <= 0.25:
            return None

        sx = int(self.screen_w / 2 + (cam_x * self.fov) / cam_z)
        sy = int(self.screen_h / 2 - (cam_y * self.fov) / cam_z)
        return (sx, sy)

    def get_depth(self, x: float, y: float, z: float) -> float:
        vx = x - self.pos[0]
        vy = y - self.pos[1]
        vz = z - self.pos[2]
        return float(vx * self.F[0] + vy * self.F[1] + vz * self.F[2])


class CyberVisualizer3D:
    def __init__(self, env_cls, maze, ga: FastNeuroEvolution, start_pt, goal_pt):
        pygame.init()
        pygame.font.init()

        self.screen_w = 1200
        self.screen_h = 760
        self.hud_h = 100
        self.viewport_h = self.screen_h - self.hud_h

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NevoRL Autonomous Complex Labyrinth [Bird's-Eye AI + 3D View]")
        self.clock = pygame.time.Clock()

        font_names = ["Consolas", "dejavusansmono", "monospace", "courier"]
        self.font_main = pygame.font.SysFont(font_names, 14, bold=True)
        self.font_big = pygame.font.SysFont(font_names, 18, bold=True)
        self.font_tiny = pygame.font.SysFont(font_names, 11)

        self.maze = maze
        self.rows, self.cols = maze.shape[0], maze.shape[1]
        self.ga = ga
        self.env_cls = env_cls
        self.start_pt = start_pt
        self.goal_pt = goal_pt

        self.swarm_size = 40
        self.env = env_cls(num_envs=self.swarm_size, grid_map=maze)
        self.obs, _ = self.env.reset()

        self.camera = LookAtCamera3D(self.screen_w, self.viewport_h, fov=560.0)

        # 0 = 3D Chase Cam, 1 = 3D Isometric Orbit, 2 = 2D Tactical View
        self.cam_mode = 0
        self.cam_eye_smoothed = np.array([start_pt[0] - 3.0, start_pt[1], 2.5], dtype=np.float32)

        self.drift_ribbons: List[List[float]] = []
        self.show_swarm = True
        self.paused = False

    def draw_3d_cube(self, surface: pygame.Surface, gx: int, gy: int):
        H = 1.25
        v = [
            (gx, gy, 0.0), (gx + 1.0, gy, 0.0), (gx + 1.0, gy + 1.0, 0.0), (gx, gy + 1.0, 0.0),
            (gx, gy, H), (gx + 1.0, gy, H), (gx + 1.0, gy + 1.0, H), (gx, gy + 1.0, H),
        ]
        proj = [self.camera.project_point(*pt) for pt in v]
        if any(p is None for p in proj):
            return

        top_face = [proj[4], proj[5], proj[6], proj[7]]
        pygame.draw.polygon(surface, (25, 34, 56), top_face)
        pygame.draw.polygon(surface, (56, 189, 248), top_face, 2)

        if self.camera.pos[1] > gy + 1.0:
            south_face = [proj[7], proj[6], proj[2], proj[3]]
            pygame.draw.polygon(surface, (18, 25, 43), south_face)
            pygame.draw.line(surface, (30, 41, 59), proj[7], proj[6], 1)
        elif self.camera.pos[1] < gy:
            north_face = [proj[4], proj[5], proj[1], proj[0]]
            pygame.draw.polygon(surface, (14, 20, 36), north_face)
            pygame.draw.line(surface, (30, 41, 59), proj[4], proj[5], 1)

        if self.camera.pos[0] > gx + 1.0:
            east_face = [proj[5], proj[6], proj[2], proj[1]]
            pygame.draw.polygon(surface, (22, 30, 50), east_face)
        elif self.camera.pos[0] < gx:
            west_face = [proj[4], proj[7], proj[3], proj[0]]
            pygame.draw.polygon(surface, (16, 22, 38), west_face)

    def draw_3d_car(self, surface: pygame.Surface, pos: np.ndarray, heading: float, z: float, vz: float, steer: float, color=(0, 229, 255)):
        pitch = math.atan2(vz, 0.38) * 0.45
        roll = -steer * 0.40

        local_pts = [
            (0.55, 0.0, 0.08),       # 0: Nose
            (0.32, -0.22, 0.16),     # 1: Left hood
            (0.32, 0.22, 0.16),      # 2: Right hood
            (-0.05, -0.24, 0.32),    # 3: Left roof
            (-0.05, 0.24, 0.32),     # 4: Right roof
            (-0.55, -0.28, 0.20),    # 5: Left rear
            (-0.55, 0.28, 0.20),     # 6: Right rear
            (-0.65, -0.32, 0.38),    # 7: Left spoiler
            (-0.65, 0.32, 0.38),     # 8: Right spoiler
        ]

        ch, sh = math.cos(heading), math.sin(heading)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        proj_pts = []
        for lx, ly, lz in local_pts:
            x1 = lx * cp - lz * sp
            y1 = ly * cr - (lx * sp + lz * cp) * sr
            z1 = ly * sr + (lx * sp + lz * cp) * cr

            wx = pos[0] + (x1 * ch - y1 * sh)
            wy = pos[1] + (x1 * sh + y1 * ch)
            wz = z + z1

            p = self.camera.project_point(wx, wy, wz)
            proj_pts.append(p)

        if any(p is None for p in proj_pts):
            return

        hood = [proj_pts[0], proj_pts[1], proj_pts[2]]
        cockpit = [proj_pts[1], proj_pts[2], proj_pts[4], proj_pts[3]]
        rear = [proj_pts[3], proj_pts[4], proj_pts[6], proj_pts[5]]

        pygame.draw.polygon(surface, color, hood)
        pygame.draw.polygon(surface, (255, 255, 255), hood, 1)

        cabin_c = (192, 132, 252) if z > 0.08 else (30, 41, 59)
        pygame.draw.polygon(surface, cabin_c, cockpit)
        pygame.draw.polygon(surface, (255, 255, 255), cockpit, 1)

        pygame.draw.polygon(surface, (15, 23, 42), rear)
        pygame.draw.polygon(surface, color, rear, 1)

        pygame.draw.line(surface, (244, 63, 94), proj_pts[5], proj_pts[7], 3)
        pygame.draw.line(surface, (244, 63, 94), proj_pts[6], proj_pts[8], 3)
        pygame.draw.line(surface, (244, 63, 94), proj_pts[7], proj_pts[8], 2)

        if z > 0.05:
            sp = self.camera.project_point(pos[0], pos[1], 0.02)
            if sp:
                r = int(max(4, 16 - z * 3))
                shadow_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, (0, 0, 0, 130), (0, 0, r * 2, r * 2))
                surface.blit(shadow_surf, (sp[0] - r, sp[1] - r))

    def draw_ribbons(self, surface: pygame.Surface):
        survivors = []
        for r in self.drift_ribbons:
            r[3] -= 0.04
            if r[3] > 0:
                survivors.append(r)
                p = self.camera.project_point(r[0], r[1], r[2])
                if p:
                    alpha = int((r[3] / r[4]) * 220)
                    pygame.draw.circle(surface, (192, 132, 252, alpha), p, max(1, int(3 * (r[3] / r[4]))))
        self.drift_ribbons = survivors

    def draw_pip_minimap(self, surface: pygame.Surface, c_pos: np.ndarray, c_target: np.ndarray):
        mw, mh = 180, 144
        mx, my = self.screen_w - mw - 20, 20
        pip_surf = pygame.Surface((mw, mh))
        pip_surf.fill((15, 23, 42))

        cw, ch = mw / self.cols, mh / self.rows
        for y in range(self.rows):
            for x in range(self.cols):
                if self.maze[y, x] == 1:
                    pygame.draw.rect(pip_surf, (30, 41, 59), (int(x * cw), int(y * ch), int(cw) + 1, int(ch) + 1))

        tx, ty = int(c_target[0] * cw), int(c_target[1] * ch)
        pygame.draw.circle(pip_surf, (16, 185, 129), (tx, ty), 4)

        cx, cy = int(c_pos[0] * cw), int(c_pos[1] * ch)
        pygame.draw.circle(pip_surf, (0, 240, 255), (cx, cy), 3)

        pygame.draw.rect(pip_surf, (56, 189, 248), (0, 0, mw, mh), 2)
        surface.blit(pip_surf, (mx, my))
        surface.blit(self.font_tiny.render("BIRD'S-EYE RADAR [PiP]", True, (56, 189, 248)), (mx + 8, my + 6))

    def draw_2d_tactical_view(self, surface: pygame.Surface, c_pos: np.ndarray, c_head: float, c_target: np.ndarray, c_alt: float):
        cell_size = min((self.screen_w - 300) / self.cols, self.viewport_h / self.rows)
        off_x = 40
        off_y = 20

        for y in range(self.rows):
            for x in range(self.cols):
                rect = (int(off_x + x * cell_size), int(off_y + y * cell_size), int(cell_size), int(cell_size))
                if self.maze[y, x] == 1:
                    pygame.draw.rect(surface, (30, 41, 59), rect)
                    pygame.draw.rect(surface, (15, 23, 42), rect, 1)

        gx, gy = int(off_x + c_target[0] * cell_size), int(off_y + c_target[1] * cell_size)
        pygame.draw.circle(surface, (16, 185, 129), (gx, gy), 14, 2)
        pygame.draw.circle(surface, (52, 211, 153), (gx, gy), 8)

        ax, ay = int(off_x + c_pos[0] * cell_size), int(off_y + c_pos[1] * cell_size)
        car_len = int(14 * (1.0 + c_alt * 0.2))
        hx = int(ax + math.cos(c_head) * car_len)
        hy = int(ay + math.sin(c_head) * car_len)

        car_c = (192, 132, 252) if c_alt > 0.05 else (0, 240, 255)
        pygame.draw.circle(surface, car_c, (ax, ay), 6)
        pygame.draw.line(surface, (255, 255, 255), (ax, ay), (hx, hy), 3)

    def draw_side_panel(self, obs, hidden_act):
        panel_x = self.screen_w - 240
        panel_w = 240
        pygame.draw.rect(self.screen, (15, 23, 42), (panel_x, 0, panel_w, self.viewport_h))
        pygame.draw.line(self.screen, (30, 41, 59), (panel_x, 0), (panel_x, self.viewport_h), 2)

        txt = self.font_big.render("NEURAL MONITOR", True, (56, 189, 248))
        self.screen.blit(txt, (panel_x + 15, 16))

        lbl_rays = self.font_main.render("LOCAL LIDAR SENSORS", True, (148, 163, 184))
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

        # Bird's-Eye Compass pointing directly to Goal Beacon
        lbl_gps = self.font_main.render("BIRD'S-EYE GPS BEACON", True, (148, 163, 184))
        self.screen.blit(lbl_gps, (panel_x + 15, 335))
        compass_cx, compass_cy = panel_x + 115, 410
        pygame.draw.circle(self.screen, (20, 26, 38), (compass_cx, compass_cy), 45)
        pygame.draw.circle(self.screen, (56, 189, 248), (compass_cx, compass_cy), 45, 1)

        vx, vy = float(obs[0, 7]), float(obs[0, 8])
        end_x = compass_cx + int(vx * 40)
        end_y = compass_cy + int(vy * 40)
        pygame.draw.line(self.screen, (0, 240, 255), (compass_cx, compass_cy), (end_x, end_y), 3)
        pygame.draw.circle(self.screen, (0, 240, 255), (end_x, end_y), 5)

    def draw_hud(self, action, speed, slip_deg, path_dist, altitude):
        hud_y = self.viewport_h
        hud_rect = pygame.Rect(0, hud_y, self.screen_w, self.hud_h)
        pygame.draw.rect(self.screen, (11, 15, 23), hud_rect)
        pygame.draw.line(self.screen, (30, 41, 59), (0, hud_y), (self.screen_w, hud_y), 2)

        txt_gen = self.font_big.render(f"GEN {self.ga.generation:03d} [OpenAI-ES]", True, (56, 189, 248))
        txt_spd = self.font_main.render(f"SPEED     : {speed:4.2f} u/f", True, (241, 245, 249))
        alt_color = (192, 132, 252) if altitude > 0.05 else (148, 163, 184)
        txt_alt = self.font_main.render(f"ALTITUDE  : {altitude:4.2f} m", True, alt_color)
        txt_dist = self.font_main.render(f"DIST TO GOAL: {path_dist:4.1f} m", True, (52, 211, 153))

        self.screen.blit(txt_gen, (25, hud_y + 12))
        self.screen.blit(txt_spd, (25, hud_y + 38))
        self.screen.blit(txt_alt, (25, hud_y + 56))
        self.screen.blit(txt_dist, (25, hud_y + 74))

        steer_val = float(action[0, 0])
        gas_val = float(action[0, 1])
        jump_val = float(action[0, 2])
        pygame.draw.line(self.screen, (30, 41, 59), (280, hud_y + 10), (280, hud_y + 90), 1)

        txt_act = self.font_big.render("AI 3D ACTUATORS", True, (148, 163, 184))
        self.screen.blit(txt_act, (300, hud_y + 12))

        # Steer
        pygame.draw.rect(self.screen, (20, 26, 38), (300, hud_y + 38, 120, 12))
        center_x = 300 + 60
        steer_bar_w = int(steer_val * 58)
        bar_color = (244, 63, 94) if steer_val < 0 else (56, 189, 248)
        pygame.draw.rect(self.screen, bar_color, (center_x if steer_val > 0 else center_x + steer_bar_w, hud_y + 38, abs(steer_bar_w), 12))
        self.screen.blit(self.font_tiny.render(f"STEER [{steer_val:+.2f}]", True, (203, 213, 225)), (430, hud_y + 38))

        # Gas
        pygame.draw.rect(self.screen, (20, 26, 38), (300, hud_y + 56, 120, 12))
        gas_bar_w = int(max(0.0, gas_val) * 120)
        pygame.draw.rect(self.screen, (34, 197, 94), (300, hud_y + 56, gas_bar_w, 12))
        self.screen.blit(self.font_tiny.render(f"GAS   [{gas_val:.2f}]", True, (203, 213, 225)), (430, hud_y + 56))

        # Thruster
        is_firing = jump_val > 0.0
        jump_color = (192, 132, 252) if is_firing else (71, 85, 105)
        jump_state = "FIRING [SAFE LAUNCH]" if is_firing else "GROUND"
        self.screen.blit(self.font_main.render(f"3D THRUSTER: [{jump_state}]", True, jump_color), (300, hud_y + 74))

        # Camera Controls Guide
        pygame.draw.line(self.screen, (30, 41, 59), (600, hud_y + 10), (600, hud_y + 90), 1)
        mode_str = ["3D CHASE CAM", "3D ISOMETRIC ORBIT", "2D TACTICAL RADAR"][self.cam_mode]
        self.screen.blit(self.font_main.render(f"CAMERA: [{mode_str}] (Press 'C')", True, (168, 85, 247)), (620, hud_y + 12))
        keys = [
            "[C]     Toggle Camera (3D Chase / 3D Isometric / 2D)",
            "[SPACE] Pause / Play Simulation",
            "[G]     Toggle Ghost Swarm",
            "[E]     Live Background Training +10 Generations",
        ]
        for i, k in enumerate(keys):
            self.screen.blit(self.font_tiny.render(k, True, (100, 116, 139)), (620, hud_y + 32 + i * 15))

    def run(self, video_path: str | None = None, max_frames: int | None = None):
        running = True
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
                    elif event.key == pygame.K_c:
                        self.cam_mode = (self.cam_mode + 1) % 3
                        print(f"[*] Camera switched to mode: {['3D Chase', '3D Isometric Orbit', '2D Radar'][self.cam_mode]}")
                    elif event.key == pygame.K_g:
                        self.show_swarm = not self.show_swarm
                    elif event.key == pygame.K_e:
                        print("[*] Training +10 Generations live in background...")
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
            c_vz = float(self.env.state["vz"][0])
            steer = float(actions[0, 0])

            speed = float(np.linalg.norm(c_vel))
            vel_h = float(np.arctan2(c_vel[1], c_vel[0])) if speed > 0.05 else c_head
            slip_deg = abs(float(np.degrees((vel_h - c_head + np.pi) % (2 * np.pi) - np.pi)))

            # Spawn 3D Tire Ribbons
            ch, sh = math.cos(c_head), math.sin(c_head)
            self.drift_ribbons.append([c_pos[0] - ch * 0.4 - sh * 0.25, c_pos[1] - sh * 0.4 + ch * 0.25, c_alt + 0.05, 1.0, 1.0])
            self.drift_ribbons.append([c_pos[0] - ch * 0.4 + sh * 0.25, c_pos[1] - sh * 0.4 - ch * 0.25, c_alt + 0.05, 1.0, 1.0])

            self.screen.fill((10, 14, 26))

            if self.cam_mode == 0:  # Mode 0: 3D Chase Camera
                target_eye = np.array([
                    c_pos[0] - math.cos(c_head) * 3.2,
                    c_pos[1] - math.sin(c_head) * 3.2,
                    max(1.4, c_alt + 1.8)
                ], dtype=np.float32)
                self.cam_eye_smoothed += (target_eye - self.cam_eye_smoothed) * 0.25

                look_target = np.array([
                    c_pos[0] + math.cos(c_head) * 1.5,
                    c_pos[1] + math.sin(c_head) * 1.5,
                    c_alt + 0.3
                ], dtype=np.float32)

                self.camera.update_look_at(self.cam_eye_smoothed, look_target)

            elif self.cam_mode == 1:  # Mode 1: 3D Isometric Overview
                iso_eye = np.array([c_pos[0] - 8.0, c_pos[1] - 8.0, 11.5], dtype=np.float32)
                self.cam_eye_smoothed += (iso_eye - self.cam_eye_smoothed) * 0.25
                iso_target = np.array([c_pos[0], c_pos[1], c_alt], dtype=np.float32)
                self.camera.update_look_at(self.cam_eye_smoothed, iso_target)

            if self.cam_mode in (0, 1):
                # Ground Grid Lines
                for gx in range(0, self.cols + 1, 2):
                    p1 = self.camera.project_point(gx, 0.0, 0.0)
                    p2 = self.camera.project_point(gx, self.rows, 0.0)
                    if p1 and p2:
                        pygame.draw.line(self.screen, (20, 27, 45), p1, p2, 1)

                # Depth-sorted wall cubes
                visible_cubes = []
                for gy in range(self.rows):
                    for gx in range(self.cols):
                        if self.maze[gy, gx] == 1:
                            depth = self.camera.get_depth(gx + 0.5, gy + 0.5, 0.6)
                            if 0.5 < depth < 20.0:
                                visible_cubes.append((depth, gx, gy))

                visible_cubes.sort(key=lambda item: item[0], reverse=True)

                for _, gx, gy in visible_cubes:
                    self.draw_3d_cube(self.screen, gx, gy)

                # 3D Goal Beacon Pillar
                gp_base = self.camera.project_point(c_target[0], c_target[1], 0.0)
                gp_top = self.camera.project_point(c_target[0], c_target[1], 4.5)
                if gp_base and gp_top:
                    pygame.draw.line(self.screen, (16, 185, 129), gp_base, gp_top, 4)
                    pygame.draw.circle(self.screen, (52, 211, 153), gp_top, 8)
                    pulse = int(math.sin(time.time() * 8.0) * 6 + 14)
                    pygame.draw.circle(self.screen, (52, 211, 153), gp_base, pulse, 2)

                # 3D Drift Ribbons
                self.draw_ribbons(self.screen)

                # Ghost Swarm (3D)
                if self.show_swarm:
                    for i in range(1, self.swarm_size):
                        sp = self.env.state["pos"][i]
                        sz = float(self.env.state["z"][i])
                        depth = self.camera.get_depth(sp[0], sp[1], sz)
                        if 0.5 < depth < 16.0:
                            p = self.camera.project_point(sp[0], sp[1], sz)
                            if p:
                                pygame.draw.circle(self.screen, (236, 72, 153), p, 3)

                # Champion 3D Car
                self.draw_3d_car(self.screen, c_pos, c_head, c_alt, c_vz, steer, color=(0, 229, 255))

                # PiP Radar
                self.draw_pip_minimap(self.screen, c_pos, c_target)

            else:  # Mode 2: 2D Tactical View
                self.draw_2d_tactical_view(self.screen, c_pos, c_head, c_target, c_alt)

            self.draw_side_panel(self.obs, self.ga.last_hidden)
            self.draw_hud(actions, speed, slip_deg, float(self.obs[0, 6]), c_alt)

            pygame.display.flip()

            if video_writer is not None:
                frame = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
                video_writer.append_data(frame)

            frame_count += 1
            if video_writer is None:
                self.clock.tick(60)

        if video_writer is not None:
            video_writer.close()
            print(f"[+] 3D MP4 video successfully written to: {video_path} ({frame_count} frames)")

        pygame.quit()


# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NevoRL Autonomous Complex Labyrinth [Bird's-Eye AI + 3D View]")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to specified path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600 = 10s)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    parser.add_argument("--pop-size", type=int, default=1024, help="Evolution population size (default: 1024)")
    parser.add_argument("--procedural", action="store_true", help="Generate a random procedural labyrinth instead of static ASCII")
    parser.add_argument("--seed", type=int, default=None, help="Seed for procedural labyrinth generator")
    args = parser.parse_args()

    print("=================================================================")
    print("   NevoRL Embodied AI: Bird's-Eye AI + Autonomous Wall Jumping   ")
    print("=================================================================")

    if args.procedural:
        print(f"1. Generating Procedural Labyrinth (30x24, seed={args.seed})...")
        maze, start_pt, goal_pt = generate_procedural_maze(30, 24, seed=args.seed)
    else:
        print("1. Parsing Monospace Equal-Width ASCII Labyrinth (30x24)...")
        maze, start_pt, goal_pt = parse_ascii_track(ASCII_CIRCUIT)

    print(f"   Track Dimensions : {maze.shape[1]}x{maze.shape[0]} cells")
    print(f"   Start Spawn Point: {start_pt}")
    print(f"   Goal Target Point: {goal_pt}")

    print("\n2. Compiling NevoRL Pure Physics Environment (Zero BFS)...")
    env_src = build_environment_source(start_pt, goal_pt)
    compiler = NevoRLCompiler()
    CyberArenaCls = compiler.compile_source(env_src)

    print(f"\n3. Evolving {args.pop_size:,} agents via Antithetic ES ({args.generations} Generations)...")
    train_envs = CyberArenaCls(num_envs=args.pop_size, grid_map=maze)
    ga = FastNeuroEvolution(pop_size=args.pop_size, in_dim=16, out_dim=3)

    elapsed, sps, top_fit = ga.train_epoch(train_envs, generations=args.generations, rollout_steps=340, verbose=True)
    print(f"\n   Done in {elapsed:.2f}s! ({sps:,.0f} agent-steps/sec)")
    print(f"   Champion Fitness: {top_fit:.1f}")

    print("\n4. Launching CyberVisualizer3D...")
    print("   Camera Controls:")
    print("     [C]     Toggle View: 3D Chase Cam <-> 3D Isometric Orbit <-> 2D Radar")
    print("     [SPACE] Pause / Play")
    print("     [G]     Toggle Ghost Swarm")
    print("     [E]     Live Background Training +10 Generations\n")

    viz = CyberVisualizer3D(CyberArenaCls, maze, ga, start_pt, goal_pt)
    viz.run(video_path=args.video, max_frames=args.frames if args.video else None)


if __name__ == "__main__":
    main()

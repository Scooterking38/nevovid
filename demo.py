# demo.py
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import Tuple, List, Optional
import numpy as np

try:
    import pyray as pr
except ImportError:
    print("\n[!] 'raylib' is not installed.")
    print("    Please install it using: pip install raylib\n")
    sys.exit(1)


# =====================================================================
# 1. 3D TRACK GEOMETRY (Continuous Multi-Stage Stunt Track)
# =====================================================================
class Track3D:
    def __init__(self):
        self.width = 4.0  # Road width (X in [-2.0, +2.0])
        self.start_pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.goal_pos = np.array([0.0, 3.8, 33.0], dtype=np.float32)

        # Stage landmarks
        self.hurdle_z = 7.0       # Red laser hurdle (Y in [0.5, 1.3])
        self.ramp_start_z = 11.0  # Smooth incline begins
        self.ramp_end_z = 18.0    # Takeoff lip (Height: Y = 2.8m)
        self.chasm_end_z = 22.5   # Receiving deck (Void gap: 4.5m)
        self.finish_z = 34.0      # Track boundary

    def evaluate_ground(self, x: float, z: float) -> Tuple[float, float, bool]:
        """
        Calculates exact ground height Y, surface slope angle, and whether in void.
        """
        # Fell off track sides
        if abs(x) > (self.width / 2.0 + 0.3):
            return -100.0, 0.0, True

        # Stage 1: Ground Straightaway (Z: 0 -> 11m)
        if 0.0 <= z < self.ramp_start_z:
            return 0.5, 0.0, False

        # Stage 2: Incline Launch Ramp (Z: 11 -> 18m, climbs Y: 0.5 -> 2.8m)
        if self.ramp_start_z <= z < self.ramp_end_z:
            t = (z - self.ramp_start_z) / (self.ramp_end_z - self.ramp_start_z)
            y = 0.5 + t * 2.3
            slope = math.atan2(2.3, 7.0)
            return y, slope, False

        # Stage 3: The Void Chasm (Z: 18 -> 22.5m, no floor!)
        if self.ramp_end_z <= z < self.chasm_end_z:
            return -100.0, 0.0, True

        # Stage 4: Elevated Landing Deck & Goal Summit (Z: 22.5 -> 35m, Y = 2.8m)
        if self.chasm_end_z <= z <= self.finish_z:
            return 2.8, 0.0, False

        return -100.0, 0.0, True


# =====================================================================
# 2. CONTINUOUS 3D RIGID-BODY SPHERE SIMULATOR
# =====================================================================
class MarblePhysics3D:
    def __init__(self, track: Track3D, num_agents: int = 512):
        self.track = track
        self.num_agents = num_agents
        self.radius = 0.40

        self.pos = np.zeros((num_agents, 3), dtype=np.float32)
        self.vel = np.zeros((num_agents, 3), dtype=np.float32)
        self.grounded = np.zeros(num_agents, dtype=bool)
        self.dead = np.zeros(num_agents, dtype=bool)
        self.steps = np.zeros(num_agents, dtype=np.int32)

        self.cleared_hurdle = np.zeros(num_agents, dtype=bool)
        self.cleared_chasm = np.zeros(num_agents, dtype=bool)
        self.reached_goal = np.zeros(num_agents, dtype=bool)

        self.reset()

    def reset(self, indices: Optional[np.ndarray] = None):
        if indices is None:
            indices = np.arange(self.num_agents, dtype=np.int32)
        self.pos[indices] = self.track.start_pos
        self.vel[indices] = 0.0
        self.grounded[indices] = True
        self.dead[indices] = False
        self.steps[indices] = 0
        self.cleared_hurdle[indices] = False
        self.cleared_chasm[indices] = False
        self.reached_goal[indices] = False

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = 1.0 / 60.0
        gravity = 14.0

        steer_x = np.clip(actions[:, 0], -1.0, 1.0)
        throttle_z = np.clip(actions[:, 1], 0.4, 1.0)  # Always committed forward roll
        trigger_jump = actions[:, 2] > 0.0

        # Rolling torque / acceleration
        self.vel[:, 0] += steer_x * 24.0 * dt
        self.vel[:, 2] += throttle_z * 22.0 * dt

        # Realistic ground drag vs air resistance
        self.vel[:, 0] *= 0.965
        self.vel[:, 2] *= 0.992

        # Vertical Launch Impulse: v_y = 6.6 m/s (launches ball 1.55m high, flies 7.5m forward)
        can_jump = self.grounded & trigger_jump
        self.vel[:, 1] = np.where(can_jump, 6.6, self.vel[:, 1] - gravity * dt)

        # Integration
        cand_x = self.pos[:, 0] + self.vel[:, 0] * dt
        cand_y = self.pos[:, 1] + self.vel[:, 1] * dt
        cand_z = self.pos[:, 2] + self.vel[:, 2] * dt

        # Continuous Surface Constraint
        self.grounded[:] = False
        for i in range(self.num_agents):
            floor_y, slope, void = self.track.evaluate_ground(float(cand_x[i]), float(cand_z[i]))

            if not void:
                target_y = floor_y + self.radius
                if cand_y[i] <= target_y:
                    cand_y[i] = target_y
                    self.vel[i, 1] = max(0.0, self.vel[i, 1])
                    self.grounded[i] = True

            if void and cand_y[i] < -2.0:
                self.dead[i] = True

        # Laser Barrier Collision (Z = 7.0m, Height Y: 0.5 -> 1.3m)
        near_laser = np.abs(cand_z - self.track.hurdle_z) < 0.4
        at_laser_h = cand_y < 1.35
        hit_laser = near_laser & at_laser_h & (~self.dead)

        # Hitting laser knocks ball backward
        self.vel[hit_laser, 2] = -3.5

        # Successfully jumping over the laser barrier
        jumped_hurdle = near_laser & (cand_y >= 1.35) & (~self.cleared_hurdle)
        self.cleared_hurdle |= jumped_hurdle

        # Successfully clearing the chasm gap onto receiving deck
        cleared_gap = (cand_z >= self.track.chasm_end_z) & (cand_y >= 2.6) & (~self.cleared_chasm)
        self.cleared_chasm |= cleared_gap

        # Reached Summit Goal Beacon
        to_goal = self.track.goal_pos[None, :] - np.column_stack([cand_x, cand_y, cand_z])
        d_goal = np.linalg.norm(to_goal, axis=-1)
        reached_summit = (d_goal < 2.0) & (~self.reached_goal)
        self.reached_goal |= reached_summit

        self.pos[:, 0] = cand_x
        self.pos[:, 1] = cand_y
        self.pos[:, 2] = cand_z
        self.steps += 1

        # Reward signals
        fwd_rew = self.vel[:, 2] * 12.0
        center_pen = -np.abs(self.pos[:, 0]) * 3.0

        hurdle_bonus = np.where(jumped_hurdle, 2500.0, 0.0)
        hurdle_tax = np.where(hit_laser, 100.0, 0.0)
        chasm_bonus = np.where(cleared_gap, 5000.0, 0.0)
        goal_bonus = np.where(reached_summit, 12000.0, 0.0)
        fall_tax = np.where(self.dead, 150.0, 0.0)

        rewards = fwd_rew + center_pen + hurdle_bonus - hurdle_tax + chasm_bonus + goal_bonus - fall_tax
        dones = self.reached_goal | self.dead | (self.steps >= 350)

        obs = self.get_observations()
        return obs, rewards, dones

    def get_observations(self) -> np.ndarray:
        """
        10-Dimensional Vector:
        [0]: Lane offset (X / 2.0)
        [1]: Altitude (Y / 4.0)
        [2]: Track Progress (Z / 35.0)
        [3..5]: Velocity (vx, vy, vz)
        [6]: Grounded (1.0 = on road)
        [7]: Hurdle Proximity Sensor (spikes to +1.0 within 1.5m before laser)
        [8]: Chasm Proximity Sensor (spikes to +1.0 within 1.5m before takeoff lip)
        [9]: Direction to Goal
        """
        to_goal = self.track.goal_pos[None, :] - self.pos
        dist = np.linalg.norm(to_goal, axis=-1, keepdims=True)
        dir_to_goal = to_goal / (dist + 1e-6)

        # Proximity activation: spikes to 1.0 directly in the takeoff window
        dist_hurdle = self.track.hurdle_z - self.pos[:, 2]
        hurdle_sensor = np.clip(1.0 - np.abs(dist_hurdle - 0.75) / 1.0, 0.0, 1.0)

        dist_chasm = self.track.ramp_end_z - self.pos[:, 2]
        chasm_sensor = np.clip(1.0 - np.abs(dist_chasm - 0.75) / 1.0, 0.0, 1.0)

        obs = np.column_stack([
            self.pos[:, 0] / 2.0,
            self.pos[:, 1] / 4.0,
            self.pos[:, 2] / 35.0,
            self.vel[:, 0] * 0.1,
            self.vel[:, 1] * 0.1,
            self.vel[:, 2] * 0.1,
            self.grounded.astype(np.float32),
            hurdle_sensor,
            chasm_sensor,
            dir_to_goal[:, 2]
        ]).astype(np.float32)

        return obs


# =====================================================================
# 3. ANTITHETIC EVOLUTION STRATEGY (10-DIM AGENT)
# =====================================================================
class NeuroMarblePolicy:
    def __init__(self, in_dim=10, out_dim=3):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.generation = 0

        self.mW1 = np.random.randn(in_dim, 24).astype(np.float32) * 0.02
        self.mb1 = np.zeros(24, dtype=np.float32)
        self.mW2 = np.random.randn(24, 16).astype(np.float32) * 0.02
        self.mb2 = np.zeros(16, dtype=np.float32)
        self.mW3 = np.random.randn(16, out_dim).astype(np.float32) * 0.02
        self.mb3 = np.zeros(out_dim, dtype=np.float32)

        self.vW1 = np.zeros_like(self.mW1)
        self.vb1 = np.zeros_like(self.mb1)
        self.vW2 = np.zeros_like(self.mW2)
        self.vb2 = np.zeros_like(self.mb2)
        self.vW3 = np.zeros_like(self.mW3)
        self.vb3 = np.zeros_like(self.mb3)

        # Prior Wiring: Centerline steering
        self.mW1[0, 0] = -2.5    # Steer opposite to lane offset
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.5

        # Prior Wiring: Forward momentum
        self.mW1[5, 1] = 1.4     # Forward speed
        self.mb1[1] = 0.8
        self.mW2[1, 1] = 1.5
        self.mW3[1, 1] = 1.4
        self.mb3[1] = 1.0        # Continuous forward commitment

        # Prior Wiring: Proximity-triggered jumping
        self.mW1[7, 2] = 3.4     # Hurdle sensor -> JUMP!
        self.mW1[8, 2] = 3.8     # Chasm takeoff sensor -> LAUNCH!
        self.mW1[6, 2] = 1.2     # Grounded check
        self.mb1[2] = -0.4       # Grounded when not near obstacles
        self.mW2[2, 2] = 2.0
        self.mW3[2, 2] = 1.8
        self.mb3[2] = 0.1

        self.pop_size = 512
        self.W1 = np.zeros((self.pop_size, in_dim, 24), dtype=np.float32)
        self.b1 = np.zeros((self.pop_size, 24), dtype=np.float32)
        self.W2 = np.zeros((self.pop_size, 24, 16), dtype=np.float32)
        self.b2 = np.zeros((self.pop_size, 16), dtype=np.float32)
        self.W3 = np.zeros((self.pop_size, 16, out_dim), dtype=np.float32)
        self.b3 = np.zeros((self.pop_size, out_dim), dtype=np.float32)

        self.sigma = 0.08
        self._sample_antithetic()

    def _sample_antithetic(self):
        half = self.pop_size // 2
        self.eW1 = np.random.randn(half, self.in_dim, 24).astype(np.float32)
        self.eb1 = np.random.randn(half, 24).astype(np.float32)
        self.eW2 = np.random.randn(half, 24, 16).astype(np.float32)
        self.eb2 = np.random.randn(half, 16).astype(np.float32)
        self.eW3 = np.random.randn(half, 16, self.out_dim).astype(np.float32)
        self.eb3 = np.random.randn(half, self.out_dim).astype(np.float32)

        self.W1[:half] = self.mW1 + self.sigma * self.eW1
        self.W1[half:] = self.mW1 - self.sigma * self.eW1
        self.b1[:half] = self.mb1 + self.sigma * self.eb1
        self.b1[half:] = self.mb1 - self.sigma * self.eb1

        self.W2[:half] = self.mW2 + self.sigma * self.eW2
        self.W2[half:] = self.mW2 - self.sigma * self.eW2
        self.b2[:half] = self.mb2 + self.sigma * self.eb2
        self.b2[half:] = self.mb2 - self.sigma * self.eb2

        self.W3[:half] = self.mW3 + self.sigma * self.eW3
        self.W3[half:] = self.mW3 - self.sigma * self.eW3
        self.b3[:half] = self.mb3 + self.sigma * self.eb3
        self.b3[half:] = self.mb3 - self.sigma * self.eb3

        # Agent 0 is deterministic master champion
        self.W1[0] = self.mW1
        self.b1[0] = self.mb1
        self.W2[0] = self.mW2
        self.b2[0] = self.mb2
        self.W3[0] = self.mW3
        self.b3[0] = self.mb3

    def forward(self, obs: np.ndarray) -> np.ndarray:
        N = obs.shape[0]
        w1, b1 = self.W1[:N], self.b1[:N]
        w2, b2 = self.W2[:N], self.b2[:N]
        w3, b3 = self.W3[:N], self.b3[:N]

        h1 = np.tanh(np.matmul(obs[:, None, :], w1).squeeze(1) + b1)
        h2 = np.tanh(np.matmul(h1[:, None, :], w2).squeeze(1) + b2)
        out = np.tanh(np.matmul(h2[:, None, :], w3).squeeze(1) + b3)

        steer = out[:, 0:1]
        throttle = 0.50 + 0.50 * out[:, 1:2]
        jump = out[:, 2:3]
        return np.hstack([steer, throttle, jump]).astype(np.float32)

    def evolve(self, fitness: np.ndarray):
        self.generation += 1
        half = self.pop_size // 2

        norm = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-6)
        diff = (norm[:half] - norm[half:])[:, None, None]
        diff_b = (norm[:half] - norm[half:])[:, None]

        gW1 = np.mean(diff * self.eW1, axis=0)
        gb1 = np.mean(diff_b * self.eb1, axis=0)
        gW2 = np.mean(diff * self.eW2, axis=0)
        gb2 = np.mean(diff_b * self.eb2, axis=0)
        gW3 = np.mean(diff * self.eW3, axis=0)
        gb3 = np.mean(diff_b * self.eb3, axis=0)

        lr, beta = 0.05, 0.85
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
            self.mW1 = 0.75 * self.mW1 + 0.25 * self.W1[top_idx]
            self.mb1 = 0.75 * self.mb1 + 0.25 * self.b1[top_idx]
            self.mW2 = 0.75 * self.mW2 + 0.25 * self.W2[top_idx]
            self.mb2 = 0.75 * self.mb2 + 0.25 * self.b2[top_idx]
            self.mW3 = 0.75 * self.mW3 + 0.25 * self.W3[top_idx]
            self.mb3 = 0.75 * self.mb3 + 0.25 * self.b3[top_idx]

        self.sigma = max(0.03, self.sigma * 0.985)
        self._sample_antithetic()

    def train_epoch(self, engine: MarblePhysics3D, generations=60, rollout_steps=320, verbose=True):
        t0 = time.perf_counter()
        top_fit = -9999.0

        for gen in range(generations):
            engine.reset()
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            alive = np.ones(self.pop_size, dtype=bool)
            max_z = np.zeros(self.pop_size, dtype=np.float32)

            obs = engine.get_observations()

            for _ in range(rollout_steps):
                act = self.forward(obs)
                obs, rewards, dones = engine.step(act)

                cur_z = engine.pos[:, 2]
                max_z = np.maximum(max_z, cur_z)

                fitness += np.where(alive, rewards, 0.0)
                alive &= ~dones
                if not np.any(alive):
                    break

            fitness += max_z * 80.0

            top_fit = float(np.max(fitness))
            furthest_z = float(np.max(max_z))
            num_solved = int(np.sum(engine.reached_goal))
            hurdle_cleared = int(np.sum(engine.cleared_hurdle))
            chasm_cleared = int(np.sum(engine.cleared_chasm))

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Max Z: {furthest_z:4.1f}m | Hurdle Jump: {hurdle_cleared:3d}/{self.pop_size} | Chasm Leap: {chasm_cleared:3d}/{self.pop_size} | Solved: {num_solved:3d}/{self.pop_size}")

            self.evolve(fitness)

        elapsed = time.perf_counter() - t0
        sps = (self.pop_size * rollout_steps * generations) / max(elapsed, 1e-5)
        return elapsed, sps, top_fit


# =====================================================================
# 4. HARDWARE RAYLIB 3D VISUALIZER & RIGHT-SIDE-UP VIDEO EXPORTER
# =====================================================================
class CyberMarbleVisualizer:
    def __init__(self, track: Track3D, ga: NeuroMarblePolicy):
        self.track = track
        self.ga = ga
        self.swarm_size = 32
        self.engine = MarblePhysics3D(track, num_agents=self.swarm_size)
        self.engine.reset()

        self.screen_w = 1600
        self.screen_h = 920
        self.hud_h = 120
        self.viewport_h = self.screen_h - self.hud_h

        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_HIGHDPI | pr.FLAG_VSYNC_HINT)
        pr.init_window(self.screen_w, self.screen_h, "CyberMarble 3D: Autonomous Stunt AI Course")
        pr.set_target_fps(60)

        # Chase Camera parameters
        self.cam_smoothed = [track.start_pos[0], 2.8, track.start_pos[2] - 5.5]

        # Standard upright 3D camera (+Y is UP)
        self.camera = pr.Camera3D(
            pr.Vector3(self.cam_smoothed[0], self.cam_smoothed[1], self.cam_smoothed[2]),
            pr.Vector3(track.start_pos[0], track.start_pos[1], track.start_pos[2] + 2.5),
            pr.Vector3(0.0, 1.0, 0.0),
            54.0,
            pr.CAMERA_PERSPECTIVE,
        )

        self.ribbons: List[List[float]] = []
        self.roll_angle = 0.0
        self.show_swarm = True
        self.paused = False

    def draw_3d_track(self):
        # 1. Stage 1 Runway (Z: 0 -> 11m, Y = 0.5)
        pr.draw_cube(pr.Vector3(0.0, 0.25, 5.5), 4.0, 0.5, 11.0, pr.Color(22, 32, 54, 255))
        pr.draw_cube_wires(pr.Vector3(0.0, 0.25, 5.5), 4.0, 0.5, 11.0, pr.Color(0, 220, 255, 255))

        # Lane markings
        for z in range(0, 11, 2):
            pr.draw_line_3d(pr.Vector3(-1.8, 0.51, float(z)), pr.Vector3(1.8, 0.51, float(z)), pr.Color(0, 180, 240, 180))

        # 2. Obstacle 1: The High-Voltage Laser Hurdle (Z = 7.0m)
        pr.draw_cylinder_wires(pr.Vector3(-2.1, 0.25, self.track.hurdle_z), 0.2, 0.2, 1.8, 8, pr.RED)
        pr.draw_cylinder_wires(pr.Vector3(2.1, 0.25, self.track.hurdle_z), 0.2, 0.2, 1.8, 8, pr.RED)
        pulse = abs(math.sin(time.time() * 8.0)) * 0.12
        hurdle_center = pr.Vector3(0.0, 0.90, self.track.hurdle_z)
        pr.draw_cube(hurdle_center, 4.2, 0.45 + pulse, 0.15, pr.Color(255, 30, 80, 220))
        pr.draw_cube_wires(hurdle_center, 4.2, 0.45 + pulse, 0.15, pr.Color(255, 150, 180, 255))

        # 3. Stage 2 Incline Launch Ramp (Z: 11 -> 18m, climbs Y: 0.5 -> 2.8m)
        ramp_center = pr.Vector3(0.0, 1.65, 14.5)
        pr.draw_cube(ramp_center, 4.0, 0.4, 7.0, pr.Color(28, 44, 75, 255))
        pr.draw_cube_wires(ramp_center, 4.0, 0.4, 7.0, pr.Color(0, 240, 255, 255))

        # Ramp climbing arrows
        for z in range(12, 18, 2):
            t = (z - 11.0) / 7.0
            y = 0.5 + t * 2.3 + 0.22
            pr.draw_line_3d(pr.Vector3(-1.5, y, float(z)), pr.Vector3(0.0, y, float(z) + 0.6), pr.Color(0, 255, 200, 255))
            pr.draw_line_3d(pr.Vector3(1.5, y, float(z)), pr.Vector3(0.0, y, float(z) + 0.6), pr.Color(0, 255, 200, 255))

        # Takeoff kicker lip
        kicker_pos = pr.Vector3(0.0, 2.9, 17.8)
        pr.draw_cube(kicker_pos, 4.0, 0.35, 0.4, pr.Color(240, 80, 40, 255))
        pr.draw_cube_wires(kicker_pos, 4.0, 0.35, 0.4, pr.YELLOW)

        # 4. Stage 3 The Void Chasm (Z: 18 -> 22.5m)
        pr.draw_plane(pr.Vector3(0.0, -1.0, 20.25), pr.Vector2(10.0, 4.5), pr.Color(6, 10, 20, 255))
        pr.draw_line_3d(pr.Vector3(-2.2, 2.8, 18.0), pr.Vector3(-2.2, -1.0, 18.0), pr.RED)
        pr.draw_line_3d(pr.Vector3(2.2, 2.8, 18.0), pr.Vector3(2.2, -1.0, 18.0), pr.RED)

        # 5. Stage 4 Receiving Sky-Deck (Z: 22.5 -> 35m, Y = 2.8m)
        deck_center = pr.Vector3(0.0, 2.55, 28.75)
        pr.draw_cube(deck_center, 4.5, 0.5, 12.5, pr.Color(20, 35, 65, 255))
        pr.draw_cube_wires(deck_center, 4.5, 0.5, 12.5, pr.Color(0, 255, 180, 255))

        # 6. Stage 5 Summit Goal Beacon
        beacon_base = pr.Vector3(0.0, 2.8, 33.0)
        pr.draw_cylinder(beacon_base, 0.5, 0.5, 5.5, 16, pr.Color(0, 255, 128, 190))
        pr.draw_sphere(pr.Vector3(0.0, 8.3, 33.0), 0.7, pr.Color(100, 255, 180, 255))
        pr.draw_sphere_wires(pr.Vector3(0.0, 8.3, 33.0), 0.75, 8, 8, pr.WHITE)

        ring_r = 0.5 + float(math.fmod(time.time() * 2.5, 2.5))
        pr.draw_circle_3d(beacon_base, ring_r, pr.Vector3(0, 1, 0), 0.0, pr.Color(0, 255, 128, 220))

    def draw_3d_marble(self, pos: np.ndarray, vel: np.ndarray, grounded: bool):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        fwd_speed = float(vel[2])

        if abs(fwd_speed) > 0.05:
            self.roll_angle += (fwd_speed / self.engine.radius) * (1.0 / 60.0)

        center = pr.Vector3(x, y, z)

        # Glowing Neon Cyan Core
        ball_c = pr.Color(0, 245, 255, 255) if grounded else pr.Color(230, 90, 255, 255)
        pr.draw_sphere(center, self.engine.radius, ball_c)
        pr.draw_sphere_wires(center, self.engine.radius, 10, 10, pr.WHITE)

        # Rotating Gyro-Rings
        pr.draw_circle_3d(center, self.engine.radius + 0.05, pr.Vector3(1, 0, 0), math.degrees(self.roll_angle), pr.Color(255, 220, 50, 240))
        pr.draw_circle_3d(center, self.engine.radius + 0.05, pr.Vector3(0, 0, 1), 0.0, pr.Color(255, 60, 140, 200))

        # Rocket Thruster Flame when Airborne
        if not grounded:
            flame_p = pr.Vector3(x, y - 0.45, z - 0.2)
            pr.draw_sphere(flame_p, 0.20, pr.ORANGE)
            pr.draw_sphere(flame_p, 0.10, pr.YELLOW)

    def draw_ribbons(self):
        survivors = []
        for r in self.ribbons:
            r[3] -= 0.035
            if r[3] > 0:
                survivors.append(r)
                pos = pr.Vector3(float(r[0]), float(r[1]), float(r[2]))
                alpha = int((r[3] / r[4]) * 220)
                pr.draw_sphere(pos, 0.12 * (r[3] / r[4]), pr.Color(220, 80, 255, alpha))
        self.ribbons = survivors

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        video_writer = None
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD 3D Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

        frame_count = 0
        obs = self.engine.get_observations()

        while not pr.window_should_close():
            if max_frames is not None and frame_count >= max_frames:
                break

            if pr.is_key_pressed(pr.KEY_SPACE):
                self.paused = not self.paused
            elif pr.is_key_pressed(pr.KEY_G):
                self.show_swarm = not self.show_swarm
            elif pr.is_key_pressed(pr.KEY_R):
                self.engine.reset()

            if not self.paused:
                actions = self.ga.forward(obs)
                obs, rewards, dones = self.engine.step(actions)
                if dones[0]:
                    self.engine.reset(np.array([0], dtype=np.int32))

            c_pos = self.engine.pos[0]
            c_vel = self.engine.vel[0]
            c_grounded = bool(self.engine.grounded[0])
            speed = float(math.sqrt(float(c_vel[0]**2 + c_vel[2]**2)))

            self.ribbons.append([c_pos[0], c_pos[1] - 0.15, c_pos[2], 1.0, 1.0])

            # Upright 3D Chase Camera tracking (+Y is UP)
            target_x = c_pos[0] - 0.1
            target_y = max(1.8, c_pos[1] + 2.0)
            target_z = c_pos[2] - 5.5

            self.cam_smoothed[0] += (target_x - self.cam_smoothed[0]) * 0.25
            self.cam_smoothed[1] += (target_y - self.cam_smoothed[1]) * 0.25
            self.cam_smoothed[2] += (target_z - self.cam_smoothed[2]) * 0.25

            self.camera.position = pr.Vector3(float(self.cam_smoothed[0]), float(self.cam_smoothed[1]), float(self.cam_smoothed[2]))
            self.camera.target = pr.Vector3(float(c_pos[0]), float(c_pos[1] + 0.4), float(c_pos[2] + 2.5))
            self.camera.up = pr.Vector3(0.0, 1.0, 0.0)

            # -------------------------------------------------------------
            # RENDER 3D SCENE
            # -------------------------------------------------------------
            pr.begin_drawing()
            pr.clear_background(pr.Color(12, 16, 28, 255))

            pr.begin_mode_3d(self.camera)

            # Draw Ground Grid
            pr.draw_grid(36, 1.0)

            # Draw Course
            self.draw_3d_track()

            # Trajectory Ribbons
            self.draw_ribbons()

            # Ghost Swarm
            if self.show_swarm:
                for i in range(1, self.swarm_size):
                    sp = self.engine.pos[i]
                    pr.draw_sphere(pr.Vector3(float(sp[0]), float(sp[1]), float(sp[2])), 0.22, pr.Color(240, 60, 160, 180))

            # Champion Marble
            self.draw_3d_marble(c_pos, c_vel, c_grounded)

            pr.end_mode_3d()

            # -------------------------------------------------------------
            # 2D HUD OVERLAYS
            # -------------------------------------------------------------
            hud_y = self.viewport_h
            pr.draw_rectangle(0, hud_y, self.screen_w, self.hud_h, pr.Color(14, 20, 36, 255))
            pr.draw_line(0, hud_y, self.screen_w, hud_y, pr.Color(0, 220, 255, 255))

            dist_beacon = float(np.linalg.norm(c_pos - self.track.goal_pos))
            pr.draw_text(f"GEN {self.ga.generation:03d} [OpenAI-ES]", 30, hud_y + 16, 26, pr.Color(0, 220, 255, 255))
            pr.draw_text(f"FORWARD SPEED: {speed*3.6:4.1f} km/h", 30, hud_y + 48, 18, pr.WHITE)
            pr.draw_text(f"ELEVATION Y  : {c_pos[1]:4.2f} m", 30, hud_y + 70, 18, pr.Color(230, 140, 255, 255))
            pr.draw_text(f"DIST TO GOAL : {dist_beacon:4.1f} m", 30, hud_y + 92, 18, pr.LIME)

            # Actuators
            pr.draw_line(340, hud_y + 12, 340, hud_y + 110, pr.Color(45, 65, 100, 255))
            pr.draw_text("AI 3D MOTORS", 365, hud_y + 16, 22, pr.Color(200, 225, 255, 255))

            # Steering
            steer_val = float(actions[0, 0])
            pr.draw_rectangle(365, hud_y + 48, 140, 16, pr.Color(32, 45, 75, 255))
            s_bar_w = int(steer_val * 68)
            s_color = pr.RED if steer_val < 0 else pr.Color(0, 220, 255, 255)
            pr.draw_rectangle(435 if steer_val > 0 else 435 + s_bar_w, hud_y + 48, abs(s_bar_w), 16, s_color)
            pr.draw_text(f"STEER [{steer_val:+.2f}]", 515, hud_y + 48, 16, pr.WHITE)

            # Throttle
            throttle_val = float(actions[0, 1])
            pr.draw_rectangle(365, hud_y + 70, 140, 16, pr.Color(32, 45, 75, 255))
            pr.draw_rectangle(365, hud_y + 70, int(max(0.0, throttle_val) * 140), 16, pr.LIME)
            pr.draw_text(f"ROLL  [{throttle_val:.2f}]", 515, hud_y + 70, 16, pr.WHITE)

            # Thruster
            is_firing = float(actions[0, 2]) > 0.0
            j_color = pr.Color(230, 140, 255, 255) if is_firing else pr.GRAY
            j_text = "JUMP THRUSTER: [ACTIVE LAUNCH]" if is_firing else "JUMP THRUSTER: [GROUND]"
            pr.draw_text(j_text, 365, hud_y + 94, 17, j_color)

            # Stage Milestone Badges
            pr.draw_line(720, hud_y + 12, 720, hud_y + 110, pr.Color(45, 65, 100, 255))
            pr.draw_text("STAGE CLEARANCES", 745, hud_y + 16, 20, pr.Color(220, 140, 255, 255))

            h_col = pr.LIME if c_pos[2] > 7.5 else pr.YELLOW
            pr.draw_text("1. HURDLE JUMP  : [CLEARED]" if c_pos[2] > 7.5 else "1. HURDLE JUMP  : [PENDING]", 745, hud_y + 44, 15, h_col)

            c_col = pr.LIME if c_pos[2] > 22.5 else pr.YELLOW
            pr.draw_text("2. CHASM 4.5m   : [CLEARED]" if c_pos[2] > 22.5 else "2. CHASM 4.5m   : [PENDING]", 745, hud_y + 66, 15, c_col)

            g_col = pr.LIME if c_pos[2] >= 32.5 else pr.SKYBLUE
            pr.draw_text("3. SUMMIT BEACON: [VICTORY]" if c_pos[2] >= 32.5 else "3. SUMMIT BEACON: [RACING]", 745, hud_y + 88, 15, g_col)

            pr.end_drawing()

            # -------------------------------------------------------------
            # VIDEO FRAME CAPTURE (Correct Upright Orientation)
            # -------------------------------------------------------------
            if video_writer is not None:
                img = pr.load_image_from_screen()
                buf = pr.ffi.buffer(img.data, img.width * img.height * 4)
                raw_frame = np.frombuffer(buf, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3]
                # Raylib returns frame buffer where row 0 is top; np.ascontiguousarray ensures proper raster write
                frame = np.ascontiguousarray(raw_frame)
                video_writer.append_data(frame)
                pr.unload_image(img)

            frame_count += 1

        if video_writer is not None:
            video_writer.close()
            print(f"[+] 3D MP4 Video successfully saved to: {video_path} ({frame_count} frames)")

        pr.close_window()


# =====================================================================
# 5. ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CyberMarble 3D: Autonomous Stunt AI Course")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberMarble 3D: Autonomous Jumping & Stunt Obstacle Course    ")
    print("=================================================================")

    track = Track3D()
    print("1. Constructing 3D Multi-Stage Obstacle Track...")
    print(f"   Launch Pad    : Z = 0.0m -> 11.0m (Ground level)")
    print(f"   Laser Hurdle  : Z = 7.0m (Requires timing vertical jump!)")
    print(f"   Ascending Ramp: Z = 11.0m -> 18.0m (Climbs Y: 0.5m -> 2.8m)")
    print(f"   The Void Chasm: Z = 18.0m -> 22.5m (4.5m aerial leap across abyss!)")
    print(f"   Summit Goal   : Z = 33.0m (Beacon on Elevated Sky-Deck)")

    print(f"\n2. Evolving 512 Agents via Antithetic ES ({args.generations} Generations)...")
    engine = MarblePhysics3D(track, num_agents=512)
    policy = NeuroMarblePolicy(in_dim=10, out_dim=3)

    elapsed, sps, top_fit = policy.train_epoch(engine, generations=args.generations, rollout_steps=320, verbose=True)
    print(f"\n   Training Finished in {elapsed:.2f}s! ({sps:,.0f} agent-steps/sec)")
    print(f"   Champion Fitness: {top_fit:.1f}")

    print("\n3. Launching Studio-Grade Raylib 3D Engine...")
    print("   Controls: [SPACE] Pause | [R] Reset | [G] Toggle Ghost Swarm\n")

    max_frames = args.frames if args.video else None
    viz = CyberMarbleVisualizer(track, policy)
    viz.run(video_path=args.video, max_frames=max_frames)


if __name__ == "__main__":
    main()

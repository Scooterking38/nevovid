# demo.py
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import Tuple, List, Optional
import numpy as np

# Require raylib
try:
    import pyray as pr
except ImportError:
    print("\n[!] 'raylib' is not installed.")
    print("    Please install it using: pip install raylib\n")
    sys.exit(1)


# =====================================================================
# 1. CONTINUOUS 3D STUNT COURSE GEOMETRY
# =====================================================================
class CyberCourse3D:
    def __init__(self):
        # Track width is 4.0m (centered at X = 0, bounds: X in [-2.0, +2.0])
        self.width = 4.0
        self.start_pos = np.array([0.0, 0.9, 1.0], dtype=np.float32)
        self.goal_pos = np.array([0.0, 3.4, 32.0], dtype=np.float32)

        # Obstacle Locations
        self.laser_z = 6.0       # Laser hurdle at Z = 6.0m
        self.ramp_start_z = 10.0 # Ramp climbs Z = 10.0 -> 18.0m (Y: 0.5 -> 3.0m)
        self.chasm_start_z = 18.0# Takeoff kicker at Z = 18.0m
        self.chasm_end_z = 22.5  # Receiving deck across 4.5m chasm!
        self.track_end_z = 35.0  # Summit Goal Deck

    def get_track_surface(self, x: float, z: float) -> Tuple[float, float, bool]:
        """
        Returns (surface_y, slope_angle_rad, is_in_chasm).
        """
        # Off-track check (fell off the left/right sides)
        if abs(x) > (self.width / 2.0 + 0.3):
            return -999.0, 0.0, True

        # Zone 1: Ground Runway (Z: 0.0 -> 10.0m)
        if 0.0 <= z < self.ramp_start_z:
            return 0.5, 0.0, False

        # Zone 2: Continuous Ascending Launch Ramp (Z: 10.0 -> 18.0m)
        if self.ramp_start_z <= z < self.chasm_start_z:
            t = (z - self.ramp_start_z) / (self.chasm_start_z - self.ramp_start_z)
            y = 0.5 + t * 2.5
            slope = math.atan2(2.5, 8.0)
            # Add an upward kicker curve right at the takeoff lip (Z: 17.0 -> 18.0)
            if z > 17.0:
                y += (z - 17.0) * 0.25
                slope += 0.15
            return y, slope, False

        # Zone 3: The 3D Chasm Gap (Z: 18.0 -> 22.5m, 4.5m of empty air!)
        if self.chasm_start_z <= z < self.chasm_end_z:
            return -999.0, 0.0, True  # Bottomless abyss

        # Zone 4: Receiving Elevated Sky-Deck & Summit (Z: 22.5 -> 35.0m)
        if self.chasm_end_z <= z <= self.track_end_z:
            return 3.0, 0.0, False

        return -999.0, 0.0, True


# =====================================================================
# 2. VECTORIZED 3D MARBLE PHYSICS & JUMP SIMULATOR
# =====================================================================
class Fast3DMarbleEngine:
    def __init__(self, course: CyberCourse3D, num_envs: int = 512):
        self.course = course
        self.num_envs = num_envs
        self.radius = 0.38

        self.pos = np.zeros((num_envs, 3), dtype=np.float32)
        self.vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.grounded = np.zeros(num_envs, dtype=bool)
        self.crashed = np.zeros(num_envs, dtype=bool)
        self.steps = np.zeros(num_envs, dtype=np.int32)

        # Track milestone clearances
        self.cleared_hurdle = np.zeros(num_envs, dtype=bool)
        self.cleared_chasm = np.zeros(num_envs, dtype=bool)
        self.reached_goal = np.zeros(num_envs, dtype=bool)

        self.reset_all()

    def reset_all(self, indices: Optional[np.ndarray] = None):
        if indices is None:
            indices = np.arange(self.num_envs, dtype=np.int32)
        self.pos[indices] = self.course.start_pos
        self.vel[indices] = 0.0
        self.grounded[indices] = True
        self.crashed[indices] = False
        self.steps[indices] = 0
        self.cleared_hurdle[indices] = False
        self.cleared_chasm[indices] = False
        self.reached_goal[indices] = False

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        actions: [N, 3] -> act[0]=steer_x, act[1]=throttle_z, act[2]=jump_thruster (>0.0)
        """
        dt = 1.0 / 60.0
        gravity = 14.5  # m/s^2

        steer_x = np.clip(actions[:, 0], -1.0, 1.0)
        throttle_z = np.clip(0.45 + 0.55 * actions[:, 1], 0.45, 1.0)  # Always committed forward drive
        wants_jump = actions[:, 2] > 0.0

        # Lateral and Longitudinal Acceleration
        accel_lat = np.where(self.grounded, 28.0, 6.0)
        self.vel[:, 0] += steer_x * accel_lat * dt
        self.vel[:, 2] += throttle_z * 24.0 * dt

        # Forward rolling drag
        self.vel[:, 0] *= 0.965
        self.vel[:, 2] *= 0.990

        # Jump Launch: Vertical Impulse (v_y = 6.8 m/s -> launches 1.6m high, flies 7.8m horizontally)
        can_jump = self.grounded & wants_jump
        self.vel[:, 1] = np.where(can_jump, 6.8, self.vel[:, 1] - gravity * dt)

        # Candidate position
        cand_x = self.pos[:, 0] + self.vel[:, 0] * dt
        cand_y = self.pos[:, 1] + self.vel[:, 1] * dt
        cand_z = self.pos[:, 2] + self.vel[:, 2] * dt

        # Terrain & Slope Follow
        self.grounded[:] = False
        for i in range(self.num_envs):
            surf_y, slope, in_chasm = self.course.get_track_surface(float(cand_x[i]), float(cand_z[i]))

            if not in_chasm:
                target_floor_y = surf_y + self.radius
                if cand_y[i] <= target_floor_y:
                    cand_y[i] = target_floor_y
                    self.vel[i, 1] = max(0.0, self.vel[i, 1])
                    self.grounded[i] = True

            # Falling into Abyss
            if in_chasm and cand_y[i] < -1.5:
                self.crashed[i] = True

        # -------------------------------------------------------------
        # OBSTACLE 1: LASER HURDLE (Z = 6.0m, Height Y: 0.5 -> 1.2m)
        # -------------------------------------------------------------
        near_laser = np.abs(cand_z - self.course.laser_z) < 0.45
        at_laser_height = cand_y < 1.35
        hit_laser = near_laser & at_laser_height & (~self.crashed)

        # Hitting laser repels and penalizes
        self.vel[hit_laser, 2] = -4.0

        # Successfully jumping over the laser hurdle
        cleared_hurdle_now = near_laser & (cand_y >= 1.35) & (~self.cleared_hurdle)
        self.cleared_hurdle |= cleared_hurdle_now

        # -------------------------------------------------------------
        # OBSTACLE 2: 4.5m CHASM LEAP (Z: 18.0 -> 22.5m)
        # -------------------------------------------------------------
        cleared_chasm_now = (cand_z >= self.course.chasm_end_z) & (cand_y >= 2.8) & (~self.cleared_chasm)
        self.cleared_chasm |= cleared_chasm_now

        # -------------------------------------------------------------
        # GOAL SUMMIT BEACON (Z >= 32.0m)
        # -------------------------------------------------------------
        to_goal = self.course.goal_pos[None, :] - np.column_stack([cand_x, cand_y, cand_z])
        dist_goal = np.linalg.norm(to_goal, axis=-1)
        reached_now = (dist_goal < 2.0) & (~self.reached_goal)
        self.reached_goal |= reached_now

        self.pos[:, 0] = cand_x
        self.pos[:, 1] = cand_y
        self.pos[:, 2] = cand_z
        self.steps += 1

        # -------------------------------------------------------------
        # REWARD SHAPING: FORCES THE BOT TO JUMP & USE ENVIRONMENT
        # -------------------------------------------------------------
        fwd_reward = self.vel[:, 2] * 12.0
        center_penalty = -np.abs(self.pos[:, 0]) * 3.0  # Penalty for wandering off centerline

        # Big Stage-Clearance Bounties:
        laser_bounty = np.where(cleared_hurdle_now, 2000.0, 0.0)
        laser_tax = np.where(hit_laser, 120.0, 0.0)

        chasm_bounty = np.where(cleared_chasm_now, 4000.0, 0.0)
        goal_bounty = np.where(reached_now, 10000.0, 0.0)
        crash_tax = np.where(self.crashed, 150.0, 0.0)

        rewards = fwd_reward + center_penalty + laser_bounty - laser_tax + chasm_bounty + goal_bounty - crash_tax
        dones = self.reached_goal | self.crashed | (self.steps >= 340)

        obs = self.get_observations()
        return obs, rewards, dones

    def get_observations(self) -> np.ndarray:
        """
        12-Dim State Observation:
        - [0]: Lane offset X / 2.0 (-1=left edge, +1=right edge)
        - [1]: Elevation Y / 4.0
        - [2]: Progress along track Z / 35.0
        - [3..5]: Velocities (v_x, v_y, v_z)
        - [6]: Grounded flag
        - [7]: Distance to Laser Hurdle (activates within 3m of laser!)
        - [8]: Distance to Chasm Takeoff (activates within 3m of ramp edge!)
        - [9..11]: 3D Direction to Goal Beacon
        """
        to_goal = self.course.goal_pos[None, :] - self.pos
        dist_goal = np.linalg.norm(to_goal, axis=-1, keepdims=True)
        dir_goal = to_goal / (dist_goal + 1e-6)

        # Proximity Sensors: trigger directly before obstacles
        dist_to_laser = np.clip((self.course.laser_z - self.pos[:, 2]) / 3.0, -1.0, 1.0)
        dist_to_chasm = np.clip((self.course.chasm_start_z - self.pos[:, 2]) / 3.0, -1.0, 1.0)

        obs = np.column_stack([
            self.pos[:, 0] / 2.0,                     # 0: Lane offset X
            self.pos[:, 1] / 4.0,                     # 1: Elevation Y
            self.pos[:, 2] / 35.0,                    # 2: Track Progress Z
            self.vel[:, 0] * 0.1,                     # 3: Vel X
            self.vel[:, 1] * 0.1,                     # 4: Vel Y (Vertical)
            self.vel[:, 2] * 0.1,                     # 5: Vel Z (Forward Speed)
            self.grounded.astype(np.float32),         # 6: Grounded Flag
            dist_to_laser,                            # 7: Proximity to Laser Hurdle
            dist_to_chasm,                            # 8: Proximity to Chasm Takeoff
            dir_goal[:, 0], dir_goal[:, 1], dir_goal[:, 2]  # 9..11: 3D Goal Unit Vector
        ]).astype(np.float32)

        return obs


# =====================================================================
# 3. ANTITHETIC EVOLUTION STRATEGY (OpenAI-ES with Jump Prior)
# =====================================================================
class Fast3DMarblePolicy:
    def __init__(self, in_dim=12, out_dim=3):
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

        # --- SEED INDUCTIVE JUMP & STEERING PRIOR ---
        # 1. Stay centered on track: lane offset (idx 0) steers opposite to center
        self.mW1[0, 0] = -2.4    # Offset left (+X) -> steer left (-X)
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.5

        # 2. Full forward speed commitment
        self.mW1[5, 1] = 1.2     # Vel Z
        self.mb1[1] = 0.8
        self.mW2[1, 1] = 1.5
        self.mW3[1, 1] = 1.4
        self.mb3[1] = 1.2        # Maximum forward roll drive

        # 3. Dedicated Jump-Timing Trigger Neurons
        # Trigger jump when approaching Laser Hurdle (sensor 7: 0.1 < dist < 0.6)
        self.mW1[7, 2] = -2.8    # dist_to_laser approaching 0 -> JUMP!
        # Trigger jump when approaching Chasm Takeoff (sensor 8: 0.1 < dist < 0.6)
        self.mW1[8, 2] = -3.2    # dist_to_chasm approaching 0 -> LAUNCH CHASM!
        self.mW1[6, 2] = 1.4     # Grounded flag
        self.mb1[2] = 0.2
        self.mW2[2, 2] = 2.0
        self.mW3[2, 2] = 1.8
        self.mb3[2] = 0.1        # Active jump exploration bias

        self.pop_size = 512
        self.W1 = np.zeros((self.pop_size, in_dim, 32), dtype=np.float32)
        self.b1 = np.zeros((self.pop_size, 32), dtype=np.float32)
        self.W2 = np.zeros((self.pop_size, 32, 16), dtype=np.float32)
        self.b2 = np.zeros((self.pop_size, 16), dtype=np.float32)
        self.W3 = np.zeros((self.pop_size, 16, out_dim), dtype=np.float32)
        self.b3 = np.zeros((self.pop_size, out_dim), dtype=np.float32)

        self.sigma = 0.08
        self._sample_antithetic()

    def _sample_antithetic(self):
        half = self.pop_size // 2
        self.eW1 = np.random.randn(half, self.in_dim, 32).astype(np.float32)
        self.eb1 = np.random.randn(half, 32).astype(np.float32)
        self.eW2 = np.random.randn(half, 32, 16).astype(np.float32)
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

        # Index 0 is clean Master Policy
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
        raw_out = np.tanh(np.matmul(h2[:, None, :], w3).squeeze(1) + b3)

        steer = raw_out[:, 0:1]
        throttle = np.clip(0.50 + 0.50 * raw_out[:, 1:2], 0.40, 1.0)
        jump = raw_out[:, 2:3]

        return np.hstack([steer, throttle, jump]).astype(np.float32)

    def evolve(self, fitness: np.ndarray):
        self.generation += 1
        half = self.pop_size // 2

        fit_norm = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-6)
        diff = (fit_norm[:half] - fit_norm[half:])[:, None, None]
        diff_b = (fit_norm[:half] - fit_norm[half:])[:, None]

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

    def train_epoch(self, engine: Fast3DMarbleEngine, generations=60, rollout_steps=320, verbose=True):
        t0 = time.perf_counter()
        top_fit = -9999.0

        for gen in range(generations):
            engine.reset_all()
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            alive = np.ones(self.pop_size, dtype=bool)
            steps_to_goal = np.full(self.pop_size, rollout_steps, dtype=np.int32)
            max_z = np.zeros(self.pop_size, dtype=np.float32)

            obs = engine.get_observations()

            for step_idx in range(rollout_steps):
                act = self.forward(obs)
                obs, rewards, dones = engine.step(act)

                cur_z = engine.pos[:, 2]
                max_z = np.maximum(max_z, cur_z)

                newly_solved = engine.reached_goal & alive & (steps_to_goal == rollout_steps)
                steps_to_goal = np.where(newly_solved, step_idx + 1, steps_to_goal)

                fitness += np.where(alive, rewards, 0.0)
                alive &= ~dones
                if not np.any(alive):
                    break

            # Progress bonus along Z
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
# 4. HARDWARE RAYLIB 3D VISUALIZER & VIDEO RECORDER
# =====================================================================
class CyberMarbleVisualizer:
    def __init__(self, course: CyberCourse3D, ga: Fast3DMarblePolicy):
        self.course = course
        self.ga = ga
        self.swarm_size = 32
        self.engine = Fast3DMarbleEngine(course, num_envs=self.swarm_size)
        self.engine.reset_all()

        self.screen_w = 1600
        self.screen_h = 920
        self.hud_h = 120
        self.viewport_h = self.screen_h - self.hud_h
        self.view_3d_w = self.screen_w - 280

        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_HIGHDPI | pr.FLAG_VSYNC_HINT)
        pr.init_window(self.screen_w, self.screen_h, "CyberMarble 3D: Autonomous Jumping AI Course")
        pr.set_target_fps(60)

        # Camera modes: 0 = 3D Chase Cam, 1 = 3D Side Stunt Angle, 2 = 3D Aerial Orbit
        self.cam_mode = 0
        self.cam_smoothed = [course.start_pos[0] - 4.5, 3.2, course.start_pos[2] - 5.0]

        self.camera = pr.Camera3D(
            pr.Vector3(self.cam_smoothed[0], self.cam_smoothed[1], self.cam_smoothed[2]),
            pr.Vector3(course.start_pos[0], course.start_pos[1], course.start_pos[2]),
            pr.Vector3(0.0, 1.0, 0.0),
            52.0,
            pr.CAMERA_PERSPECTIVE,
        )

        self.ribbons: List[List[float]] = []
        self.ball_roll = 0.0
        self.show_swarm = True
        self.paused = False

    def draw_course_3d(self):
        # 1. Ground Runway (Z: 0 -> 10m)
        pr.draw_cube(pr.Vector3(0.0, 0.25, 5.0), 4.0, 0.5, 10.0, pr.Color(22, 32, 54, 255))
        pr.draw_cube_wires(pr.Vector3(0.0, 0.25, 5.0), 4.0, 0.5, 10.0, pr.Color(0, 220, 255, 255))

        # Runway Centerline Grid
        for z in range(0, 11, 2):
            pr.draw_line_3d(pr.Vector3(-1.8, 0.52, float(z)), pr.Vector3(1.8, 0.52, float(z)), pr.Color(0, 180, 240, 180))

        # 2. Obstacle 1: High-Voltage Laser Barrier (Z = 6.0m)
        hurdle_center = pr.Vector3(0.0, 0.85, 6.0)
        pr.draw_cylinder_wires(pr.Vector3(-2.1, 0.25, 6.0), 0.2, 0.2, 1.6, 8, pr.RED)
        pr.draw_cylinder_wires(pr.Vector3(2.1, 0.25, 6.0), 0.2, 0.2, 1.6, 8, pr.RED)
        # Laser beam
        pulse = abs(math.sin(time.time() * 8.0)) * 0.1
        pr.draw_cube(hurdle_center, 4.2, 0.4 + pulse, 0.15, pr.Color(255, 30, 80, 220))
        pr.draw_cube_wires(hurdle_center, 4.2, 0.4 + pulse, 0.15, pr.Color(255, 150, 180, 255))

        # 3. Continuous Ascending Launch Ramp (Z: 10 -> 18m)
        ramp_center = pr.Vector3(0.0, 1.75, 14.0)
        pr.draw_cube(ramp_center, 4.0, 0.4, 8.0, pr.Color(28, 44, 75, 255))
        pr.draw_cube_wires(ramp_center, 4.0, 0.4, 8.0, pr.Color(0, 240, 255, 255))

        # Ramp Climbing Chevrons (>>>)
        for z in range(11, 18, 2):
            t = (z - 10.0) / 8.0
            y = 0.5 + t * 2.5 + 0.22
            pr.draw_line_3d(pr.Vector3(-1.5, y, float(z)), pr.Vector3(0.0, y, float(z) + 0.5), pr.Color(0, 255, 200, 255))
            pr.draw_line_3d(pr.Vector3(1.5, y, float(z)), pr.Vector3(0.0, y, float(z) + 0.5), pr.Color(0, 255, 200, 255))

        # Takeoff Kicker Lip (Z = 17.5 -> 18.0)
        kicker_pos = pr.Vector3(0.0, 3.1, 17.75)
        pr.draw_cube(kicker_pos, 4.0, 0.35, 0.5, pr.Color(240, 80, 40, 255))
        pr.draw_cube_wires(kicker_pos, 4.0, 0.35, 0.5, pr.YELLOW)

        # 4. The Chasm Gap (Z: 18.0 -> 22.5m)
        # Deep abyss hazard grid below
        pr.draw_plane(pr.Vector3(0.0, -1.0, 20.25), pr.Vector2(10.0, 4.5), pr.Color(6, 10, 20, 255))
        pr.draw_line_3d(pr.Vector3(-2.2, 3.0, 18.0), pr.Vector3(-2.2, -1.0, 18.0), pr.RED)
        pr.draw_line_3d(pr.Vector3(2.2, 3.0, 18.0), pr.Vector3(2.2, -1.0, 18.0), pr.RED)

        # 5. Elevated Receiving Sky-Deck & Summit (Z: 22.5 -> 35.0m, Y = 3.0m)
        deck_center = pr.Vector3(0.0, 2.75, 28.75)
        pr.draw_cube(deck_center, 4.5, 0.5, 12.5, pr.Color(20, 35, 65, 255))
        pr.draw_cube_wires(deck_center, 4.5, 0.5, 12.5, pr.Color(0, 255, 180, 255))

        # 6. 3D Emerald Goal Beacon at Summit
        beacon_base = pr.Vector3(0.0, 3.0, 32.0)
        pr.draw_cylinder(beacon_base, 0.5, 0.5, 5.5, 16, pr.Color(0, 255, 128, 190))
        pr.draw_sphere(pr.Vector3(0.0, 8.5, 32.0), 0.7, pr.Color(100, 255, 180, 255))
        pr.draw_sphere_wires(pr.Vector3(0.0, 8.5, 32.0), 0.75, 8, 8, pr.WHITE)

        # Expanding Energy Rings on Goal Deck
        ring_r = 0.5 + float(math.fmod(time.time() * 2.5, 2.5))
        pr.draw_circle_3d(beacon_base, ring_r, pr.Vector3(0, 1, 0), 0.0, pr.Color(0, 255, 128, 220))

    def draw_marble_3d(self, pos: np.ndarray, vel: np.ndarray, grounded: bool):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        fwd_speed = float(vel[2])

        if abs(fwd_speed) > 0.05:
            self.ball_roll += (fwd_speed / self.engine.radius) * (1.0 / 60.0)

        center = pr.Vector3(x, y, z)

        # 1. High-Vis Cyan Cyberpunk Sphere
        ball_c = pr.Color(0, 245, 255, 255) if grounded else pr.Color(230, 90, 255, 255)
        pr.draw_sphere(center, self.engine.radius, ball_c)
        pr.draw_sphere_wires(center, self.engine.radius, 10, 10, pr.WHITE)

        # 2. Dual Rolling Gyro-Rings (rotate as the ball rolls!)
        pr.draw_circle_3d(center, self.engine.radius + 0.05, pr.Vector3(1, 0, 0), math.degrees(self.ball_roll), pr.Color(255, 220, 50, 240))
        pr.draw_circle_3d(center, self.engine.radius + 0.05, pr.Vector3(0, 0, 1), 0.0, pr.Color(255, 60, 140, 200))

        # 3. Rocket Thruster Plasma Flame when Airborne
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

            # Inputs
            if pr.is_key_pressed(pr.KEY_SPACE):
                self.paused = not self.paused
            elif pr.is_key_pressed(pr.KEY_C):
                self.cam_mode = (self.cam_mode + 1) % 3
                modes = ["3D Chase Cam", "3D Stunt Side Angle", "3D Aerial Overview"]
                print(f"[*] Camera switched to: {modes[self.cam_mode]}")
            elif pr.is_key_pressed(pr.KEY_G):
                self.show_swarm = not self.show_swarm
            elif pr.is_key_pressed(pr.KEY_R):
                self.engine.reset_all()

            # Physics Simulation Step
            if not self.paused:
                actions = self.ga.forward(obs)
                obs, rewards, dones = self.engine.step(actions)
                if dones[0]:
                    self.engine.reset_all(np.array([0], dtype=np.int32))

            c_pos = self.engine.pos[0]
            c_vel = self.engine.vel[0]
            c_grounded = bool(self.engine.grounded[0])
            speed = float(math.sqrt(float(c_vel[0]**2 + c_vel[2]**2)))

            self.ribbons.append([c_pos[0], c_pos[1] - 0.15, c_pos[2], 1.0, 1.0])

            # 3D Camera Controls
            if self.cam_mode == 0:  # 3D Chase Camera
                target_x = c_pos[0] - 0.2
                target_y = max(2.0, c_pos[1] + 2.2)
                target_z = c_pos[2] - 5.5

                self.cam_smoothed[0] += (target_x - self.cam_smoothed[0]) * 0.25
                self.cam_smoothed[1] += (target_y - self.cam_smoothed[1]) * 0.25
                self.cam_smoothed[2] += (target_z - self.cam_smoothed[2]) * 0.25

                self.camera.position = pr.Vector3(float(self.cam_smoothed[0]), float(self.cam_smoothed[1]), float(self.cam_smoothed[2]))
                self.camera.target = pr.Vector3(float(c_pos[0]), float(c_pos[1] + 0.4), float(c_pos[2] + 2.5))
                self.camera.up = pr.Vector3(0.0, 1.0, 0.0)
                self.camera.fovy = 54.0

            elif self.cam_mode == 1:  # 3D Stunt Side Camera
                self.camera.position = pr.Vector3(float(c_pos[0] + 6.5), float(c_pos[1] + 3.2), float(c_pos[2] - 1.0))
                self.camera.target = pr.Vector3(float(c_pos[0]), float(c_pos[1] + 0.2), float(c_pos[2] + 1.5))
                self.camera.up = pr.Vector3(0.0, 1.0, 0.0)
                self.camera.fovy = 52.0

            elif self.cam_mode == 2:  # 3D Aerial Overview
                self.camera.position = pr.Vector3(0.0, 24.0, 16.0)
                self.camera.target = pr.Vector3(0.0, 2.0, 18.0)
                self.camera.up = pr.Vector3(0.0, 0.0, 1.0)
                self.camera.fovy = 60.0

            # -------------------------------------------------------------
            # RENDER 3D SCENE
            # -------------------------------------------------------------
            pr.begin_drawing()
            pr.clear_background(pr.Color(12, 16, 28, 255))

            pr.begin_mode_3d(self.camera)

            # Draw 3D Continuous Stunt Course
            self.draw_course_3d()

            # Draw 3D Trajectory Ribbon Trail
            self.draw_ribbons()

            # Ghost Swarm
            if self.show_swarm:
                for i in range(1, self.swarm_size):
                    sp = self.engine.pos[i]
                    pr.draw_sphere(pr.Vector3(float(sp[0]), float(sp[1]), float(sp[2])), 0.22, pr.Color(240, 60, 160, 180))

            # Champion AI Cyber-Marble
            self.draw_marble_3d(c_pos, c_vel, c_grounded)

            pr.end_mode_3d()

            # -------------------------------------------------------------
            # CRISP 2D OVERLAYS: HUD & TELEMETRY
            # -------------------------------------------------------------
            # 1. Tactical Radar PiP in Top-Left
            pr.draw_rectangle(25, 25, 120, 200, pr.Color(16, 24, 44, 235))
            pr.draw_rectangle_lines(25, 25, 120, 200, pr.Color(0, 220, 255, 255))
            scale_z = 200.0 / 35.0

            # Track outline in PiP
            pr.draw_rectangle(75, int(25 + 0 * scale_z), 20, int(10 * scale_z), pr.Color(45, 65, 105, 255))
            pr.draw_line(65, int(25 + 6.0 * scale_z), 105, int(25 + 6.0 * scale_z), pr.RED)  # Laser
            pr.draw_rectangle(75, int(25 + 10 * scale_z), 20, int(8 * scale_z), pr.Color(45, 65, 105, 255))
            pr.draw_rectangle(75, int(25 + 22.5 * scale_z), 20, int(12.5 * scale_z), pr.Color(45, 65, 105, 255))
            pr.draw_circle(85, int(25 + 32.0 * scale_z), 5, pr.GREEN)  # Goal

            # Ball dot in PiP
            ball_pip_y = int(25 + c_pos[2] * scale_z)
            ball_pip_x = int(85 + (c_pos[0] / 2.0) * 15.0)
            pr.draw_circle(ball_pip_x, ball_pip_y, 4, pr.Color(0, 245, 255, 255))
            pr.draw_text("TRACK RADAR", 32, 30, 12, pr.Color(0, 220, 255, 255))

            # 2. Right-Hand Neural Monitor
            panel_x = self.view_3d_w
            panel_w = self.screen_w - panel_x
            pr.draw_rectangle(panel_x, 0, panel_w, self.viewport_h, pr.Color(20, 28, 48, 255))
            pr.draw_line(panel_x, 0, panel_x, self.viewport_h, pr.Color(0, 220, 255, 255))

            pr.draw_text("NEURAL MONITOR", panel_x + 20, 20, 22, pr.Color(0, 220, 255, 255))
            pr.draw_text("AI GYRO SENSORS", panel_x + 20, 60, 16, pr.Color(190, 210, 240, 255))

            sensor_labels = ["LANE OFFSET", "ELEVATION", "TRACK DIST", "SPEED", "LASER PROX", "CHASM PROX"]
            sensor_vals = [float(obs[0, 0]), float(obs[0, 1]), float(obs[0, 2]), float(obs[0, 5]), float(obs[0, 7]), float(obs[0, 8])]
            for i, (lbl, val) in enumerate(zip(sensor_labels, sensor_vals)):
                by = 90 + i * 26
                pr.draw_text(lbl, panel_x + 20, by, 13, pr.LIGHTGRAY)
                pr.draw_rectangle(panel_x + 130, by, 110, 14, pr.Color(32, 45, 75, 255))
                bar_w = int(np.clip(abs(val), 0.0, 1.0) * 110)
                bar_c = pr.Color(0, 220, 255, 255) if val >= 0 else pr.Color(255, 60, 90, 255)
                pr.draw_rectangle(panel_x + 130, by, bar_w, 14, bar_c)

            # Milestones Tracker
            pr.draw_text("OBSTACLE STATUS", panel_x + 20, 280, 16, pr.Color(190, 210, 240, 255))
            h_stat = "[HURDLE: CLEARED]" if c_pos[2] > 6.5 else "[HURDLE: PENDING]"
            h_col = pr.GREEN if c_pos[2] > 6.5 else pr.YELLOW
            pr.draw_text(h_stat, panel_x + 20, 310, 14, h_col)

            c_stat = "[CHASM: CLEARED]" if c_pos[2] > 22.5 else "[CHASM: PENDING]"
            c_col = pr.GREEN if c_pos[2] > 22.5 else pr.YELLOW
            pr.draw_text(c_stat, panel_x + 20, 335, 14, c_col)

            g_stat = "[SUMMIT GOAL: WON]" if c_pos[2] >= 32.0 else "[SUMMIT: RACING]"
            g_col = pr.GREEN if c_pos[2] >= 32.0 else pr.SKYBLUE
            pr.draw_text(g_stat, panel_x + 20, 360, 14, g_col)

            # 3. Bottom Control HUD
            hud_y = self.viewport_h
            pr.draw_rectangle(0, hud_y, self.screen_w, self.hud_h, pr.Color(14, 20, 36, 255))
            pr.draw_line(0, hud_y, self.screen_w, hud_y, pr.Color(0, 220, 255, 255))

            dist_beacon = float(np.linalg.norm(c_pos - self.course.goal_pos))
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

            # Camera Guide
            pr.draw_line(720, hud_y + 12, 720, hud_y + 110, pr.Color(45, 65, 100, 255))
            c_modes = ["3D CHASE CAM", "3D STUNT SIDE ANGLE", "3D AERIAL OVERVIEW"]
            pr.draw_text(f"CAMERA: [{c_modes[self.cam_mode]}] (Press 'C')", 745, hud_y + 16, 20, pr.Color(220, 140, 255, 255))
            keys = [
                "[C]     Toggle Camera (Chase / Side Stunt / Aerial)",
                "[SPACE] Pause / Play Simulation",
                "[G]     Toggle Ghost Swarm",
                "[R]     Reset to Launchpad",
            ]
            for i, k in enumerate(keys):
                pr.draw_text(k, 745, hud_y + 44 + i * 18, 14, pr.Color(190, 210, 235, 255))

            pr.end_drawing()

            # -------------------------------------------------------------
            # RECORD VIDEO (DO NOT FLIP VERTICAL - STAYS RIGHT-SIDE UP!)
            # -------------------------------------------------------------
            if video_writer is not None:
                img = pr.load_image_from_screen()
                buf = pr.ffi.buffer(img.data, img.width * img.height * 4)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3]
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
    parser = argparse.ArgumentParser(description="CyberMarble 3D: Autonomous Jumping AI Course")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberMarble 3D: Autonomous Jumping & Stunt Obstacle Course    ")
    print("=================================================================")

    course = CyberCourse3D()
    print("1. Constructing 3D Multi-Stage Obstacle Course...")
    print(f"   Launch Pad    : Z = 0.0m -> 10.0m (Ground level)")
    print(f"   Laser Hurdle  : Z = 6.0m (Requires timing vertical jump!)")
    print(f"   Ascending Ramp: Z = 10.0m -> 18.0m (Smooth 17.4 deg incline)")
    print(f"   The Void Chasm: Z = 18.0m -> 22.5m (4.5m aerial leap across abyss!)")
    print(f"   Summit Goal   : Z = 32.0m (Beacon on Elevated Sky-Deck)")

    print(f"\n2. Evolving 512 Agents via Antithetic ES ({args.generations} Generations)...")
    engine = Fast3DMarbleEngine(course, num_envs=512)
    policy = Fast3DMarblePolicy(in_dim=12, out_dim=3)

    elapsed, sps, top_fit = policy.train_epoch(engine, generations=args.generations, rollout_steps=320, verbose=True)
    print(f"\n   Training Finished in {elapsed:.2f}s! ({sps:,.0f} agent-steps/sec)")
    print(f"   Champion Fitness: {top_fit:.1f}")

    print("\n3. Launching Studio-Grade Raylib 3D Engine...")
    print("   Controls: [C] Toggle View (Chase / Side Stunt / Aerial) | [SPACE] Pause | [R] Reset\n")

    max_frames = args.frames if args.video else None
    viz = CyberMarbleVisualizer(course, policy)
    viz.run(video_path=args.video, max_frames=max_frames)


if __name__ == "__main__":
    main()

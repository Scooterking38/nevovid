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
# 1. 3D CYBER-ARENA GEOMETRY DEFINITION
# =====================================================================
class PlatformBox:
    def __init__(self, min_pt: List[float], max_pt: List[float], color: pr.Color, border_color: pr.Color):
        self.min_pt = np.array(min_pt, dtype=np.float32)
        self.max_pt = np.array(max_pt, dtype=np.float32)
        self.center = (self.min_pt + self.max_pt) * 0.5
        self.size = self.max_pt - self.min_pt
        self.color = color
        self.border_color = border_color


class CyberMarbleCourse:
    def __init__(self):
        self.start_pos = np.array([3.0, 1.2, 3.0], dtype=np.float32)
        self.goal_pos = np.array([28.0, 4.4, 21.0], dtype=np.float32)

        # 3D Solid Platforms [min_x, min_y, min_z, max_x, max_y, max_z]
        # (Y is UP in standard 3D OpenGL / Raylib)
        self.platforms: List[PlatformBox] = [
            # 1. Tier 0 Ground Launchpad
            PlatformBox([0.0, 0.0, 0.0], [10.0, 0.6, 6.0], pr.Color(24, 34, 58, 255), pr.Color(0, 220, 255, 255)),

            # 2. Ramp 1 Climbing to Tier 1 (X: 10 -> 18, Y climbs 0.6 -> 2.6)
            PlatformBox([10.0, 0.4, 1.0], [18.0, 2.6, 5.0], pr.Color(30, 48, 85, 255), pr.Color(0, 180, 240, 255)),

            # 3. Tier 1 Highway Bridge (Y = 2.6, X: 18 -> 22, Z runs 1.0 -> 14.0)
            PlatformBox([18.0, 2.2, 1.0], [22.0, 2.6, 14.0], pr.Color(22, 36, 66, 255), pr.Color(0, 240, 255, 255)),

            # 4. Takeoff Kicker Ramp at Chasm Edge (Z: 14.0 -> 16.0, Y climbs 2.6 -> 3.2)
            PlatformBox([18.0, 2.4, 14.0], [22.0, 3.2, 16.0], pr.Color(160, 50, 100, 255), pr.Color(255, 60, 120, 255)),

            # 5. The Void Chasm (Gap from Z: 16.0 to 20.0!)

            # 6. Receiving Sky-Deck Across the Chasm (Z: 20.0 -> 24.0, Y = 3.2)
            PlatformBox([18.0, 2.8, 20.0], [22.0, 3.2, 24.0], pr.Color(25, 45, 80, 255), pr.Color(0, 255, 180, 255)),

            # 7. Summit Ramp (X climbs 22.0 -> 25.0 onto Summit)
            PlatformBox([22.0, 3.0, 19.0], [25.0, 3.8, 23.0], pr.Color(35, 55, 95, 255), pr.Color(0, 220, 255, 255)),

            # 8. Summit Goal Island (Y = 3.8, elevated summit!)
            PlatformBox([25.0, 3.6, 18.0], [31.0, 4.0, 24.0], pr.Color(18, 55, 65, 255), pr.Color(0, 255, 150, 255)),

            # 9. Perimeter Safety Curbs on Spawn Runway
            PlatformBox([0.0, 0.6, 0.0], [10.0, 1.2, 0.5], pr.Color(50, 70, 110, 255), pr.Color(0, 180, 240, 255)),
            PlatformBox([0.0, 0.6, 5.5], [10.0, 1.2, 6.0], pr.Color(50, 70, 110, 255), pr.Color(0, 180, 240, 255)),
        ]

        # Low Pulsing Laser Hurdle at Bridge Midpoint (Z = 8.5)
        # Height is 0.8m above track. Rolling hits it, jumping sails over it!
        self.laser_hurdle = PlatformBox([18.0, 2.6, 8.4], [22.0, 3.4, 8.6], pr.Color(255, 20, 70, 220), pr.Color(255, 100, 150, 255))


# =====================================================================
# 2. VECTORIZED 3D BALL PHYSICS & COLLISION DYNAMICS
# =====================================================================
class FastVectorizedBallEngine:
    def __init__(self, course: CyberMarbleCourse, num_envs: int = 512):
        self.course = course
        self.num_envs = num_envs
        self.radius = 0.38  # Ball radius

        self.pos = np.zeros((num_envs, 3), dtype=np.float32)
        self.vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.grounded = np.zeros(num_envs, dtype=bool)
        self.crashed = np.zeros(num_envs, dtype=bool)
        self.steps = np.zeros(num_envs, dtype=np.int32)

        self.reset_all()

    def reset_all(self, indices: Optional[np.ndarray] = None):
        if indices is None:
            indices = np.arange(self.num_envs, dtype=np.int32)
        self.pos[indices] = self.course.start_pos
        self.vel[indices] = 0.0
        self.grounded[indices] = True
        self.crashed[indices] = False
        self.steps[indices] = 0

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        actions: [N, 3] -> act[0]=steer_x, act[1]=steer_z (rolling torque), act[2]=jump (>0.0)
        """
        dt = 1.0 / 60.0
        gravity = 9.81 * 1.5

        drive_x = np.clip(actions[:, 0], -1.0, 1.0)
        drive_z = np.clip(actions[:, 1], -1.0, 1.0)
        jump_cmd = actions[:, 2] > 0.0

        # Ground vs Air Acceleration
        accel_mag = np.where(self.grounded, 24.0, 6.0)
        self.vel[:, 0] += drive_x * accel_mag * dt
        self.vel[:, 2] += drive_z * accel_mag * dt

        # Jump Impulse
        can_jump = self.grounded & jump_cmd
        self.vel[:, 1] = np.where(can_jump, 6.2, self.vel[:, 1] - gravity * dt)

        # Rolling Resistance
        drag = np.where(self.grounded, 0.982, 0.995)
        self.vel[:, 0] *= drag
        self.vel[:, 2] *= drag

        # Speed clamp
        speed_horiz = np.sqrt(self.vel[:, 0]**2 + self.vel[:, 2]**2)
        max_speed = 9.5  # m/s
        scale = np.minimum(1.0, max_speed / (speed_horiz + 1e-6))
        self.vel[:, 0] *= scale
        self.vel[:, 2] *= scale

        # Candidate position
        cand_pos = self.pos + self.vel * dt
        self.grounded[:] = False

        # Continuous Sphere-vs-AABB Collision Resolution
        for plat in self.course.platforms:
            closest_x = np.clip(cand_pos[:, 0], plat.min_pt[0], plat.max_pt[0])
            closest_y = np.clip(cand_pos[:, 1], plat.min_pt[1], plat.max_pt[1])
            closest_z = np.clip(cand_pos[:, 2], plat.min_pt[2], plat.max_pt[2])

            diff_x = cand_pos[:, 0] - closest_x
            diff_y = cand_pos[:, 1] - closest_y
            diff_z = cand_pos[:, 2] - closest_z
            dist_sq = diff_x**2 + diff_y**2 + diff_z**2

            colliding = dist_sq < (self.radius**2)
            if np.any(colliding):
                dist = np.sqrt(np.maximum(1e-8, dist_sq))
                norm_x = np.where(dist > 1e-5, diff_x / dist, 0.0)
                norm_y = np.where(dist > 1e-5, diff_y / dist, 1.0)
                norm_z = np.where(dist > 1e-5, diff_z / dist, 0.0)

                overlap = np.maximum(0.0, self.radius - dist)
                cand_pos[colliding, 0] += norm_x[colliding] * overlap[colliding]
                cand_pos[colliding, 1] += norm_y[colliding] * overlap[colliding]
                cand_pos[colliding, 2] += norm_z[colliding] * overlap[colliding]

                # Normal velocity component
                v_dot_n = (self.vel[:, 0] * norm_x + self.vel[:, 1] * norm_y + self.vel[:, 2] * norm_z)
                rebound = np.where(colliding & (v_dot_n < 0), v_dot_n * 1.25, 0.0)
                self.vel[:, 0] -= rebound * norm_x
                self.vel[:, 1] -= rebound * norm_y
                self.vel[:, 2] -= rebound * norm_z

                # Marked grounded if normal points upward
                is_floor = colliding & (norm_y > 0.6)
                self.grounded |= is_floor

        # Laser Hurdle Collision (Hit triggers bounce & penalty)
        hurdle = self.course.laser_hurdle
        in_hx = (cand_pos[:, 0] + self.radius > hurdle.min_pt[0]) & (cand_pos[:, 0] - self.radius < hurdle.max_pt[0])
        in_hy = (cand_pos[:, 1] + self.radius > hurdle.min_pt[1]) & (cand_pos[:, 1] - self.radius < hurdle.max_pt[1])
        in_hz = (cand_pos[:, 2] + self.radius > hurdle.min_pt[2]) & (cand_pos[:, 2] - self.radius < hurdle.max_pt[2])
        hit_laser = in_hx & in_hy & in_hz
        self.vel[hit_laser, 2] = -3.5  # Bounced back by laser barrier

        # Abyss Fall Check
        fell_in_void = cand_pos[:, 1] < -2.0
        self.crashed |= fell_in_void

        self.pos[:] = cand_pos
        self.steps += 1

        # -------------------------------------------------------------
        # 3. REWARD COMPUTATION
        # -------------------------------------------------------------
        to_goal = self.course.goal_pos[None, :] - self.pos
        dist_3d = np.linalg.norm(to_goal, axis=-1)
        dir_3d = to_goal / (dist_3d[:, None] + 1e-6)

        # 3D Velocity progress directly toward Summit Beacon
        progress_vel = self.vel[:, 0] * dir_3d[:, 0] + self.vel[:, 1] * dir_3d[:, 1] + self.vel[:, 2] * dir_3d[:, 2]
        progress_reward = progress_vel * 15.0
        elevation_reward = np.maximum(0.0, self.pos[:, 1] - 0.5) * 8.0

        # Jump clearance bonus: rewarded for flying over laser barrier
        cleared_laser = (self.pos[:, 2] > 9.0) & (self.pos[:, 1] > 3.4)
        jump_bonus = np.where(cleared_laser, 12.0, 0.0)

        # Reached Summit Beacon
        reached_goal = (dist_3d < 2.2) & (self.pos[:, 1] >= 3.7)
        finish_bounty = np.where(reached_goal, 10000.0, 0.0)
        fall_tax = np.where(self.crashed, 80.0, 0.0)

        rewards = progress_reward + elevation_reward + jump_bonus + finish_bounty - fall_tax
        dones = reached_goal | self.crashed | (self.steps >= 360)

        obs = self.get_observations(dist_3d, dir_3d)
        return obs, rewards, dones

    def get_observations(self, dist_3d: np.ndarray, dir_3d: np.ndarray) -> np.ndarray:
        """14-Dim Observation Vector (Relative Beacon Vector, 3D Velocity, Clearance, Floor Contact)."""
        # Horizontal heading angle towards goal
        speed = np.sqrt(self.vel[:, 0]**2 + self.vel[:, 2]**2)

        # Distance to upcoming obstacle / hurdle
        hurdle_dist = np.clip((self.course.laser_hurdle.center[2] - self.pos[:, 2]) / 8.0, -1.0, 1.0)
        chasm_dist = np.clip((16.0 - self.pos[:, 2]) / 8.0, -1.0, 1.0)

        obs = np.column_stack([
            dir_3d[:, 0], dir_3d[:, 1], dir_3d[:, 2],  # 0..2: 3D Unit Vector to Goal Beacon
            dist_3d * 0.03,                           # 3: Normalized Distance to Goal
            self.vel[:, 0] * 0.1,                     # 4: Velocity X
            self.vel[:, 1] * 0.1,                     # 5: Velocity Y (Vertical Climb)
            self.vel[:, 2] * 0.1,                     # 6: Velocity Z (Forward Track Speed)
            speed * 0.1,                              # 7: Total Horizontal Speed
            self.pos[:, 1] * 0.25,                    # 8: Elevation Y
            hurdle_dist,                              # 9: Distance to Laser Hurdle
            chasm_dist,                               # 10: Distance to Chasm Takeoff Kicker
            self.grounded.astype(np.float32),         # 11: Ground Contact Flag
            np.clip(self.pos[:, 0] / 30.0, 0.0, 1.0), # 12: Normalized Arena X
            np.clip(self.pos[:, 2] / 24.0, 0.0, 1.0), # 13: Normalized Arena Z
        ]).astype(np.float32)

        return obs


# =====================================================================
# 3. FAST NEUROEVOLUTION (Antithetic ES Optimizer)
# =====================================================================
class FastBallPolicy:
    def __init__(self, in_dim=14, out_dim=3):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.generation = 0

        # Master Policy Weights
        self.mW1 = np.random.randn(in_dim, 32).astype(np.float32) * 0.03
        self.mb1 = np.zeros(32, dtype=np.float32)
        self.mW2 = np.random.randn(32, 16).astype(np.float32) * 0.03
        self.mb2 = np.zeros(16, dtype=np.float32)
        self.mW3 = np.random.randn(16, out_dim).astype(np.float32) * 0.03
        self.mb3 = np.zeros(out_dim, dtype=np.float32)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.mW1)
        self.vb1 = np.zeros_like(self.mb1)
        self.vW2 = np.zeros_like(self.mW2)
        self.vb2 = np.zeros_like(self.mb2)
        self.vW3 = np.zeros_like(self.mW3)
        self.vb3 = np.zeros_like(self.mb3)

        # --- SEED HIGH-PERFORMANCE NAVIGATION & JUMP PRIOR ---
        # 1. Drive forward along track (Z direction)
        self.mW1[2, 1] = 2.4     # Goal Dir Z -> drive forward
        self.mW1[7, 1] = 1.2     # Speed maintenance
        self.mb1[1] = 0.5
        self.mW2[1, 1] = 1.8
        self.mW3[1, 1] = 1.6
        self.mb3[1] = 1.0        # Continuous forward drive commitment

        # 2. Steer lateral alignment (X direction)
        self.mW1[0, 0] = 2.2     # Goal Dir X -> steer
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.5

        # 3. Timed Jump Thruster: Fires when approaching hurdle or chasm edge
        self.mW1[9, 2] = -2.4    # Close to hurdle -> JUMP!
        self.mW1[10, 2] = -2.6   # Close to chasm takeoff -> JUMP!
        self.mW1[11, 2] = 1.5    # Grounded flag
        self.mb1[2] = 0.2
        self.mW2[2, 2] = 2.0
        self.mW3[2, 2] = 1.8
        self.mb3[2] = 0.1

        # Batched Weights for parallel rollouts
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

        # Controls: [drive_x, drive_z, jump]
        drive_x = raw_out[:, 0:1]
        drive_z = 0.50 + 0.50 * raw_out[:, 1:2]  # Always positive forward roll
        jump = raw_out[:, 2:3]

        return np.hstack([drive_x, drive_z, jump]).astype(np.float32)

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

    def train_epoch(self, engine: FastVectorizedBallEngine, generations=60, rollout_steps=320, verbose=True):
        t0 = time.perf_counter()
        top_fit = -9999.0

        for gen in range(generations):
            engine.reset_all()
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            alive = np.ones(self.pop_size, dtype=bool)
            completed = np.zeros(self.pop_size, dtype=bool)
            steps_to_goal = np.full(self.pop_size, rollout_steps, dtype=np.int32)
            max_elevation = np.zeros(self.pop_size, dtype=np.float32)

            init_dist = np.linalg.norm(engine.pos - engine.course.goal_pos, axis=-1)
            min_dist = init_dist.copy()

            obs = engine.get_observations(
                init_dist,
                (engine.course.goal_pos - engine.pos) / init_dist[:, None],
            )

            for step_idx in range(rollout_steps):
                act = self.forward(obs)
                obs, rewards, dones = engine.step(act)

                cur_dist = obs[:, 3] / 0.03
                elev = engine.pos[:, 1]
                max_elevation = np.maximum(max_elevation, elev)

                reached = alive & (cur_dist < 2.2) & (elev >= 3.7)
                newly_done = reached & (~completed)
                completed |= reached
                steps_to_goal = np.where(newly_done, step_idx + 1, steps_to_goal)

                fitness += np.where(alive, rewards, 0.0)
                min_dist = np.where(alive & (cur_dist < min_dist), cur_dist, min_dist)

                alive &= ~dones
                if not np.any(alive):
                    break

            progress = np.maximum(0.0, init_dist - min_dist)
            fitness += progress * 80.0

            finish_bonus = np.where(completed, 10000.0 + (rollout_steps - steps_to_goal) * 30.0, 0.0)
            fitness += finish_bonus
            fitness += np.where(max_elevation > 3.4, 400.0, 0.0)

            top_fit = float(np.max(fitness))
            min_rem = float(np.min(min_dist))
            num_solved = int(np.sum(completed))

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Closest: {min_rem:4.1f}m | Solved: {num_solved:3d}/{self.pop_size} | Max Elev: {float(np.max(max_elevation)):.2f}m")

            self.evolve(fitness)

        elapsed = time.perf_counter() - t0
        sps = (self.pop_size * rollout_steps * generations) / max(elapsed, 1e-5)
        return elapsed, sps, top_fit


# =====================================================================
# 4. STUDIO-GRADE RAYLIB 3D HARDWARE VISUALIZER
# =====================================================================
class CyberMarbleVisualizer:
    def __init__(self, course: CyberMarbleCourse, ga: FastBallPolicy):
        self.course = course
        self.ga = ga
        self.swarm_size = 32
        self.engine = FastVectorizedBallEngine(course, num_envs=self.swarm_size)
        self.engine.reset_all()

        self.screen_w = 1600
        self.screen_h = 920
        self.hud_h = 120
        self.viewport_h = self.screen_h - self.hud_h
        self.view_3d_w = self.screen_w - 280

        # Enable MSAA 4x hardware anti-aliasing
        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_HIGHDPI | pr.FLAG_VSYNC_HINT)
        pr.init_window(self.screen_w, self.screen_h, "CyberMarble 3D: Autonomous Rolling & Jumping AI")
        pr.set_target_fps(60)

        # 0 = 3D Chase Camera, 1 = 3D Overhead Aerial View, 2 = 2D Tactical View
        self.cam_mode = 0
        self.cam_smoothed = [course.start_pos[0] - 5.0, 3.5, course.start_pos[2] - 5.0]

        self.camera = pr.Camera3D(
            pr.Vector3(self.cam_smoothed[0], self.cam_smoothed[1], self.cam_smoothed[2]),
            pr.Vector3(course.start_pos[0], course.start_pos[1], course.start_pos[2]),
            pr.Vector3(0.0, 1.0, 0.0),
            52.0,
            pr.CAMERA_PERSPECTIVE,
        )

        # 3D Ribbon Trail History
        self.ribbons: List[List[float]] = []  # [x, y, z, lifetime, max_life]
        self.ball_rotation_x = 0.0
        self.ball_rotation_z = 0.0
        self.show_swarm = True
        self.paused = False

    def draw_3d_marble(self, pos: np.ndarray, vel: np.ndarray, grounded: bool):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        speed = math.sqrt(float(vel[0]**2 + vel[2]**2))

        # Roll rotation angle update
        if speed > 0.05:
            self.ball_rotation_z -= (vel[0] / self.engine.radius) * (1.0 / 60.0)
            self.ball_rotation_x += (vel[2] / self.engine.radius) * (1.0 / 60.0)

        ball_center = pr.Vector3(x, y, z)

        # 1. Glowing Cyberpunk Marble Core
        core_color = pr.Color(0, 245, 255, 255) if grounded else pr.Color(230, 80, 255, 255)
        pr.draw_sphere(ball_center, self.engine.radius, core_color)
        pr.draw_sphere_wires(ball_center, self.engine.radius, 12, 12, pr.RAYWHITE)

        # 2. Dual Gyro-Stabilizer Rings (Rotating in 3D around sphere)
        pr.draw_circle_3d(ball_center, self.engine.radius + 0.06, pr.Vector3(1, 0, 0), math.degrees(self.ball_rotation_x), pr.Color(255, 230, 80, 220))
        pr.draw_circle_3d(ball_center, self.engine.radius + 0.06, pr.Vector3(0, 0, 1), math.degrees(self.ball_rotation_z), pr.Color(255, 60, 140, 220))

        # 3. Rocket Thruster Plasma Plume when Airborne
        if not grounded:
            thrust_pos = pr.Vector3(x, y - self.engine.radius - 0.15, z)
            pr.draw_sphere(thrust_pos, 0.22, pr.ORANGE)
            pr.draw_sphere(thrust_pos, 0.12, pr.YELLOW)

        # 4. Ground Shadow
        pr.draw_circle_3d(pr.Vector3(x, y - self.engine.radius + 0.02, z), self.engine.radius * 0.9, pr.Vector3(0, 1, 0), 0.0, pr.Color(0, 0, 0, 160))

    def draw_course(self):
        # 1. Multi-Tier Platforms with Glowing Neon Borders
        for plat in self.course.platforms:
            pr.draw_cube(pr.Vector3(float(plat.center[0]), float(plat.center[1]), float(plat.center[2])),
                         float(plat.size[0]), float(plat.size[1]), float(plat.size[2]), plat.color)
            pr.draw_cube_wires(pr.Vector3(float(plat.center[0]), float(plat.center[1]), float(plat.center[2])),
                               float(plat.size[0]), float(plat.size[1]), float(plat.size[2]), plat.border_color)

        # 2. Glowing Laser Hurdle Barrier (Pulsing Red)
        hurdle = self.course.laser_hurdle
        pulse = abs(math.sin(time.time() * 6.0)) * 0.15
        pr.draw_cube(pr.Vector3(float(hurdle.center[0]), float(hurdle.center[1]), float(hurdle.center[2])),
                     float(hurdle.size[0]), float(hurdle.size[1]), float(hurdle.size[2]), pr.Color(255, 20, 70, 210))
        pr.draw_cube_wires(pr.Vector3(float(hurdle.center[0]), float(hurdle.center[1]), float(hurdle.center[2])),
                           float(hurdle.size[0]), float(hurdle.size[1]), float(hurdle.size[2]), pr.Color(255, 120, 160, 255))

        # 3. 3D Emerald Goal Beacon & Orbiting Crystal at Summit
        gx, gy, gz = float(self.course.goal_pos[0]), float(self.course.goal_pos[1]), float(self.course.goal_pos[2])
        beacon_base = pr.Vector3(gx, gy - 0.4, gz)
        pr.draw_cylinder(beacon_base, 0.5, 0.5, 6.0, 18, pr.Color(0, 255, 128, 190))
        crystal_pos = pr.Vector3(gx, gy + 4.5, gz)
        pr.draw_sphere(crystal_pos, 0.7, pr.Color(100, 255, 180, 255))
        pr.draw_sphere_wires(crystal_pos, 0.75, 8, 8, pr.WHITE)

        # Expanding Ground Energy Rings
        ring_r = 0.5 + float(math.fmod(time.time() * 2.5, 3.0))
        pr.draw_circle_3d(beacon_base, ring_r, pr.Vector3(0, 1, 0), 0.0, pr.Color(0, 255, 128, 200))

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
        init_dist = np.linalg.norm(self.engine.pos - self.course.goal_pos, axis=-1)
        obs = self.engine.get_observations(
            init_dist,
            (self.course.goal_pos - self.engine.pos) / init_dist[:, None],
        )

        while not pr.window_should_close():
            if max_frames is not None and frame_count >= max_frames:
                break

            # Keyboard Inputs
            if pr.is_key_pressed(pr.KEY_SPACE):
                self.paused = not self.paused
            elif pr.is_key_pressed(pr.KEY_C):
                self.cam_mode = (self.cam_mode + 1) % 3
                modes = ["3D Chase Cam", "3D Aerial Overview", "2D Tactical Radar"]
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

            # Append 3D Ribbon Trail
            self.ribbons.append([c_pos[0], c_pos[1] - 0.1, c_pos[2], 1.0, 1.0])

            # 3D Camera Smoothing
            if self.cam_mode == 0:  # 3D Chase Camera
                target_x = c_pos[0] - 4.5
                target_y = max(2.5, c_pos[1] + 2.8)
                target_z = c_pos[2] - 4.5

                self.cam_smoothed[0] += (target_x - self.cam_smoothed[0]) * 0.20
                self.cam_smoothed[1] += (target_y - self.cam_smoothed[1]) * 0.20
                self.cam_smoothed[2] += (target_z - self.cam_smoothed[2]) * 0.20

                self.camera.position = pr.Vector3(float(self.cam_smoothed[0]), float(self.cam_smoothed[1]), float(self.cam_smoothed[2]))
                self.camera.target = pr.Vector3(float(c_pos[0] + 1.2), float(c_pos[1] + 0.4), float(c_pos[2] + 1.2))
                self.camera.up = pr.Vector3(0.0, 1.0, 0.0)
                self.camera.fovy = 52.0

            elif self.cam_mode == 1:  # 3D Aerial Orbit View
                self.camera.position = pr.Vector3(15.0, 24.0, -2.0)
                self.camera.target = pr.Vector3(16.0, 2.0, 14.0)
                self.camera.up = pr.Vector3(0.0, 1.0, 0.0)
                self.camera.fovy = 58.0

            # -------------------------------------------------------------
            # BEGIN HARDWARE 3D RENDERING
            # -------------------------------------------------------------
            pr.begin_drawing()
            # Deep Space Synthwave Sky
            pr.clear_background(pr.Color(12, 16, 28, 255))

            if self.cam_mode in (0, 1):
                pr.begin_mode_3d(self.camera)

                # Ground Grid & Chasm Depth Lines
                pr.draw_grid(36, 1.0)

                # 3D Multi-Tier Course Platforms
                self.draw_course()

                # 3D Trajectory Ribbon Particles
                self.draw_ribbons()

                # Ghost Swarm (3D)
                if self.show_swarm:
                    for i in range(1, self.swarm_size):
                        sp = self.engine.pos[i]
                        pr.draw_sphere(pr.Vector3(float(sp[0]), float(sp[1]), float(sp[2])), 0.24, pr.Color(240, 60, 160, 200))

                # Champion 3D Cyber-Marble
                self.draw_3d_marble(c_pos, c_vel, c_grounded)

                pr.end_mode_3d()

            else:  # Mode 2: 2D Tactical View
                cell_s = int(min((self.view_3d_w - 80) / 32.0, self.viewport_h / 26.0))
                off_x, off_y = 50, 30

                for plat in self.course.platforms:
                    rx = int(off_x + plat.min_pt[0] * cell_s)
                    ry = int(off_y + plat.min_pt[2] * cell_s)
                    rw = int(plat.size[0] * cell_s)
                    rh = int(plat.size[2] * cell_s)
                    pr.draw_rectangle(rx, ry, rw, rh, pr.Color(35, 50, 90, 255))
                    pr.draw_rectangle_lines(rx, ry, rw, rh, pr.Color(0, 220, 255, 255))

                # Hurdle & Goal
                hrx = int(off_x + self.course.laser_hurdle.min_pt[0] * cell_s)
                hry = int(off_y + self.course.laser_hurdle.min_pt[2] * cell_s)
                pr.draw_rectangle(hrx, hry, int(self.course.laser_hurdle.size[0] * cell_s), int(self.course.laser_hurdle.size[2] * cell_s), pr.RED)

                gx = int(off_x + self.course.goal_pos[0] * cell_s)
                gy = int(off_y + self.course.goal_pos[2] * cell_s)
                pr.draw_circle(gx, gy, 12, pr.GREEN)

                # Ball
                bx = int(off_x + c_pos[0] * cell_s)
                by = int(off_y + c_pos[2] * cell_s)
                pr.draw_circle(bx, by, 7, pr.Color(0, 245, 255, 255))

            # -------------------------------------------------------------
            # CRISP 2D OVERLAYS (PiP Radar, Neural Monitor, HUD)
            # -------------------------------------------------------------
            # 1. PiP Tactical Radar in Top-Left
            pr.draw_rectangle(25, 25, 200, 160, pr.Color(16, 24, 44, 235))
            pr.draw_rectangle_lines(25, 25, 200, 160, pr.Color(0, 220, 255, 255))
            cw, ch = 200.0 / 32.0, 160.0 / 26.0

            for plat in self.course.platforms:
                pr.draw_rectangle(int(25 + plat.min_pt[0] * cw), int(25 + plat.min_pt[2] * ch),
                                  int(plat.size[0] * cw) + 1, int(plat.size[2] * ch) + 1, pr.Color(45, 65, 105, 255))

            pr.draw_circle(int(25 + self.course.goal_pos[0] * cw), int(25 + self.course.goal_pos[2] * ch), 6, pr.GREEN)
            pr.draw_circle(int(25 + c_pos[0] * cw), int(25 + c_pos[2] * ch), 5, pr.Color(0, 245, 255, 255))
            pr.draw_text("RADAR [PiP]", 34, 32, 16, pr.Color(0, 220, 255, 255))

            # 2. Right-Hand Neural Monitor
            panel_x = self.view_3d_w
            panel_w = self.screen_w - panel_x
            pr.draw_rectangle(panel_x, 0, panel_w, self.viewport_h, pr.Color(20, 28, 48, 255))
            pr.draw_line(panel_x, 0, panel_x, self.viewport_h, pr.Color(0, 220, 255, 255))

            pr.draw_text("NEURAL MONITOR", panel_x + 20, 20, 22, pr.Color(0, 220, 255, 255))
            pr.draw_text("AI GYRO SENSORS", panel_x + 20, 60, 16, pr.Color(190, 210, 240, 255))

            sensor_labels = ["BEACON DX", "BEACON DY", "BEACON DZ", "SPEED", "ELEVATION", "HURDLE PROX", "CHASM PROX"]
            sensor_vals = [float(obs[0, 0]), float(obs[0, 1]), float(obs[0, 2]), speed * 0.1, float(c_pos[1]) * 0.25, float(obs[0, 9]), float(obs[0, 10])]
            for i, (lbl, val) in enumerate(zip(sensor_labels, sensor_vals)):
                by = 90 + i * 24
                pr.draw_text(lbl, panel_x + 20, by, 13, pr.LIGHTGRAY)
                pr.draw_rectangle(panel_x + 120, by, 120, 14, pr.Color(32, 45, 75, 255))
                bar_w = int(np.clip(abs(val), 0.0, 1.0) * 120)
                bar_c = pr.Color(0, 220, 255, 255) if val >= 0 else pr.Color(255, 60, 90, 255)
                pr.draw_rectangle(panel_x + 120, by, bar_w, 14, bar_c)

            # Hidden Layer
            pr.draw_text("HIDDEN LAYER (16-Tanh)", panel_x + 20, 280, 16, pr.Color(190, 210, 240, 255))
            for row in range(4):
                for col in range(4):
                    idx = row * 4 + col
                    intensity = int(abs(math.sin(time.time() * 2.0 + idx)) * 200 + 40)
                    rx, ry = panel_x + 25 + col * 52, 310 + row * 30
                    pr.draw_rectangle(rx, ry, 44, 22, pr.Color(0, int(intensity * 0.8), intensity, 255))
                    pr.draw_text(f"N{idx:02d}", rx + 8, ry + 4, 13, pr.WHITE)

            # Compass Pointing to Goal Beacon
            pr.draw_text("BIRD'S-EYE BEACON", panel_x + 20, 460, 16, pr.Color(190, 210, 240, 255))
            cx, cy = panel_x + 130, 550
            pr.draw_circle(cx, cy, 54, pr.Color(28, 42, 70, 255))
            pr.draw_circle_lines(cx, cy, 54, pr.Color(0, 220, 255, 255))

            to_beacon = self.course.goal_pos - c_pos
            norm_b = math.sqrt(float(to_beacon[0]**2 + to_beacon[2]**2)) + 1e-6
            bx, bz = float(to_beacon[0] / norm_b), float(to_beacon[2] / norm_b)
            pr.draw_line(cx, cy, int(cx + bx * 48), int(cy + bz * 48), pr.Color(0, 255, 255, 255))
            pr.draw_circle(int(cx + bx * 48), int(cy + bz * 48), 6, pr.Color(0, 255, 255, 255))

            # 3. Bottom HUD
            hud_y = self.viewport_h
            pr.draw_rectangle(0, hud_y, self.screen_w, self.hud_h, pr.Color(14, 20, 36, 255))
            pr.draw_line(0, hud_y, self.screen_w, hud_y, pr.Color(0, 220, 255, 255))

            d_3d = float(np.linalg.norm(c_pos - self.course.goal_pos))
            pr.draw_text(f"GEN {self.ga.generation:03d} [OpenAI-ES]", 30, hud_y + 16, 26, pr.Color(0, 220, 255, 255))
            pr.draw_text(f"ROLLING SPEED: {speed*3.6:4.1f} km/h", 30, hud_y + 48, 18, pr.WHITE)
            elev_c = pr.Color(230, 140, 255, 255) if c_pos[1] > 2.0 else pr.Color(180, 200, 225, 255)
            pr.draw_text(f"ELEVATION Y  : {c_pos[1]:4.2f} m", 30, hud_y + 70, 18, elev_c)
            pr.draw_text(f"DIST TO GOAL : {d_3d:4.1f} m", 30, hud_y + 92, 18, pr.LIME)

            # Actuators
            pr.draw_line(340, hud_y + 12, 340, hud_y + 110, pr.Color(45, 65, 100, 255))
            pr.draw_text("BALL 3D MOTORS", 365, hud_y + 16, 22, pr.Color(200, 225, 255, 255))

            # Lateral Steer Torque
            drive_x = float(actions[0, 0])
            pr.draw_rectangle(365, hud_y + 48, 140, 16, pr.Color(32, 45, 75, 255))
            s_bar_w = int(drive_x * 68)
            s_color = pr.RED if drive_x < 0 else pr.Color(0, 220, 255, 255)
            pr.draw_rectangle(435 if drive_x > 0 else 435 + s_bar_w, hud_y + 48, abs(s_bar_w), 16, s_color)
            pr.draw_text(f"TORQUE X [{drive_x:+.2f}]", 515, hud_y + 48, 16, pr.WHITE)

            # Forward Drive Force
            drive_z = float(actions[0, 1])
            pr.draw_rectangle(365, hud_y + 70, 140, 16, pr.Color(32, 45, 75, 255))
            pr.draw_rectangle(365, hud_y + 70, int(max(0.0, drive_z) * 140), 16, pr.LIME)
            pr.draw_text(f"ROLL Z   [{drive_z:.2f}]", 515, hud_y + 70, 16, pr.WHITE)

            # Jump Thruster
            is_firing = float(actions[0, 2]) > 0.0
            j_color = pr.Color(230, 140, 255, 255) if is_firing else pr.GRAY
            j_text = "VERTICAL JUMP THRUSTER: [ACTIVE LAUNCH]" if is_firing else "VERTICAL JUMP THRUSTER: [GROUND]"
            pr.draw_text(j_text, 365, hud_y + 94, 17, j_color)

            # Camera Guide
            pr.draw_line(720, hud_y + 12, 720, hud_y + 110, pr.Color(45, 65, 100, 255))
            c_modes = ["3D CHASE CAM", "3D AERIAL OVERVIEW", "2D TACTICAL RADAR"]
            pr.draw_text(f"CAMERA: [{c_modes[self.cam_mode]}] (Press 'C')", 745, hud_y + 16, 20, pr.Color(220, 140, 255, 255))
            keys = [
                "[C]     Toggle Camera Mode (Chase / Aerial / 2D)",
                "[SPACE] Pause / Play Simulation",
                "[G]     Toggle Ghost Swarm",
                "[R]     Reset to Launchpad",
            ]
            for i, k in enumerate(keys):
                pr.draw_text(k, 745, hud_y + 44 + i * 18, 14, pr.Color(190, 210, 235, 255))

            pr.end_drawing()

            # Record Video
            if video_writer is not None:
                img = pr.load_image_from_screen()
                pr.image_flip_vertical(img)
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
    parser = argparse.ArgumentParser(description="CyberMarble 3D: Autonomous Rolling & Jumping AI")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberMarble 3D: Autonomous Rolling & Jumping Stunt AI Course  ")
    print("=================================================================")

    course = CyberMarbleCourse()
    print("1. Constructing 3D Multi-Tier Stunt Course...")
    print(f"   Launch Spawn Pt : {course.start_pos} (Tier 0: Ground)")
    print(f"   Summit Goal Pt  : {course.goal_pos} (Tier 2: Sky-Deck, Y=4.4m)")
    print(f"   Laser Hurdle    : Z = 8.5m (Must time jump over beam)")
    print(f"   The Void Chasm  : Z = 16.0m -> 20.0m (4-meter aerial gap leap)")

    print(f"\n2. Evolving 512 Agents via Antithetic ES ({args.generations} Generations)...")
    engine = FastVectorizedBallEngine(course, num_envs=512)
    policy = FastBallPolicy(in_dim=14, out_dim=3)

    elapsed, sps, top_fit = policy.train_epoch(engine, generations=args.generations, rollout_steps=320, verbose=True)
    print(f"\n   Training Finished in {elapsed:.2f}s! ({sps:,.0f} agent-steps/sec)")
    print(f"   Champion Fitness: {top_fit:.1f}")

    print("\n3. Launching Hardware-Accelerated Raylib 3D Engine...")
    print("   Controls: [C] Toggle View (Chase / Aerial / 2D) | [SPACE] Pause | [R] Reset\n")

    max_frames = args.frames if args.video else None
    viz = CyberMarbleVisualizer(course, policy)
    viz.run(video_path=args.video, max_frames=max_frames)


if __name__ == "__main__":
    main()

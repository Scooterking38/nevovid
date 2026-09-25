# tag_arena.py
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import mujoco
except ImportError:
    print("\n[!] 'mujoco' is not installed.")
    print("    Please install it using: pip install mujoco\n")
    sys.exit(1)


# =====================================================================
# 1. 3D TACTICAL BOX CONTAINER ARENA (MJCF XML)
# =====================================================================
MJCF_TAG_ARENA = """
<mujoco model="cyber_tag_topdown">
  <compiler autolimits="true" coordinate="local"/>
  <!-- Heavy snappy gravity (-34 m/s^2) for rapid vertical jumping -->
  <option gravity="0 0 -34.0" timestep="0.016666"/>

  <visual>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.28 0.28 0.38" specular="0.6 0.6 0.6"/>
    <rgba fog="0.06 0.08 0.16 1"/>
    <quality shadowsize="2048"/>
    <global elevation="-84" azimuth="90" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.14 0.28" rgb2="0.03 0.04 0.08" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.10 0.14 0.24" rgb2="0.06 0.08 0.15"/>
    <material name="grid_mat" texture="grid" texrepeat="20 20" reflectance="0.25"/>

    <material name="wall_mat" rgba="0.14 0.18 0.28 1" specular="0.4" shininess="0.3"/>
    <material name="pillar_mat" rgba="0.18 0.24 0.40 1" specular="0.8" shininess="0.6"/>
    <material name="hurdle_mat" rgba="0.8 0.2 0.3 1" emission="0.4"/>
    <material name="hub_mat" rgba="0.12 0.32 0.52 1" specular="0.7"/>

    <material name="tagger_mat" rgba="1.0 0.12 0.22 1" emission="0.85" specular="1" shininess="1"/>
    <material name="avoider_mat" rgba="0.0 0.95 1.0 1" emission="0.85" specular="1" shininess="1"/>
  </asset>

  <worldbody>
    <light pos="0 0 20" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.6 0.6 0.6"/>
    <light pos="0 6 12" dir="0 0 -1" diffuse="0.5 0.6 0.8"/>

    <!-- Arena Floor -->
    <geom name="floor" type="plane" size="12 12 1" pos="0 0 0" material="grid_mat"/>

    <!-- 4 Enclosing Container Walls (16m x 16m Box, 4m High) -->
    <geom name="wall_north" type="box" size="8.2 0.2 2.0" pos="0 8.0 2.0" material="wall_mat"/>
    <geom name="wall_south" type="box" size="8.2 0.2 2.0" pos="0 -8.0 2.0" material="wall_mat"/>
    <geom name="wall_east"  type="box" size="0.2 8.2 2.0" pos="8.0 0 2.0" material="wall_mat"/>
    <geom name="wall_west"  type="box" size="0.2 8.2 2.0" pos="-8.0 0 2.0" material="wall_mat"/>

    <!-- 4 Stealth Pillars (Block LiDAR beams for stealth ambushes!) -->
    <geom name="pillar_nw" type="box" size="1.2 1.2 1.6" pos="-4.5 4.5 1.6" material="pillar_mat"/>
    <geom name="pillar_ne" type="box" size="1.2 1.2 1.6" pos="4.5 4.5 1.6" material="pillar_mat"/>
    <geom name="pillar_sw" type="box" size="1.2 1.2 1.6" pos="-4.5 -4.5 1.6" material="pillar_mat"/>
    <geom name="pillar_se" type="box" size="1.2 1.2 1.6" pos="4.5 -4.5 1.6" material="pillar_mat"/>

    <!-- Center Elevated Tactical Hub (Height: 0.9m) -->
    <geom name="center_hub" type="box" size="1.8 1.8 0.45" pos="0 0 0.45" material="hub_mat"/>

    <!-- 2 Low Corridor Hurdles (Must jump over to pass!) -->
    <geom name="hurdle_n" type="box" size="1.5 0.15 0.3" pos="0 5.0 0.3" material="hurdle_mat"/>
    <geom name="hurdle_s" type="box" size="1.5 0.15 0.3" pos="0 -5.0 0.3" material="hurdle_mat"/>

    <!-- AGENT 1: TAGGER (Red Predator Cyber-Sphere) -->
    <body name="tagger" pos="-5 -5 0.5">
      <freejoint name="tagger_joint"/>
      <geom name="tagger_geom" type="sphere" size="0.42" mass="1.2"
            friction="2.0 0.1 0.02" solref="0.015 1.0" material="tagger_mat"/>
    </body>

    <!-- AGENT 2: AVOIDER (Cyan Prey Cyber-Sphere) -->
    <body name="avoider" pos="5 5 0.5">
      <freejoint name="avoider_joint"/>
      <geom name="avoider_geom" type="sphere" size="0.36" mass="0.9"
            friction="2.0 0.1 0.02" solref="0.015 1.0" material="avoider_mat"/>
    </body>
  </worldbody>
</mujoco>
"""


# =====================================================================
# 2. PURE 360° LIDAR ENGINE (No Cheat Inputs)
# =====================================================================
class CyberTagEnv:
    def __init__(self, num_lidar_rays: int = 16):
        self.model = mujoco.MjModel.from_xml_string(MJCF_TAG_ARENA)
        self.data = mujoco.MjData(self.model)

        self.tagger_bid = self.model.body("tagger").id
        self.avoider_bid = self.model.body("avoider").id
        self.tagger_gid = self.model.geom("tagger_geom").id
        self.avoider_gid = self.model.geom("avoider_geom").id

        self.r_tagger = 0.42
        self.r_avoider = 0.36
        self.tag_dist_threshold = self.r_tagger + self.r_avoider + 0.06

        self.num_rays = num_lidar_rays
        self.max_range = 16.0  # Full arena span

        # 360-degree unit direction vectors
        angles = np.linspace(0.0, 2 * np.pi, self.num_rays, endpoint=False)
        self.ray_dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float64)

        self.steps = 0
        self.max_steps = 350
        self.is_tagged = False

        # Hit caches for visualization
        self.tagger_hits: List[Tuple[float, float, float, bool]] = []
        self.avoider_hits: List[Tuple[float, float, float, bool]] = []

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

        # Randomize spawn positions
        angle_t = np.random.uniform(0, 2 * np.pi)
        dist_t = np.random.uniform(4.0, 6.5)
        self.data.qpos[0:3] = [dist_t * np.cos(angle_t), dist_t * np.sin(angle_t), 0.5]
        self.data.qpos[3:7] = [1, 0, 0, 0]

        angle_a = angle_t + np.pi + np.random.uniform(-0.6, 0.6)
        dist_a = np.random.uniform(4.0, 6.5)
        self.data.qpos[7:10] = [dist_a * np.cos(angle_a), dist_a * np.sin(angle_a), 0.5]
        self.data.qpos[10:14] = [1, 0, 0, 0]

        self.data.qvel[:] = 0.0
        self.is_tagged = False
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

        return self.get_observations()

    def _cast_360_lidar(self, origin: np.ndarray, own_body_id: int, opponent_geom_id: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float, bool]]]:
        """
        Fires 16 true 3D rays in a full 360-degree circular fan.
        """
        geomid = np.empty(1, dtype=np.int32)
        obs_dist = np.ones(self.num_rays, dtype=np.float32)
        opp_dist = np.ones(self.num_rays, dtype=np.float32)
        hit_pts = []

        pnt = np.ascontiguousarray(origin, dtype=np.float64)

        for k in range(self.num_rays):
            vec = self.ray_dirs[k]
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                pnt=pnt,
                vec=vec,
                geomgroup=None,
                flg_static=1,
                bodyexclude=own_body_id,
                geomid=geomid,
            )

            is_opp = False
            hit_d = self.max_range
            if 0.0 <= dist <= self.max_range:
                hit_d = dist
                norm_d = float(dist / self.max_range)
                if geomid[0] == opponent_geom_id:
                    opp_dist[k] = norm_d
                    is_opp = True
                else:
                    obs_dist[k] = norm_d

            hx = origin[0] + vec[0] * hit_d
            hy = origin[1] + vec[1] * hit_d
            hz = origin[2] + vec[2] * hit_d
            hit_pts.append((hx, hy, hz, is_opp))

        return obs_dist, opp_dist, hit_pts

    def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        PURE 360-DEGREE LIDAR ONLY (32 Features):
        - [0..15] : 16 Distance beams to static walls & pillars
        - [16..31]: 16 Detection beams spotting the opponent
        """
        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        # Tagger 360 LiDAR
        t_walls, t_opp, self.tagger_hits = self._cast_360_lidar(p_tag, self.tagger_bid, self.avoider_gid)
        obs_tagger = np.concatenate([t_walls, t_opp]).astype(np.float32)

        # Avoider 360 LiDAR
        a_walls, a_opp, self.avoider_hits = self._cast_360_lidar(p_avd, self.avoider_bid, self.tagger_gid)
        obs_avoider = np.concatenate([a_walls, a_opp]).astype(np.float32)

        return obs_tagger, obs_avoider

    def step(self, act_tagger: np.ndarray, act_avoider: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool]:
        self.data.qfrc_applied[:] = 0.0

        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        # Snappy Drive (No floating ice momentum!)
        self.data.qfrc_applied[0] = act_tagger[0] * 38.0
        self.data.qfrc_applied[1] = act_tagger[1] * 38.0

        # Tagger Rocket Jump
        if act_tagger[2] > 0.0 and p_tag[2] < 2.0 and abs(self.data.qvel[2]) < 0.6:
            self.data.qfrc_applied[2] = 95.0

        # Avoider Snappy Drive & High Jump
        self.data.qfrc_applied[6] = act_avoider[0] * 34.0
        self.data.qfrc_applied[7] = act_avoider[1] * 34.0

        if act_avoider[2] > 0.0 and p_avd[2] < 2.0 and abs(self.data.qvel[8]) < 0.6:
            self.data.qfrc_applied[8] = 90.0

        # High damping to eliminate excessive gliding
        self.data.qvel[0:2] *= 0.88
        self.data.qvel[3:5] *= 0.88
        self.data.qvel[6:8] *= 0.88
        self.data.qvel[9:11] *= 0.88

        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        p_tag_after = self.data.qpos[0:3]
        p_avd_after = self.data.qpos[7:10]
        dist = np.linalg.norm(p_tag_after - p_avd_after)

        # Check Tag Collision
        tagged = False
        if dist < self.tag_dist_threshold:
            tagged = True
        else:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if (c.geom1 == self.tagger_gid and c.geom2 == self.avoider_gid) or \
                   (c.geom2 == self.tagger_gid and c.geom1 == self.avoider_gid):
                    tagged = True
                    break

        self.is_tagged = tagged

        rew_tagger = -dist * 0.4 - 0.1
        rew_avoider = dist * 0.4 + 0.4

        if tagged:
            rew_tagger += 300.0
            rew_avoider -= 300.0

        done = tagged or (self.steps >= self.max_steps)
        obs_t, obs_a = self.get_observations()

        return (obs_t, obs_a), (rew_tagger, rew_avoider), done


# =====================================================================
# 3. SEPARATE 360° LIDAR POLICIES
# =====================================================================
class AgentPolicy:
    def __init__(self, in_dim=32, out_dim=3, role="tagger"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.role = role

        self.W1 = np.random.randn(in_dim, 32).astype(np.float32) * 0.04
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = np.random.randn(32, 16).astype(np.float32) * 0.04
        self.b2 = np.zeros(16, dtype=np.float32)
        self.W3 = np.random.randn(16, out_dim).astype(np.float32) * 0.04
        self.b3 = np.zeros(out_dim, dtype=np.float32)

        # Inductive 360° LiDAR Priors:
        angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        for k in range(16):
            dx, dy = np.cos(angles[k]), np.sin(angles[k])
            if role == "tagger":
                # Steer towards opponent when spotted on ray k
                self.W1[16 + k, 0] += -dx * 2.2
                self.W1[16 + k, 1] += -dy * 2.2
                # Repel away from walls and pillars
                self.W1[k, 0] += dx * 1.0
                self.W1[k, 1] += dy * 1.0
            else:
                # Flee away from opponent when spotted on ray k
                self.W1[16 + k, 0] += dx * 2.5
                self.W1[16 + k, 1] += dy * 2.5
                # Repel away from walls
                self.W1[k, 0] += dx * 1.2
                self.W1[k, 1] += dy * 1.2
                # Reflex jump when hunter lunges close!
                self.W1[16 + k, 2] += -1.5

        self.W2[0, 0] = 1.6
        self.W2[1, 1] = 1.6
        self.W2[2, 2] = 1.6
        self.W3[0, 0] = 1.5
        self.W3[1, 1] = 1.5
        self.W3[2, 2] = 1.5

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.mb2) if hasattr(self, 'mb2') else np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, obs: np.ndarray, weights=None) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3) if weights is None else weights
        h1 = np.tanh(np.dot(obs, w1) + b1)
        h2 = np.tanh(np.dot(h1, w2) + b2)
        out = np.tanh(np.dot(h2, w3) + b3)
        return out


# =====================================================================
# 4. ADVERSARIAL CO-EVOLUTION TRAINER
# =====================================================================
class CoEvolutionaryTagTrainer:
    def __init__(self, env: CyberTagEnv, policy_tagger: AgentPolicy, policy_avoider: AgentPolicy):
        self.env = env
        self.tagger = policy_tagger
        self.avoider = policy_avoider

    def train_epoch(self, generations=40, pop_size=32, rollout_steps=320, verbose=True):
        t0 = time.perf_counter()
        half = pop_size // 2
        sigma = 0.08

        for gen in range(generations):
            eW1_t = [np.random.randn(*self.tagger.W1.shape).astype(np.float32) for _ in range(half)]
            eW2_t = [np.random.randn(*self.tagger.W2.shape).astype(np.float32) for _ in range(half)]
            eW3_t = [np.random.randn(*self.tagger.W3.shape).astype(np.float32) for _ in range(half)]

            eW1_a = [np.random.randn(*self.avoider.W1.shape).astype(np.float32) for _ in range(half)]
            eW2_a = [np.random.randn(*self.avoider.W2.shape).astype(np.float32) for _ in range(half)]
            eW3_a = [np.random.randn(*self.avoider.W3.shape).astype(np.float32) for _ in range(half)]

            fits_tagger = np.zeros(pop_size, dtype=np.float32)
            fits_avoider = np.zeros(pop_size, dtype=np.float32)
            tags_count = 0

            for i in range(pop_size):
                sign = 1.0 if i < half else -1.0
                idx = i if i < half else i - half

                w_t = (
                    self.tagger.W1 + sign * sigma * eW1_t[idx], self.tagger.b1,
                    self.tagger.W2 + sign * sigma * eW2_t[idx], self.tagger.b2,
                    self.tagger.W3 + sign * sigma * eW3_t[idx], self.tagger.b3,
                )
                w_a = (
                    self.avoider.W1 + sign * sigma * eW1_a[idx], self.avoider.b1,
                    self.avoider.W2 + sign * sigma * eW2_a[idx], self.avoider.b2,
                    self.avoider.W3 + sign * sigma * eW3_a[idx], self.avoider.b3,
                )

                obs_t, obs_a = self.env.reset()
                r_sum_t, r_sum_a = 0.0, 0.0

                for _ in range(rollout_steps):
                    act_t = self.tagger.forward(obs_t, weights=w_t)
                    act_a = self.avoider.forward(obs_a, weights=w_a)

                    (obs_t, obs_a), (rew_t, rew_a), done = self.env.step(act_t, act_a)
                    r_sum_t += rew_t
                    r_sum_a += rew_a

                    if done:
                        if self.env.is_tagged:
                            tags_count += 1
                        break

                fits_tagger[i] = r_sum_t
                fits_avoider[i] = r_sum_a

            # Gradient update for Tagger
            norm_t = (fits_tagger - np.mean(fits_tagger)) / (np.std(fits_tagger) + 1e-6)
            diff_t = norm_t[:half] - norm_t[half:]
            gW1_t = np.mean([diff_t[j] * eW1_t[j] for j in range(half)], axis=0)
            gW2_t = np.mean([diff_t[j] * eW2_t[j] for j in range(half)], axis=0)
            gW3_t = np.mean([diff_t[j] * eW3_t[j] for j in range(half)], axis=0)

            lr, beta = 0.05, 0.85
            self.tagger.vW1 = beta * self.tagger.vW1 + lr * gW1_t
            self.tagger.vW2 = beta * self.tagger.vW2 + lr * gW2_t
            self.tagger.vW3 = beta * self.tagger.vW3 + lr * gW3_t
            self.tagger.W1 += self.tagger.vW1
            self.tagger.W2 += self.tagger.vW2
            self.tagger.W3 += self.tagger.vW3

            # Gradient update for Avoider
            norm_a = (fits_avoider - np.mean(fits_avoider)) / (np.std(fits_avoider) + 1e-6)
            diff_a = norm_a[:half] - norm_a[half:]
            gW1_a = np.mean([diff_a[j] * eW1_a[j] for j in range(half)], axis=0)
            gW2_a = np.mean([diff_a[j] * eW2_a[j] for j in range(half)], axis=0)
            gW3_a = np.mean([diff_a[j] * eW3_a[j] for j in range(half)], axis=0)

            self.avoider.vW1 = beta * self.avoider.vW1 + lr * gW1_a
            self.avoider.vW2 = beta * self.avoider.vW2 + lr * gW2_a
            self.avoider.vW3 = beta * self.avoider.vW3 + lr * gW3_a
            self.avoider.W1 += self.avoider.vW1
            self.avoider.W2 += self.avoider.vW2
            self.avoider.W3 += self.avoider.vW3

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Tags: {tags_count:2d}/{pop_size} | Tagger Fit: {float(np.max(fits_tagger)):6.1f} | Avoider Fit: {float(np.max(fits_avoider)):6.1f}")

        elapsed = time.perf_counter() - t0
        return elapsed


# =====================================================================
# 5. HIGH-OCTANE TOP-DOWN VISUALIZER & VIDEO COMPOSITOR
# =====================================================================
class TopDownStudioVisualizer:
    def __init__(self, env: CyberTagEnv, policy_tagger: AgentPolicy, policy_avoider: AgentPolicy):
        self.env = env
        self.tagger = policy_tagger
        self.avoider = policy_avoider
        self.width = 1280
        self.height = 720
        self.renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)

        # High-Angle Overhead Camera (Locked top-down with 6-deg elevation for 3D jump altitude perception)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat = [0.0, 0.0, 0.5]
        self.camera.distance = 23.5
        self.camera.elevation = -84.0  # Top-down overview
        self.camera.azimuth = 90.0

        # Visual Trajectory & FX buffers
        self.tagger_trail: List[Tuple[float, float]] = []
        self.avoider_trail: List[Tuple[float, float]] = []
        self.shockwaves: List[List[float]] = []  # [x, y, radius, lifetime]
        self.total_tags = 0

    def world_to_screen(self, x: float, y: float, z: float = 0.0) -> Tuple[int, int]:
        """Maps 3D world coordinates to screen pixels for the top-down camera."""
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        # Scale factor for 16m container inside 720p height
        scale = 35.5
        sx = int(center_x + x * scale)
        sy = int(center_y - y * scale - z * 3.5)
        return sx, sy

    def composite_frame(self, raw_pixels: np.ndarray, dist: float, is_tagged: bool) -> np.ndarray:
        """Draws glowing 360 LiDAR lasers, motion ribbons, shockwaves, and tactical arcade HUD."""
        base_img = Image.fromarray(raw_pixels).convert("RGBA")
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        p_t = self.env.data.qpos[0:3]
        p_a = self.env.data.qpos[7:10]

        # 1. Update & Draw Motion Trails
        self.tagger_trail.append((p_t[0], p_t[1]))
        self.avoider_trail.append((p_a[0], p_a[1]))
        if len(self.tagger_trail) > 28:
            self.tagger_trail.pop(0)
        if len(self.avoider_trail) > 28:
            self.avoider_trail.pop(0)

        # Draw red predator ribbon
        for i in range(len(self.tagger_trail) - 1):
            pt1 = self.world_to_screen(self.tagger_trail[i][0], self.tagger_trail[i][1])
            pt2 = self.world_to_screen(self.tagger_trail[i + 1][0], self.tagger_trail[i + 1][1])
            alpha = int((i / len(self.tagger_trail)) * 140)
            draw.line([pt1, pt2], fill=(255, 30, 60, alpha), width=3)

        # Draw cyan prey ribbon
        for i in range(len(self.avoider_trail) - 1):
            pt1 = self.world_to_screen(self.avoider_trail[i][0], self.avoider_trail[i][1])
            pt2 = self.world_to_screen(self.avoider_trail[i + 1][0], self.avoider_trail[i + 1][1])
            alpha = int((i / len(self.avoider_trail)) * 140)
            draw.line([pt1, pt2], fill=(0, 220, 255, alpha), width=3)

        # 2. Draw Tagger 360° LiDAR Rays
        t_center = self.world_to_screen(p_t[0], p_t[1], p_t[2])
        for hx, hy, hz, is_opp in self.env.tagger_hits:
            hit_p = self.world_to_screen(hx, hy, hz)
            if is_opp:
                # GLOWING CRIMSON TARGET-LOCK LASER BEAM
                draw.line([t_center, hit_p], fill=(255, 20, 60, 240), width=4)
                draw.ellipse([hit_p[0] - 6, hit_p[1] - 6, hit_p[0] + 6, hit_p[1] + 6], fill=(255, 255, 255, 240), outline=(255, 20, 60, 255))
            else:
                draw.line([t_center, hit_p], fill=(255, 50, 80, 45), width=1)

        # 3. Draw Avoider 360° LiDAR Rays
        a_center = self.world_to_screen(p_a[0], p_a[1], p_a[2])
        for hx, hy, hz, is_opp in self.env.avoider_hits:
            hit_p = self.world_to_screen(hx, hy, hz)
            if is_opp:
                # ELECTRIC CYAN THREAT ALERT LASER BEAM
                draw.line([a_center, hit_p], fill=(0, 230, 255, 240), width=4)
                draw.ellipse([hit_p[0] - 6, hit_p[1] - 6, hit_p[0] + 6, hit_p[1] + 6], fill=(255, 255, 255, 240), outline=(0, 230, 255, 255))
            else:
                draw.line([a_center, hit_p], fill=(0, 180, 240, 40), width=1)

        # 4. Tag Impact Shockwave FX
        if is_tagged:
            self.total_tags += 1
            self.shockwaves.append([p_t[0], p_t[1], 0.4, 1.0])

        surv_shockwaves = []
        for sw in self.shockwaves:
            sw[2] += 0.35  # Expand radius
            sw[3] -= 0.08  # Fade out
            if sw[3] > 0.0:
                surv_shockwaves.append(sw)
                sw_center = self.world_to_screen(sw[0], sw[1])
                sr = int(sw[2] * 35.0)
                alpha = int(sw[3] * 220)
                draw.ellipse([sw_center[0] - sr, sw_center[1] - sr, sw_center[0] + sr, sw_center[1] + sr],
                             outline=(255, 220, 50, alpha), width=3)
        self.shockwaves = surv_shockwaves

        # 5. Top-Down Cyber Arcade HUD
        # Top Header Banner
        draw.rectangle([0, 0, self.width, 56], fill=(12, 16, 28, 225))
        draw.line([0, 56, self.width, 56], fill=(0, 220, 255, 255), width=2)

        # Tag Scoreboard
        draw.text((25, 14), "CYBER-TAG 3D: TOP-DOWN TACTICAL RADAR", fill=(0, 240, 255, 255))
        draw.text((480, 14), f"TAGS: {self.total_tags:02d}", fill=(255, 220, 40, 255))
        draw.text((620, 14), f"SEPARATION: {dist:4.2f}m", fill=(0, 255, 160, 255) if dist > 3.0 else (255, 60, 90, 255))
        draw.text((880, 14), f"TIME: {self.env.steps / 60.0:4.1f}s", fill=(200, 220, 245, 255))

        # Bottom Telemetry Dashboard
        hud_y = self.height - 48
        draw.rectangle([0, hud_y, self.width, self.height], fill=(12, 16, 28, 225))
        draw.line([0, hud_y, self.width, hud_y], fill=(0, 220, 255, 255), width=2)

        # Role Indicators & Jump Altitude
        t_jump = "[JUMPING!]" if p_t[2] > 0.8 else "[GROUND]"
        a_jump = "[JUMPING!]" if p_a[2] > 0.8 else "[GROUND]"
        draw.text((30, hud_y + 12), f"TAGGER [RED]   : {t_jump} | Z: {p_t[2]:4.2f}m", fill=(255, 50, 70, 255))
        draw.text((450, hud_y + 12), f"AVOIDER [CYAN] : {a_jump} | Z: {p_a[2]:4.2f}m", fill=(0, 220, 255, 255))
        draw.text((900, hud_y + 12), "SENSORS: PURE 360-DEGREE LIDAR (32-BEAM)", fill=(200, 225, 255, 255))

        # Flash Banner on Tag Event
        if is_tagged or len(self.shockwaves) > 0:
            draw.rectangle([self.width // 2 - 120, 70, self.width // 2 + 120, 115], fill=(255, 30, 70, 230))
            draw.text((self.width // 2 - 60, 82), "TAGGED!", fill=(255, 255, 255, 255))

        # Composite and return RGB array
        final_img = Image.alpha_composite(base_img, overlay).convert("RGB")
        return np.array(final_img, dtype=np.uint8)

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD Top-Down Tag Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

            obs_t, obs_a = self.env.reset()
            frame_count = 0
            limit = max_frames if max_frames else 600

            while frame_count < limit:
                act_t = self.tagger.forward(obs_t)
                act_a = self.avoider.forward(obs_a)

                (obs_t, obs_a), _, done = self.env.step(act_t, act_a)

                # Render Base 3D Frame from MuJoCo
                self.renderer.update_scene(self.env.data, camera=self.camera)
                raw_pixels = self.renderer.render()

                # Calculate separation distance
                p_t = self.env.data.qpos[0:3]
                p_a = self.env.data.qpos[7:10]
                dist = float(np.linalg.norm(p_t - p_a))

                # Composite top-down lasers, shockwaves & HUD
                frame = self.composite_frame(raw_pixels, dist, self.env.is_tagged)
                video_writer.append_data(frame)

                if done:
                    obs_t, obs_a = self.env.reset()

                frame_count += 1

            video_writer.close()
            print(f"[+] Top-Down 3D Tag Video saved successfully to: {video_path} ({frame_count} frames)")

        else:
            try:
                import mujoco.viewer
                print("[*] Launching MuJoCo Desktop Interactive Viewer (Top-Down)...")
                with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                    viewer.cam.elevation = -84.0
                    viewer.cam.lookat = [0, 0, 0.5]
                    viewer.cam.distance = 23.5
                    viewer.cam.azimuth = 90.0

                    obs_t, obs_a = self.env.reset()
                    while viewer.is_running():
                        step_start = time.time()

                        act_t = self.tagger.forward(obs_t)
                        act_a = self.avoider.forward(obs_a)

                        (obs_t, obs_a), _, done = self.env.step(act_t, act_a)
                        if done:
                            time.sleep(0.3)
                            obs_t, obs_a = self.env.reset()

                        viewer.sync()
                        elapsed = time.time() - step_start
                        if elapsed < 0.0166:
                            time.sleep(0.0166 - elapsed)
            except Exception as e:
                print(f"[!] Could not launch interactive GUI viewer: {e}")
                print("    You can record an HD video: python tag_arena.py --video tag_game.mp4")


# =====================================================================
# 6. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Top-Down 3D Multi-Agent Tag Game with Pure 360° LiDAR")
    parser.add_argument("--video", type=str, default=None, help="Path to save MP4 video output")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record (default: 600 = 10s)")
    parser.add_argument("--generations", type=int, default=40, help="Co-evolution training generations (default: 40)")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size per agent role (default: 32)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberTag 3D: High-Octane Top-Down Arena (Pure 360° LiDAR)     ")
    print("=================================================================")

    env = CyberTagEnv(num_lidar_rays=16)
    print("1. Constructing Tactical Box Container...")
    print("   Perspective : Top-Down Bird's-Eye Overview (-84 deg elevation)")
    print("   Sensors     : Pure 360-Degree LiDAR Array (16 Obstacle + 16 Opponent Beams)")
    print("   Tactics     : 4 Stealth Pillars (Block Line-of-Sight) + Center Hub + Low Hurdles")
    print("   Tagger      : Red Cyber-Sphere (Brain A: 32-dim LiDAR)")
    print("   Avoider     : Cyan Cyber-Sphere (Brain B: 32-dim LiDAR)")

    print(f"\n2. Co-Evolving Separate Neural Networks ({args.generations} Generations)...")
    policy_tagger = AgentPolicy(in_dim=32, out_dim=3, role="tagger")
    policy_avoider = AgentPolicy(in_dim=32, out_dim=3, role="avoider")

    trainer = CoEvolutionaryTagTrainer(env, policy_tagger, policy_avoider)
    elapsed = trainer.train_epoch(generations=args.generations, pop_size=args.pop_size, rollout_steps=320, verbose=True)
    print(f"\n   Co-Evolution Finished in {elapsed:.2f}s!")

    print("\n3. Launching Studio Top-Down Visualizer...")
    viz = TopDownStudioVisualizer(env, policy_tagger, policy_avoider)
    viz.run(video_path=args.video, max_frames=args.frames)


if __name__ == "__main__":
    main()

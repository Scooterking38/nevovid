# tag_arena.py
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import Tuple, List, Optional
import numpy as np

try:
    import mujoco
except ImportError:
    print("\n[!] 'mujoco' is not installed.")
    print("    Please install it using: pip install mujoco\n")
    sys.exit(1)


# =====================================================================
# 1. 3D ENCLOSED CONTAINER ARENA (Heavy Gravity & Stunt Platforms)
# =====================================================================
MJCF_TAG_ARENA = """
<mujoco model="cyber_tag_360">
  <compiler autolimits="true" coordinate="local"/>
  <!-- Punchy heavy gravity (-34 m/s^2) for snappy, non-floaty jumps -->
  <option gravity="0 0 -34.0" timestep="0.016666"/>

  <visual>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.30 0.30 0.40" specular="0.6 0.6 0.6"/>
    <rgba fog="0.06 0.08 0.16 1"/>
    <quality shadowsize="2048"/>
    <global elevation="-36" azimuth="135" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.14 0.28" rgb2="0.03 0.04 0.08" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.10 0.14 0.24" rgb2="0.06 0.08 0.15"/>
    <material name="grid_mat" texture="grid" texrepeat="18 18" reflectance="0.25"/>

    <material name="wall_mat" rgba="0.15 0.20 0.32 1" specular="0.4" shininess="0.3"/>
    <material name="block_mat" rgba="0.18 0.26 0.46 1" specular="0.8" shininess="0.6"/>
    <material name="tagger_mat" rgba="1.0 0.12 0.22 1" emission="0.85" specular="1" shininess="1"/>
    <material name="avoider_mat" rgba="0.0 0.95 1.0 1" emission="0.85" specular="1" shininess="1"/>
  </asset>

  <worldbody>
    <light pos="0 0 16" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.6 0.6 0.6"/>
    <light pos="0 6 10" dir="0 0 -1" diffuse="0.4 0.5 0.7"/>

    <!-- Arena Floor -->
    <geom name="floor" type="plane" size="12 12 1" pos="0 0 0" material="grid_mat"/>

    <!-- 4 Container Walls (16m x 16m Box, 4m High) -->
    <geom name="wall_north" type="box" size="8.2 0.2 2.0" pos="0 8.0 2.0" material="wall_mat"/>
    <geom name="wall_south" type="box" size="8.2 0.2 2.0" pos="0 -8.0 2.0" material="wall_mat"/>
    <geom name="wall_east"  type="box" size="0.2 8.2 2.0" pos="8.0 0 2.0" material="wall_mat"/>
    <geom name="wall_west"  type="box" size="0.2 8.2 2.0" pos="-8.0 0 2.0" material="wall_mat"/>

    <!-- Jumpable Stunt Platforms in Container (Tactical Cover) -->
    <geom name="center_block" type="box" size="1.8 1.8 0.4" pos="0 0 0.4" material="block_mat"/>
    <geom name="platform_a" type="box" size="1.5 1.5 0.75" pos="-4.5 4.5 0.75" material="block_mat"/>
    <geom name="platform_b" type="box" size="1.5 1.5 0.75" pos="4.5 -4.5 0.75" material="block_mat"/>

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

        # Precompute 360-degree unit direction vectors
        angles = np.linspace(0.0, 2 * np.pi, self.num_rays, endpoint=False)
        self.ray_dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float64)

        self.steps = 0
        self.max_steps = 350
        self.is_tagged = False

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

        # Randomize spawn positions
        angle_t = np.random.uniform(0, 2 * np.pi)
        dist_t = np.random.uniform(3.5, 6.0)
        self.data.qpos[0:3] = [dist_t * np.cos(angle_t), dist_t * np.sin(angle_t), 0.5]
        self.data.qpos[3:7] = [1, 0, 0, 0]

        angle_a = angle_t + np.pi + np.random.uniform(-0.5, 0.5)
        dist_a = np.random.uniform(3.5, 6.0)
        self.data.qpos[7:10] = [dist_a * np.cos(angle_a), dist_a * np.sin(angle_a), 0.5]
        self.data.qpos[10:14] = [1, 0, 0, 0]

        self.data.qvel[:] = 0.0
        self.is_tagged = False
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

        return self.get_observations()

    def _cast_360_lidar(self, origin: np.ndarray, own_body_id: int, opponent_geom_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fires 16 true 3D rays in a full 360-degree circular fan.
        Returns:
            - obstacle_dist (16): Normalized distance to walls and platforms [0..1]
            - opponent_dist (16): Normalized distance to opponent if spotted on ray [0..1]
        """
        geomid = np.empty(1, dtype=np.int32)
        obs_dist = np.ones(self.num_rays, dtype=np.float32)
        opp_dist = np.ones(self.num_rays, dtype=np.float32)

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

            if 0.0 <= dist <= self.max_range:
                norm_d = float(dist / self.max_range)
                if geomid[0] == opponent_geom_id:
                    opp_dist[k] = norm_d
                else:
                    obs_dist[k] = norm_d

        return obs_dist, opp_dist

    def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        PURE 360-DEGREE LIDAR ONLY (32 Features):
        - [0..15] : 16 Distance beams to static walls & blocks
        - [16..31]: 16 Detection beams spotting the opponent
        NO coordinates, NO velocities, NO cheat vectors.
        """
        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        # Tagger 360 LiDAR
        t_walls, t_opp = self._cast_360_lidar(p_tag, self.tagger_bid, self.avoider_gid)
        obs_tagger = np.concatenate([t_walls, t_opp]).astype(np.float32)

        # Avoider 360 LiDAR
        a_walls, a_opp = self._cast_360_lidar(p_avd, self.avoider_bid, self.tagger_gid)
        obs_avoider = np.concatenate([a_walls, a_opp]).astype(np.float32)

        return obs_tagger, obs_avoider

    def step(self, act_tagger: np.ndarray, act_avoider: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool]:
        self.data.qfrc_applied[:] = 0.0

        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        # High-Torque Snappy Drive (No floating momentum!)
        self.data.qfrc_applied[0] = act_tagger[0] * 38.0
        self.data.qfrc_applied[1] = act_tagger[1] * 38.0

        # Tagger Snappy Jump
        if act_tagger[2] > 0.0 and p_tag[2] < 2.0 and abs(self.data.qvel[2]) < 0.6:
            self.data.qfrc_applied[2] = 95.0

        # Avoider Snappy Drive & High Jump
        self.data.qfrc_applied[6] = act_avoider[0] * 34.0
        self.data.qfrc_applied[7] = act_avoider[1] * 34.0

        if act_avoider[2] > 0.0 and p_avd[2] < 2.0 and abs(self.data.qvel[8]) < 0.6:
            self.data.qfrc_applied[8] = 90.0

        # Kills excessive gliding momentum (0.88 decay stops on a dime)
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

        # -------------------------------------------------------------
        # REWARD SHAPING
        # -------------------------------------------------------------
        rew_tagger = -dist * 0.4 - 0.1
        rew_avoider = dist * 0.4 + 0.4

        if tagged:
            rew_tagger += 300.0
            rew_avoider -= 300.0

        done = tagged or (self.steps >= self.max_steps)
        obs_t, obs_a = self.get_observations()

        return (obs_t, obs_a), (rew_tagger, rew_avoider), done


# =====================================================================
# 3. SEPARATE 360° LIDAR POLICIES (Co-Evolutionary Networks)
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
        # Indices 0..15: Obstacle rays
        # Indices 16..31: Opponent rays
        angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        for k in range(16):
            dx, dy = np.cos(angles[k]), np.sin(angles[k])
            if role == "tagger":
                # Steer towards opponent when spotted on ray k
                self.W1[16 + k, 0] += -dx * 2.2
                self.W1[16 + k, 1] += -dy * 2.2
                # Repel away from walls
                self.W1[k, 0] += dx * 1.0
                self.W1[k, 1] += dy * 1.0
            else:
                # Flee away from opponent when spotted on ray k
                self.W1[16 + k, 0] += dx * 2.5
                self.W1[16 + k, 1] += dy * 2.5
                # Repel away from walls
                self.W1[k, 0] += dx * 1.2
                self.W1[k, 1] += dy * 1.2
                # Reflex jump when opponent is dangerously close!
                self.W1[16 + k, 2] += -1.4

        self.W2[0, 0] = 1.6
        self.W2[1, 1] = 1.6
        self.W2[2, 2] = 1.6
        self.W3[0, 0] = 1.5
        self.W3[1, 1] = 1.5
        self.W3[2, 2] = 1.5

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
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
# 5. STUDIO 3D VISUALIZER & VIDEO RECORDER
# =====================================================================
class TagStudioVisualizer:
    def __init__(self, env: CyberTagEnv, policy_tagger: AgentPolicy, policy_avoider: AgentPolicy):
        self.env = env
        self.tagger = policy_tagger
        self.avoider = policy_avoider
        self.width = 1280
        self.height = 720
        self.renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)

        # Dynamic Dual-Tracking 3D Camera
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 18.0
        self.camera.elevation = -34.0
        self.camera.azimuth = 135.0

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD 3D Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

            obs_t, obs_a = self.env.reset()
            frame_count = 0
            limit = max_frames if max_frames else 600

            while frame_count < limit:
                act_t = self.tagger.forward(obs_t)
                act_a = self.avoider.forward(obs_a)

                (obs_t, obs_a), _, done = self.env.step(act_t, act_a)

                # Center camera dynamically on midpoint of both agents
                p_t = self.env.data.qpos[0:3]
                p_a = self.env.data.qpos[7:10]
                mid_x = (p_t[0] + p_a[0]) * 0.5
                mid_y = (p_t[1] + p_a[1]) * 0.5
                mid_z = max(0.8, (p_t[2] + p_a[2]) * 0.5)

                self.camera.lookat = [mid_x, mid_y, mid_z]
                dist_sep = float(np.linalg.norm(p_t - p_a))
                self.camera.distance = max(14.0, min(24.0, dist_sep * 1.6 + 8.0))

                # Render Native Frame
                self.renderer.update_scene(self.env.data, camera=self.camera)
                pixels = self.renderer.render()
                video_writer.append_data(pixels)

                if done:
                    obs_t, obs_a = self.env.reset()

                frame_count += 1

            video_writer.close()
            print(f"[+] 3D Video saved successfully to: {video_path} ({frame_count} frames)")

        else:
            try:
                import mujoco.viewer
                print("[*] Launching MuJoCo Desktop Interactive 3D Viewer...")
                with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
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
                print("    You can generate an MP4 video using: python tag_arena.py --video tag_game.mp4")


# =====================================================================
# 6. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="3D Multi-Agent AI Tag Game with Pure 360-Degree LiDAR")
    parser.add_argument("--video", type=str, default=None, help="Path to save MP4 video output")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record (default: 600 = 10s)")
    parser.add_argument("--generations", type=int, default=40, help="Co-evolution training generations (default: 40)")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size per agent role (default: 32)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberTag 3D: Pure 360-Degree LiDAR Multi-Agent Tag Match      ")
    print("=================================================================")

    env = CyberTagEnv(num_lidar_rays=16)
    print("1. Constructing 3D Walled Container Arena...")
    print("   Physics   : Heavy Gravity (-34.0 m/s^2), Snappy Damped Drive (No Drifting)")
    print("   Sensors   : Pure 360° Circular LiDAR (16 Obstacle Beams + 16 Opponent Beams)")
    print("   Tagger    : Red Cyber-Sphere (Brain A: 32-dim LiDAR Input)")
    print("   Avoider   : Cyan Cyber-Sphere (Brain B: 32-dim LiDAR Input)")

    print(f"\n2. Co-Evolving Separate Neural Networks ({args.generations} Generations)...")
    policy_tagger = AgentPolicy(in_dim=32, out_dim=3, role="tagger")
    policy_avoider = AgentPolicy(in_dim=32, out_dim=3, role="avoider")

    trainer = CoEvolutionaryTagTrainer(env, policy_tagger, policy_avoider)
    elapsed = trainer.train_epoch(generations=args.generations, pop_size=args.pop_size, rollout_steps=320, verbose=True)
    print(f"\n   Co-Evolution Finished in {elapsed:.2f}s!")

    print("\n3. Launching 3D Visualizer...")
    viz = TagStudioVisualizer(env, policy_tagger, policy_avoider)
    viz.run(video_path=args.video, max_frames=args.frames)


if __name__ == "__main__":
    main()

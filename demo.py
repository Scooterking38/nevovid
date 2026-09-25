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
    import mujoco
    import mujoco.viewer
except ImportError:
    print("\n[!] 'mujoco' is not installed.")
    print("    Please install it using: pip install mujoco\n")
    sys.exit(1)


# =====================================================================
# 1. ACTUAL 3D MULTI-TIER CYBER ARENA IN MUJOCO (MJCF XML)
# =====================================================================
# Standard Robotics Coordinate Frame:
# +Z is UP, +Y is FORWARD (along track), +X is LATERAL (right)
# =====================================================================
MJCF_ARENA = """
<mujoco model="cyber_marble_3d">
  <compiler autolimits="true" coordinate="local"/>
  <option gravity="0 0 -14.0" timestep="0.016666"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.35" specular="0.6 0.6 0.6"/>
    <rgba fog="0.05 0.07 0.14 1"/>
    <quality shadowsize="2048"/>
    <global elevation="-22" azimuth="90"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.12 0.16 0.32" rgb2="0.04 0.05 0.10" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.10 0.14 0.24" rgb2="0.06 0.08 0.15"/>
    <material name="grid_mat" texture="grid" texrepeat="25 25" reflectance="0.2"/>

    <material name="track_mat" rgba="0.14 0.20 0.36 1" specular="0.6" shininess="0.4"/>
    <material name="ramp_mat" rgba="0.18 0.28 0.52 1" specular="0.8" shininess="0.6"/>
    <material name="deck_mat" rgba="0.12 0.32 0.48 1" specular="0.7" shininess="0.5"/>
    <material name="pillar_mat" rgba="0.15 0.18 0.28 1" specular="0.3" shininess="0.2"/>

    <material name="neon_pink" rgba="1.0 0.08 0.35 1" emission="0.8" specular="1" shininess="1"/>
    <material name="neon_cyan" rgba="0.0 0.95 1.0 1" emission="0.5" specular="1" shininess="1"/>
    <material name="neon_green" rgba="0.0 1.0 0.55 1" emission="0.9" specular="1" shininess="1"/>
    <material name="gold_crystal" rgba="1.0 0.85 0.2 1" emission="0.6" specular="1"/>
  </asset>

  <worldbody>
    <light pos="0 15 22" dir="0 0 -1" diffuse="0.85 0.85 0.85" specular="0.5 0.5 0.5"/>
    <light pos="0 5 12" dir="0 0 -1" diffuse="0.4 0.6 0.8"/>

    <!-- Bottom Chasm Floor / Safety Floor -->
    <geom name="abyss_floor" type="plane" size="40 40 1" pos="0 15 -4.0" material="grid_mat"/>

    <!-- Perimeter Blast Walls -->
    <geom name="wall_left" type="box" size="0.2 20 3.0" pos="-6.0 15 2.0" material="pillar_mat"/>
    <geom name="wall_right" type="box" size="0.2 20 3.0" pos="6.0 15 2.0" material="pillar_mat"/>

    <!-- STAGE 1: Ground Runway (Y: 0 -> 10m, Width X in [-2, 2], Z = 0.4m) -->
    <geom name="runway" type="box" size="2.0 5.0 0.2" pos="0 5.0 0.2" material="track_mat"/>

    <!-- OBSTACLE 1: High-Voltage Laser Barrier (at Y = 6.0m, Height Z: 0.4 -> 1.05m) -->
    <!-- Solid obstacle: Rolling crashes, jumping clears it! -->
    <geom name="laser_barrier" type="box" size="2.1 0.08 0.32" pos="0 6.0 0.72" material="neon_pink"/>
    <geom name="laser_post_l" type="cylinder" size="0.12 0.6" pos="-2.1 6.0 0.6" material="neon_pink"/>
    <geom name="laser_post_r" type="cylinder" size="0.12 0.6" pos="2.1 6.0 0.6" material="neon_pink"/>

    <!-- STAGE 2: Seamless Ascending Launch Ramp (Y: 10 -> 18m, Climbs Z: 0.4 -> 2.4m) -->
    <!-- Inclined by 14.036 degrees around X axis -->
    <geom name="ramp" type="box" size="2.0 4.123 0.15" pos="0 14.0 1.40" euler="14.036 0 0" material="ramp_mat"/>

    <!-- Underpass Tunnel: Open path underneath the bridge at Y=14, Z=0 -->
    <geom name="underpass_floor" type="box" size="4.0 2.0 0.1" pos="0 14.0 -0.1" material="track_mat"/>

    <!-- STAGE 3: THE VOID CHASM (Y: 18 -> 22.5m, 4.5m gap of empty space!) -->

    <!-- STAGE 4: Suspended Sky-Deck (Y: 22.5 -> 34m, Elevated at Z = 2.4m) -->
    <!-- Spans OVER the lower world: Impossible to go through underside! -->
    <geom name="sky_deck" type="box" size="2.2 5.75 0.15" pos="0 28.25 2.25" material="deck_mat"/>
    <geom name="sky_pillar_l" type="cylinder" size="0.25 1.2" pos="-2.0 28.25 1.2" material="pillar_mat"/>
    <geom name="sky_pillar_r" type="cylinder" size="0.25 1.2" pos="2.0 28.25 1.2" material="pillar_mat"/>

    <!-- STAGE 5: Summit Goal Beacon & Energy Column -->
    <geom name="beacon_pillar" type="cylinder" size="0.5 2.5" pos="0 32.5 4.9" material="neon_green"/>
    <geom name="beacon_crystal" type="sphere" size="0.7" pos="0 32.5 7.6" material="gold_crystal"/>

    <!-- THE CYBER-MARBLE: True 3D Free-Rolling Sphere Body -->
    <body name="marble" pos="0 1.5 0.8">
      <freejoint name="root"/>
      <geom name="marble_geom" type="sphere" size="0.38" mass="1.0"
            friction="1.4 0.05 0.005" solref="0.015 1.0" material="neon_cyan"/>
    </body>
  </worldbody>
</mujoco>
"""


# =====================================================================
# 2. VECTORIZED MUJOCO SIMULATOR & 14-DIM OBSERVATION ENGINE
# =====================================================================
class CyberMarbleEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF_ARENA)
        self.data = mujoco.MjData(self.model)

        self.marble_bid = self.model.body("marble").id
        self.marble_gid = self.model.geom("marble_geom").id
        self.laser_gid = self.model.geom("laser_barrier").id

        self.start_pos = np.array([0.0, 1.5, 0.8], dtype=np.float32)
        self.goal_pos = np.array([0.0, 32.5, 3.2], dtype=np.float32)

        self.cleared_hurdle = False
        self.cleared_chasm = False
        self.reached_goal = False
        self.steps = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = self.start_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Unit quaternion
        self.data.qvel[:] = 0.0
        self.cleared_hurdle = False
        self.cleared_chasm = False
        self.reached_goal = False
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        pos = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]

        # 3D Vector pointing to Goal Beacon
        to_goal = self.goal_pos - pos
        dist_goal = np.linalg.norm(to_goal)
        u_goal = to_goal / (dist_goal + 1e-6)

        # Proximity Sensors (0.0 when far, spikes to +1.0 within 1.5m takeoff window!)
        dist_laser = 6.0 - pos[1]
        laser_sensor = np.clip(1.0 - abs(dist_laser - 0.75) / 1.1, 0.0, 1.0)

        dist_chasm = 18.0 - pos[1]
        chasm_sensor = np.clip(1.0 - abs(dist_chasm - 0.75) / 1.1, 0.0, 1.0)

        # Ground contact check from MuJoCo contact manifold
        grounded = 1.0 if self.data.ncon > 0 and pos[2] > 0.0 else 0.0

        obs = np.array([
            pos[0] / 2.0,            # 0: Lateral offset from track centerline X
            pos[1] / 35.0,           # 1: Progress along track Y
            pos[2] / 4.0,            # 2: Elevation Z
            vel[0] * 0.1,            # 3: Lateral velocity
            vel[1] * 0.1,            # 4: Forward velocity
            vel[2] * 0.1,            # 5: Vertical velocity
            ang_vel[0] * 0.05,       # 6: Rolling pitch rate
            ang_vel[1] * 0.05,       # 7: Rolling roll rate
            grounded,                # 8: Ground contact flag
            laser_sensor,            # 9: Proximity trigger to Laser Hurdle
            chasm_sensor,            # 10: Proximity trigger to Chasm Takeoff
            u_goal[0], u_goal[1], u_goal[2]  # 11..13: 3D Direction to Goal Beacon
        ], dtype=np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        action: [steer, throttle, jump]
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(0.50 + 0.50 * action[1], 0.40, 1.0))
        jump = bool(action[2] > 0.0)

        pos_before = self.data.qpos[0:3].copy()

        # Apply continuous rolling torque in MuJoCo generalized coordinates
        # Torque around X rolls forward (+Y)
        # Force along X steers laterally
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[3] = -throttle * 14.0  # Forward rolling drive
        self.data.qfrc_applied[0] = steer * 18.0      # Lateral steer

        # 3D Rocket Thruster: applies vertical launch force
        is_grounded = self.data.ncon > 0 and pos_before[2] > 0.2
        if jump and is_grounded:
            self.data.qfrc_applied[2] = 52.0  # Vertical launch impulse

        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        pos_after = self.data.qpos[0:3]
        vel = self.data.qvel[0:3]

        # -------------------------------------------------------------
        # STAGE CLEARANCE DETECTION & REWARD SHAPING
        # -------------------------------------------------------------
        # 1. Forward velocity progress
        fwd_rew = vel[1] * 12.0
        center_pen = -abs(pos_after[0]) * 3.0

        # 2. Hurdle Jump Detection (Y crosses 6.0 while Z >= 1.15m)
        cleared_hurdle_now = (pos_before[1] < 6.0 <= pos_after[1]) and (pos_after[2] >= 1.15)
        if cleared_hurdle_now:
            self.cleared_hurdle = True
        laser_bounty = 2500.0 if cleared_hurdle_now else 0.0

        # Laser hurdle collision check
        hit_laser = False
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if (con.geom1 == self.marble_gid and con.geom2 == self.laser_gid) or \
               (con.geom2 == self.marble_gid and con.geom1 == self.laser_gid):
                hit_laser = True
                break
        laser_tax = 120.0 if hit_laser else 0.0

        # 3. Chasm Gap Clearance (Y crosses 22.5 while Z >= 2.3m)
        cleared_chasm_now = (pos_before[1] < 22.5 <= pos_after[1]) and (pos_after[2] >= 2.3)
        if cleared_chasm_now:
            self.cleared_chasm = True
        chasm_bounty = 5000.0 if cleared_chasm_now else 0.0

        # 4. Summit Goal Reached
        to_goal = np.linalg.norm(self.goal_pos - pos_after)
        reached_goal_now = (to_goal < 2.0) and (pos_after[2] >= 2.3)
        if reached_goal_now:
            self.reached_goal = True
        goal_bounty = 12000.0 if reached_goal_now else 0.0

        # 5. Fell into Abyss
        fell_in_void = pos_after[2] < -1.5
        fall_tax = 150.0 if fell_in_void else 0.0

        reward = fwd_rew + center_pen + laser_bounty - laser_tax + chasm_bounty + goal_bounty - fall_tax
        done = self.reached_goal or fell_in_void or (self.steps >= 360)

        obs = self.get_obs()
        return obs, reward, done


# =====================================================================
# 3. ANTITHETIC EVOLUTION STRATEGY (14-DIM POLICY)
# =====================================================================
class Fast3DPolicy:
    def __init__(self, in_dim=14, out_dim=3):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.generation = 0

        self.mW1 = np.random.randn(in_dim, 32).astype(np.float32) * 0.02
        self.mb1 = np.zeros(32, dtype=np.float32)
        self.mW2 = np.random.randn(32, 16).astype(np.float32) * 0.02
        self.mb2 = np.zeros(16, dtype=np.float32)
        self.mW3 = np.random.randn(16, out_dim).astype(np.float32) * 0.02
        self.mb3 = np.zeros(out_dim, dtype=np.float32)

        # Inductive Jump & Steering Prior Wiring
        self.mW1[0, 0] = -2.5    # Centerline tracking: offset X steers back to 0
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.5

        self.mW1[4, 1] = 1.4     # Forward speed maintenance
        self.mb1[1] = 0.8
        self.mW2[1, 1] = 1.5
        self.mW3[1, 1] = 1.4
        self.mb3[1] = 1.0        # Continuous forward torque

        # Proximity Jump Triggers
        self.mW1[9, 2] = 3.6     # Hurdle sensor (idx 9) -> JUMP OVER LASER!
        self.mW1[10, 2] = 4.2    # Chasm sensor (idx 10) -> LAUNCH ACROSS CHASM!
        self.mW1[8, 2] = 1.2     # Grounded check
        self.mb1[2] = -0.4       # Grounded when not near obstacles
        self.mW2[2, 2] = 2.0
        self.mW3[2, 2] = 1.8
        self.mb3[2] = 0.1

        self.vW1 = np.zeros_like(self.mW1)
        self.vb1 = np.zeros_like(self.mb1)
        self.vW2 = np.zeros_like(self.mW2)
        self.vb2 = np.zeros_like(self.mb2)
        self.vW3 = np.zeros_like(self.mW3)
        self.vb3 = np.zeros_like(self.mb3)

    def forward(self, obs: np.ndarray, weights=None) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = (self.mW1, self.mb1, self.mW2, self.mb2, self.mW3, self.mb3) if weights is None else weights

        h1 = np.tanh(np.dot(obs, w1) + b1)
        h2 = np.tanh(np.dot(h1, w2) + b2)
        out = np.tanh(np.dot(h2, w3) + b3)

        steer = float(out[0])
        throttle = float(np.clip(0.50 + 0.50 * out[1], 0.40, 1.0))
        jump = float(out[2])
        return np.array([steer, throttle, jump], dtype=np.float32)

    def train_es(self, env: CyberMarbleEnv, generations=60, pop_size=64, rollout_steps=320, verbose=True):
        t0 = time.perf_counter()
        half = pop_size // 2
        sigma = 0.08
        top_fit = -9999.0

        for gen in range(generations):
            eps_W1 = [np.random.randn(*self.mW1.shape).astype(np.float32) for _ in range(half)]
            eps_W2 = [np.random.randn(*self.mW2.shape).astype(np.float32) for _ in range(half)]
            eps_W3 = [np.random.randn(*self.mW3.shape).astype(np.float32) for _ in range(half)]

            fitnesses = np.zeros(pop_size, dtype=np.float32)
            max_y = 0.0
            hurdle_count = 0
            chasm_count = 0
            solved_count = 0

            for i in range(pop_size):
                sign = 1.0 if i < half else -1.0
                idx = i if i < half else i - half

                w1 = self.mW1 + sign * sigma * eps_W1[idx]
                w2 = self.mW2 + sign * sigma * eps_W2[idx]
                w3 = self.mW3 + sign * sigma * eps_W3[idx]
                weights = (w1, self.mb1, w2, self.mb2, w3, self.mb3)

                obs = env.reset()
                fit = 0.0

                for _ in range(rollout_steps):
                    act = self.forward(obs, weights=weights)
                    obs, rew, done = env.step(act)
                    fit += rew
                    max_y = max(max_y, float(env.data.qpos[1]))
                    if done:
                        break

                fit += max_y * 80.0
                fitnesses[i] = fit

                if env.cleared_hurdle:
                    hurdle_count += 1
                if env.cleared_chasm:
                    chasm_count += 1
                if env.reached_goal:
                    solved_count += 1

            # Antithetic gradient update
            fit_norm = (fitnesses - np.mean(fitnesses)) / (np.std(fitnesses) + 1e-6)
            diff = (fit_norm[:half] - fit_norm[half:])

            gW1 = np.zeros_like(self.mW1)
            gW2 = np.zeros_like(self.mW2)
            gW3 = np.zeros_like(self.mW3)
            for j in range(half):
                gW1 += diff[j] * eps_W1[j]
                gW2 += diff[j] * eps_W2[j]
                gW3 += diff[j] * eps_W3[j]
            gW1 /= half
            gW2 /= half
            gW3 /= half

            lr, beta = 0.05, 0.85
            self.vW1 = beta * self.vW1 + lr * gW1
            self.vW2 = beta * self.vW2 + lr * gW2
            self.vW3 = beta * self.vW3 + lr * gW3
            self.mW1 += self.vW1
            self.mW2 += self.vW2
            self.mW3 += self.vW3

            self.generation += 1
            top_fit = float(np.max(fitnesses))

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Max Y: {max_y:4.1f}m | Hurdle: {hurdle_count:2d}/{pop_size} | Chasm: {chasm_count:2d}/{pop_size} | Solved: {solved_count:2d}/{pop_size}")

        elapsed = time.perf_counter() - t0
        return elapsed, top_fit


# =====================================================================
# 4. HD RAYLIB 3D STUNT RENDERER & RIGHT-SIDE-UP VIDEO RECORDER
# =====================================================================
class MuJoCoStudioVisualizer:
    def __init__(self, env: CyberMarbleEnv, policy: Fast3DPolicy):
        self.env = env
        self.policy = policy

        self.screen_w = 1600
        self.screen_h = 920
        self.hud_h = 120
        self.viewport_h = self.screen_h - self.hud_h

        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_HIGHDPI | pr.FLAG_VSYNC_HINT)
        pr.init_window(self.screen_w, self.screen_h, "CyberMarble 3D: DeepMind MuJoCo Engine")
        pr.set_target_fps(60)

        # Chase Camera (Y is FORWARD, Z is UP in MuJoCo coordinate frame!)
        self.cam_smoothed = [0.0, -3.5, 2.5]
        self.camera = pr.Camera3D(
            pr.Vector3(0.0, -3.5, 2.5),
            pr.Vector3(0.0, 3.0, 0.8),
            pr.Vector3(0.0, 0.0, 1.0),  # +Z is UP
            54.0,
            pr.CAMERA_PERSPECTIVE
        )

        self.ribbons: List[List[float]] = []
        self.paused = False

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        video_writer = None
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD 3D Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

        frame_count = 0
        obs = self.env.reset()

        while not pr.window_should_close():
            if max_frames is not None and frame_count >= max_frames:
                break

            if pr.is_key_pressed(pr.KEY_SPACE):
                self.paused = not self.paused
            elif pr.is_key_pressed(pr.KEY_R):
                obs = self.env.reset()

            if not self.paused:
                action = self.policy.forward(obs)
                obs, rew, done = self.env.step(action)
                if done:
                    time.sleep(0.3)
                    obs = self.env.reset()

            pos = self.env.data.qpos[0:3]
            vel = self.env.data.qvel[0:3]
            speed = float(np.linalg.norm(vel))

            # Record trajectory ribbon
            self.ribbons.append([pos[0], pos[1], pos[2], 1.0])
            if len(self.ribbons) > 80:
                self.ribbons.pop(0)

            # Smooth 3D Chase Camera (+Z is UP)
            target_cx = pos[0]
            target_cy = pos[1] - 4.8
            target_cz = max(1.8, pos[2] + 1.8)

            self.cam_smoothed[0] += (target_cx - self.cam_smoothed[0]) * 0.22
            self.cam_smoothed[1] += (target_cy - self.cam_smoothed[1]) * 0.22
            self.cam_smoothed[2] += (target_cz - self.cam_smoothed[2]) * 0.22

            self.camera.position = pr.Vector3(float(self.cam_smoothed[0]), float(self.cam_smoothed[1]), float(self.cam_smoothed[2]))
            self.camera.target = pr.Vector3(float(pos[0]), float(pos[1] + 2.5), float(pos[2] + 0.3))
            self.camera.up = pr.Vector3(0.0, 0.0, 1.0)

            # -------------------------------------------------------------
            # 3D HARDWARE RENDERING
            # -------------------------------------------------------------
            pr.begin_drawing()
            pr.clear_background(pr.Color(12, 16, 28, 255))

            pr.begin_mode_3d(self.camera)

            # Ground Runway (Y: 0 -> 10m, Z = 0.2m)
            pr.draw_cube(pr.Vector3(0.0, 5.0, 0.2), 4.0, 10.0, 0.4, pr.Color(24, 34, 58, 255))
            pr.draw_cube_wires(pr.Vector3(0.0, 5.0, 0.2), 4.0, 10.0, 0.4, pr.Color(0, 220, 255, 255))

            # Obstacle 1: Glowing Pink Laser Barrier (Y = 6.0m)
            pr.draw_cube(pr.Vector3(0.0, 6.0, 0.72), 4.2, 0.16, 0.64, pr.Color(255, 20, 80, 220))
            pr.draw_cube_wires(pr.Vector3(0.0, 6.0, 0.72), 4.2, 0.16, 0.64, pr.Color(255, 120, 180, 255))

            # Stage 2: Ascending Ramp (Y: 10 -> 18m, Climbs Z: 0.4 -> 2.4m)
            # Center: Y = 14.0, Z = 1.4
            pr.draw_cube(pr.Vector3(0.0, 14.0, 1.40), 4.0, 8.24, 0.30, pr.Color(30, 48, 85, 255))
            pr.draw_cube_wires(pr.Vector3(0.0, 14.0, 1.40), 4.0, 8.24, 0.30, pr.Color(0, 240, 255, 255))

            # Underpass Road crossing at Y=14, Z=-0.1
            pr.draw_cube(pr.Vector3(0.0, 14.0, -0.1), 8.0, 4.0, 0.2, pr.Color(18, 26, 46, 255))

            # Chasm Void (Y: 18 -> 22.5m)
            pr.draw_plane(pr.Vector3(0.0, 20.25, -2.5), pr.Vector2(10.0, 4.5), pr.Color(6, 10, 20, 255))

            # Stage 4: Suspended Sky-Deck (Y: 22.5 -> 34m, Z = 2.25)
            pr.draw_cube(pr.Vector3(0.0, 28.25, 2.25), 4.4, 11.5, 0.30, pr.Color(20, 36, 68, 255))
            pr.draw_cube_wires(pr.Vector3(0.0, 28.25, 2.25), 4.4, 11.5, 0.30, pr.Color(0, 255, 180, 255))

            # Stage 5: Summit Goal Beacon
            pr.draw_cylinder(pr.Vector3(0.0, 32.5, 2.4), 0.5, 0.5, 5.0, 16, pr.Color(0, 255, 128, 190))
            pr.draw_sphere(pr.Vector3(0.0, 32.5, 7.6), 0.7, pr.Color(255, 220, 50, 255))
            pr.draw_sphere_wires(pr.Vector3(0.0, 32.5, 7.6), 0.75, 8, 8, pr.WHITE)

            # 3D Trajectory Ribbon Trail
            for r in self.ribbons:
                pr.draw_sphere(pr.Vector3(float(r[0]), float(r[1]), float(r[2])), 0.12, pr.Color(220, 80, 255, 220))

            # Champion AI Cyber-Marble (Cyan Core + Rolling Gyro-Ring)
            ball_p = pr.Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
            ball_color = pr.Color(0, 245, 255, 255) if self.env.data.ncon > 0 else pr.Color(230, 90, 255, 255)
            pr.draw_sphere(ball_p, 0.38, ball_color)
            pr.draw_sphere_wires(ball_p, 0.38, 12, 12, pr.WHITE)

            # Rocket Thruster Flame when Airborne
            if pos[2] > 0.6 and self.env.data.ncon == 0:
                pr.draw_sphere(pr.Vector3(float(pos[0]), float(pos[1]) - 0.35, float(pos[2]) - 0.35), 0.22, pr.ORANGE)
                pr.draw_sphere(pr.Vector3(float(pos[0]), float(pos[1]) - 0.35, float(pos[2]) - 0.35), 0.12, pr.YELLOW)

            pr.end_mode_3d()

            # -------------------------------------------------------------
            # 2D HUD OVERLAYS
            # -------------------------------------------------------------
            hud_y = self.viewport_h
            pr.draw_rectangle(0, hud_y, self.screen_w, self.hud_h, pr.Color(14, 20, 36, 255))
            pr.draw_line(0, hud_y, self.screen_w, hud_y, pr.Color(0, 220, 255, 255))

            dist_beacon = float(np.linalg.norm(pos - self.env.goal_pos))
            pr.draw_text(f"GEN {self.policy.generation:03d} [DeepMind MuJoCo]", 30, hud_y + 16, 26, pr.Color(0, 220, 255, 255))
            pr.draw_text(f"FORWARD SPEED : {speed * 3.6:4.1f} km/h", 30, hud_y + 48, 18, pr.WHITE)
            pr.draw_text(f"ELEVATION Z   : {pos[2]:4.2f} m", 30, hud_y + 70, 18, pr.Color(230, 140, 255, 255))
            pr.draw_text(f"DIST TO BEACON: {dist_beacon:4.1f} m", 30, hud_y + 92, 18, pr.LIME)

            # Telemetry Milestones
            pr.draw_line(420, hud_y + 12, 420, hud_y + 110, pr.Color(45, 65, 100, 255))
            pr.draw_text("STAGE CLEARANCES", 445, hud_y + 16, 20, pr.Color(200, 225, 255, 255))

            h_col = pr.LIME if self.env.cleared_hurdle else pr.YELLOW
            pr.draw_text("1. LASER HURDLE : [CLEARED]" if self.env.cleared_hurdle else "1. LASER HURDLE : [PENDING]", 445, hud_y + 44, 15, h_col)

            c_col = pr.LIME if self.env.cleared_chasm else pr.YELLOW
            pr.draw_text("2. CHASM 4.5m   : [CLEARED]" if self.env.cleared_chasm else "2. CHASM 4.5m   : [PENDING]", 445, hud_y + 66, 15, c_col)

            g_col = pr.LIME if self.env.reached_goal else pr.SKYBLUE
            pr.draw_text("3. SUMMIT BEACON: [VICTORY]" if self.env.reached_goal else "3. SUMMIT BEACON: [RACING]", 445, hud_y + 88, 15, g_col)

            # Actuators
            pr.draw_line(820, hud_y + 12, 820, hud_y + 110, pr.Color(45, 65, 100, 255))
            pr.draw_text("AI 3D MOTORS", 845, hud_y + 16, 20, pr.Color(200, 225, 255, 255))
            pr.draw_text(f"TORQUE X: {float(action[0]):+.2f} | TORQUE Y: {float(action[1]):.2f}", 845, hud_y + 48, 16, pr.WHITE)

            thruster_on = bool(action[2] > 0.0)
            t_col = pr.Color(230, 140, 255, 255) if thruster_on else pr.GRAY
            pr.draw_text("VERTICAL ROCKET THRUSTER: [ACTIVE]" if thruster_on else "VERTICAL ROCKET THRUSTER: [OFF]", 845, hud_y + 74, 16, t_col)

            pr.end_drawing()

            # Record Video: Native top-down orientation (Right-side up)
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
    parser = argparse.ArgumentParser(description="CyberMarble 3D: DeepMind MuJoCo Engine")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution training generations (default: 60)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberMarble 3D: DeepMind MuJoCo Physics & Stunt Course        ")
    print("=================================================================")

    env = CyberMarbleEnv()
    print("1. Initializing DeepMind MuJoCo 3D World (Zero Seam Glitches)...")
    print(f"   Launch Pad    : Y = 0.0m -> 10.0m (Ground level)")
    print(f"   Laser Hurdle  : Y = 6.0m (Solid hurdle; jumping required!)")
    print(f"   Ascending Ramp: Y = 10.0m -> 18.0m (14.0 deg continuous incline)")
    print(f"   The Void Chasm: Y = 18.0m -> 22.5m (4.5m aerial leap across abyss)")
    print(f"   Suspended Deck: Y = 22.5m -> 34.0m (Crosses OVER lower road at Z=2.4m)")
    print(f"   Summit Beacon : Y = 32.5m (Elevated goal platform)")

    print(f"\n2. Evolving 64 Agents via Antithetic ES ({args.generations} Generations)...")
    policy = Fast3DPolicy(in_dim=14, out_dim=3)
    elapsed, top_fit = policy.train_es(env, generations=args.generations, pop_size=64, rollout_steps=320, verbose=True)
    print(f"\n   Training Finished in {elapsed:.2f}s! Top Fitness: {top_fit:.1f}")

    print("\n3. Launching Studio-Grade 3D Engine...")
    print("   Controls: [SPACE] Pause | [R] Reset to Launchpad\n")

    max_frames = args.frames if args.video else None
    viz = MuJoCoStudioVisualizer(env, policy)
    viz.run(video_path=args.video, max_frames=max_frames)


if __name__ == "__main__":
    main()

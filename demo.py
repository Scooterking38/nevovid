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
    import pybullet as p
except ImportError:
    print("\n[!] 'pybullet' is not installed.")
    print("    Please install it using: pip install pybullet\n")
    sys.exit(1)


# =====================================================================
# 1. ACTUAL 3D MULTI-TIER CYBER ARENA BUILDER
# =====================================================================
class CyberArena3D:
    def __init__(self):
        self.width = 34.0
        self.depth = 28.0
        self.height = 7.0

        # Start and Summit Goal positions
        self.start_pos = np.array([3.0, 3.5, 0.40], dtype=np.float32)
        self.goal_pos = np.array([27.0, 21.5, 3.65], dtype=np.float32)

    def build_world(self):
        """Constructs solid, perfectly aligned 3D platforms, ramps, and overpasses."""
        # 1. Ground Slab (Tier 0: Z = 0)
        plane_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[22.0, 22.0, 0.1])
        plane_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[22.0, 22.0, 0.1], rgbaColor=[0.08, 0.10, 0.16, 1.0])
        p.createMultiBody(0, plane_col, plane_vis, [17.0, 14.0, -0.1])

        # 2. Outer Perimeter Blast Walls
        self._add_box([17.0, 0.5, 2.5], [17.0, 0.5, 2.5], [0.15, 0.20, 0.32, 1.0])
        self._add_box([17.0, 27.5, 2.5], [17.0, 0.5, 2.5], [0.15, 0.20, 0.32, 1.0])
        self._add_box([0.5, 14.0, 2.5], [0.5, 14.0, 2.5], [0.15, 0.20, 0.32, 1.0])
        self._add_box([33.5, 14.0, 2.5], [0.5, 14.0, 2.5], [0.15, 0.20, 0.32, 1.0])

        # 3. Ground Runway (X: 2 -> 10, Y: 3.5)
        self._add_box([6.0, 3.5, 0.1], [4.0, 2.5, 0.1], [0.12, 0.18, 0.28, 1.0])

        # 4. Ascending Ramp 1 (Climbs X: 10 -> 18 at Y: 3.5, Z: 0.2 -> 2.4m)
        ramp1_len = math.sqrt(8.0**2 + 2.2**2)
        pitch1 = math.atan2(2.2, 8.0)
        ramp1_orn = p.getQuaternionFromEuler([0, -pitch1, 0])
        self._add_box([14.0, 3.5, 1.3], [ramp1_len / 2.0, 2.5, 0.1], [0.14, 0.38, 0.70, 1.0], orn=ramp1_orn)

        # 5. Tier 1 Skyway Banked Sweeper (X: 18 -> 22, Y: 3.5 -> 7.0, smoothly connects East to South!)
        self._add_box([19.5, 4.0, 2.3], [2.5, 2.5, 0.1], [0.10, 0.45, 0.85, 1.0])
        self._add_box([21.0, 6.5, 2.3], [2.5, 2.5, 0.1], [0.10, 0.45, 0.85, 1.0])

        # 6. Tier 1 Highway Bridge (Z = 2.4m, spans South from Y: 7.0 to 14.0, crosses OVER lower tunnel!)
        self._add_box([21.5, 10.5, 2.3], [2.5, 4.0, 0.1], [0.10, 0.45, 0.85, 1.0])

        # Overpass Cross-Beam Arch
        self._add_box([21.5, 10.0, 2.45], [2.6, 0.4, 0.2], [0.8, 0.2, 0.9, 1.0])

        # 7. Launch Kicker Ramp at Chasm Edge (Angled UP by +16 deg to propel car across void!)
        kicker_len = math.sqrt(1.8**2 + 0.48**2)
        kicker_pitch = math.atan2(0.48, 1.8)
        k_orn = p.getQuaternionFromEuler([kicker_pitch, 0, 0])
        self._add_box([21.5, 15.2, 2.45], [2.5, kicker_len / 2.0, 0.08], [1.0, 0.35, 0.1, 1.0], orn=k_orn)

        # 8. Receiving Sky-Platform Across Chasm (Z = 2.5m, gap of 2.8m from Y: 16.0 to Y: 18.8)
        self._add_box([21.5, 20.0, 2.4], [2.5, 2.0, 0.1], [0.10, 0.55, 0.85, 1.0])

        # 9. Ascending Ramp 2 to Summit (Climbs X: 23.5 -> 26.5 at Y: 21, Z: 2.5 -> 3.6m)
        r2_len = math.sqrt(3.0**2 + 1.1**2)
        pitch2 = math.atan2(1.1, 3.0)
        r2_orn = p.getQuaternionFromEuler([0, -pitch2, 0])
        self._add_box([24.5, 20.8, 3.05], [r2_len / 2.0, 2.2, 0.08], [0.0, 0.65, 0.85, 1.0], orn=r2_orn)

        # 10. Tier 2 Summit Deck (Z = 3.6m, Goal Beacon Island!)
        self._add_box([28.0, 21.0, 3.5], [2.5, 2.5, 0.1], [0.0, 0.80, 0.65, 1.0])

        # 11. Hazard Pillars in Lower Canyon
        self._add_box([8.0, 14.0, 1.5], [1.2, 3.5, 1.5], [0.20, 0.25, 0.40, 1.0])
        self._add_box([15.0, 16.0, 1.5], [1.2, 3.5, 1.5], [0.20, 0.25, 0.40, 1.0])

        # 12. 3D Glowing Emerald Goal Beacon
        beacon_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.6, height=4.5)
        beacon_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.6, length=4.5, rgbaColor=[0.0, 1.0, 0.5, 0.7])
        p.createMultiBody(0, beacon_col, beacon_vis, [float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]) + 2.25])

        # Goal Top Crystal
        c_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.8, rgbaColor=[0.3, 1.0, 0.7, 1.0])
        p.createMultiBody(0, -1, c_vis, [float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]) + 4.8])

    def _add_box(self, pos, half_extents, rgba, orn=[0, 0, 0, 1]):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        body = p.createMultiBody(0, col, vis, pos, orn)
        p.changeDynamics(body, -1, lateralFriction=1.6, restitution=0.05)
        return body


# =====================================================================
# 2. 4-WHEEL RAYCAST SUSPENSION & AERODYNAMICS VEHICLE
# =====================================================================
class RaycastSuspensionCar3D:
    def __init__(self, start_pos: np.ndarray):
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.mass = 150.0  # kg

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.70, 0.42, 0.15])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.70, 0.42, 0.15], rgbaColor=[0.0, 0.95, 1.0, 1.0])
        self.body = p.createMultiBody(self.mass, col, vis, self.start_pos.tolist())

        p.changeDynamics(self.body, -1, lateralFriction=1.4, spinningFriction=0.1, rollingFriction=0.01, restitution=0.05)

        # Cockpit Canopy
        c_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.38, 0.28, 0.12], rgbaColor=[0.85, 0.25, 1.0, 1.0])
        p.createMultiBody(0.1, -1, c_vis, [float(self.start_pos[0]) - 0.05, float(self.start_pos[1]), float(self.start_pos[2]) + 0.24])

        # 4 Wheel Mount Offsets [X_fwd, Y_lat, Z_up]
        self.wheel_mounts = np.array([
            [0.55, -0.44, 0.0],
            [0.55, 0.44, 0.0],
            [-0.55, -0.44, 0.0],
            [-0.55, 0.44, 0.0],
        ], dtype=np.float32)

        self.susp_rest_len = 0.36
        self.spring_k = 4800.0
        self.damper_c = 440.0
        self.tire_friction = 1.8

        self.last_jump_step = -999
        self.airborne_steps = 0

    def reset(self):
        p.resetBasePositionAndOrientation(self.body, self.start_pos.tolist(), p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0])
        self.last_jump_step = -999
        self.airborne_steps = 0

    def get_pos_and_heading(self) -> Tuple[np.ndarray, np.ndarray, float]:
        pos, orn = p.getBasePositionAndOrientation(self.body)
        euler = p.getEulerFromQuaternion(orn)
        heading = euler[2]
        return np.array(pos, dtype=np.float32), np.array(euler, dtype=np.float32), heading

    def apply_control(self, steer: float, throttle: float, jump: bool, step_count: int = 0):
        pos, euler, heading = self.get_pos_and_heading()
        lin_vel, ang_vel = p.getBaseVelocity(self.body)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)

        orn = p.getBasePositionAndOrientation(self.body)[1]
        rot_mat = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)

        fwd_vec = rot_mat[:, 0]
        lat_vec = rot_mat[:, 1]
        up_vec = rot_mat[:, 2]

        fwd_speed = float(np.dot(lin_vel, fwd_vec))

        # -------------------------------------------------------------
        # 1. 4-WHEEL RAYCAST SUSPENSION & TIRE DYNAMICS
        # -------------------------------------------------------------
        ray_starts = []
        ray_ends = []
        steer_angle = float(-steer * math.radians(32))

        for mount in self.wheel_mounts:
            w_world = pos + rot_mat.dot(mount)
            ray_starts.append(w_world.tolist())
            ray_ends.append((w_world - up_vec * 0.45).tolist())

        results = p.rayTestBatch(ray_starts, ray_ends)
        wheels_grounded = 0

        for i, res in enumerate(results):
            hit_fraction = res[2]
            if hit_fraction < 1.0:
                wheels_grounded += 1
                hit_pos = np.array(res[3], dtype=np.float32)
                hit_normal = np.array(res[4], dtype=np.float32)

                dist = hit_fraction * 0.45
                compression = max(0.0, self.susp_rest_len - dist)

                v_normal = float(np.dot(lin_vel, hit_normal))
                f_susp = max(0.0, self.spring_k * compression - self.damper_c * v_normal)

                p.applyExternalForce(self.body, -1, (hit_normal * f_susp).tolist(), hit_pos.tolist(), p.WORLD_FRAME)

                is_front = (i < 2)
                if is_front:
                    cos_s, sin_s = math.cos(steer_angle), math.sin(steer_angle)
                    w_fwd = fwd_vec * cos_s + lat_vec * sin_s
                    w_lat = -fwd_vec * sin_s + lat_vec * cos_s
                else:
                    w_fwd = fwd_vec
                    w_lat = lat_vec

                # Longitudinal motor drive force (applied at rear wheels)
                if not is_front:
                    f_drive = w_fwd * (throttle * 1250.0)
                    p.applyExternalForce(self.body, -1, f_drive.tolist(), hit_pos.tolist(), p.WORLD_FRAME)

                # Lateral tire grip (slip resistance proportional to normal load)
                v_lat_slip = float(np.dot(lin_vel, w_lat))
                f_lat_mag = -np.clip(v_lat_slip * 240.0, -f_susp * self.tire_friction, f_susp * self.tire_friction)
                f_lateral = w_lat * f_lat_mag
                p.applyExternalForce(self.body, -1, f_lateral.tolist(), hit_pos.tolist(), p.WORLD_FRAME)

        # -------------------------------------------------------------
        # 2. AERODYNAMICS & MID-AIR STABILIZATION
        # -------------------------------------------------------------
        downforce = up_vec * (-float(fwd_speed**2) * 2.0)
        p.applyExternalForce(self.body, -1, downforce.tolist(), pos.tolist(), p.WORLD_FRAME)

        if wheels_grounded == 0:
            self.airborne_steps += 1
            roll_err = float(euler[0])
            pitch_err = float(euler[1] - math.atan2(lin_vel[2], max(1.0, fwd_speed)) * 0.4)
            tau_roll = -roll_err * 140.0 - ang_vel[0] * 40.0
            tau_pitch = -pitch_err * 140.0 - ang_vel[1] * 40.0
            tau_yaw = -steer * 500.0

            gyro_torque = rot_mat.dot([tau_roll, tau_pitch, tau_yaw])
            p.applyExternalTorque(self.body, -1, gyro_torque.tolist(), p.WORLD_FRAME)
        else:
            self.airborne_steps = 0

        # -------------------------------------------------------------
        # 3. 3D ROCKET THRUSTER LAUNCH (Chasm Leaps)
        # -------------------------------------------------------------
        can_jump = (step_count - self.last_jump_step) > 35
        on_deck = (wheels_grounded >= 2) and (pos[2] < 4.2)

        if jump and on_deck and can_jump and fwd_speed > 2.0:
            target_vz = float(max(5.4, lin_vel[2] + 5.4))
            p.resetBaseVelocity(self.body, linearVelocity=[float(lin_vel[0]), float(lin_vel[1]), target_vz])
            self.last_jump_step = step_count

    def query_3d_lidar(self) -> np.ndarray:
        """Fires 9 true 3D rays querying the Bullet BVH spatial acceleration tree."""
        pos, _, heading = self.get_pos_and_heading()
        ch, sh = math.cos(heading), math.sin(heading)

        from_pts = [pos.tolist()] * 9
        to_pts = []

        # 5 Horizontal forward rays [-50°, -25°, 0°, +25°, +50°]
        angles = heading + np.linspace(-math.radians(50), math.radians(50), 5)
        for a in angles:
            to_pts.append([float(pos[0] + math.cos(a) * 8.0), float(pos[1] + math.sin(a) * 8.0), float(pos[2])])

        # 2 Downward chasm sensing rays
        to_pts.append([float(pos[0] + ch * 2.5), float(pos[1] + sh * 2.5), float(pos[2] - 1.8)])
        to_pts.append([float(pos[0] + ch * 4.5), float(pos[1] + sh * 4.5), float(pos[2] - 2.8)])

        # 2 Upward ceiling rays
        to_pts.append([float(pos[0] + ch * 2.0), float(pos[1] + sh * 2.0), float(pos[2] + 1.8)])
        to_pts.append([float(pos[0] + ch * 4.0), float(pos[1] + sh * 4.0), float(pos[2] + 2.5)])

        results = p.rayTestBatch(from_pts, to_pts)
        hit_fractions = np.array([r[2] for r in results], dtype=np.float32)
        return hit_fractions


# =====================================================================
# 3. ANTITHETIC EVOLUTION STRATEGY (20-DIM 3D AGENT)
# =====================================================================
class Fast3DNeuroPolicy:
    def __init__(self, in_dim=20, out_dim=3):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mW1 = np.random.randn(in_dim, 36).astype(np.float32) * 0.02
        self.mb1 = np.zeros(36, dtype=np.float32)
        self.mW2 = np.random.randn(36, 18).astype(np.float32) * 0.02
        self.mb2 = np.zeros(18, dtype=np.float32)
        self.mW3 = np.random.randn(18, out_dim).astype(np.float32) * 0.02
        self.mb3 = np.zeros(out_dim, dtype=np.float32)

        # Navigation Prior Wiring:
        self.mW1[10, 0] = 2.4     # ego_goal_y -> steer
        self.mW1[0, 0] = -0.6     # left ray repulsion
        self.mW1[4, 0] = 0.6      # right ray repulsion
        self.mW2[0, 0] = 1.8
        self.mW3[0, 0] = 1.5

        # Speed commitment
        self.mW1[9, 1] = 1.6      # ego_goal_x -> throttle
        self.mW1[2, 1] = 1.2      # center clearance
        self.mW2[1, 1] = 1.5
        self.mW3[1, 1] = 1.2
        self.mb3[1] = 1.2

        # 3D Chasm Launch Thrusters
        self.mW1[5, 2] = -2.8     # chasm drop-off detected ahead -> JUMP!
        self.mW1[11, 2] = 2.4     # goal is elevated above (Tier 1/2) -> climb!
        self.mW2[2, 2] = 1.8
        self.mW3[2, 2] = 1.6
        self.mb3[2] = 0.15

        self.vW1 = np.zeros_like(self.mW1)
        self.vb1 = np.zeros_like(self.mb1)
        self.vW2 = np.zeros_like(self.mW2)
        self.vb2 = np.zeros_like(self.mb2)
        self.vW3 = np.zeros_like(self.mW3)
        self.vb3 = np.zeros_like(self.mb3)

        self.generation = 0
        self.last_hidden = np.zeros(18, dtype=np.float32)

    def forward(self, obs: np.ndarray, weights=None) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = (self.mW1, self.mb1, self.mW2, self.mb2, self.mW3, self.mb3) if weights is None else weights

        h1 = np.tanh(np.dot(obs, w1) + b1)
        h2 = np.tanh(np.dot(h1, w2) + b2)
        self.last_hidden = h2
        raw = np.tanh(np.dot(h2, w3) + b3)

        steer = float(raw[0])
        throttle = float(np.clip(0.65 + 0.35 * raw[1], 0.45, 1.0))
        jump = bool(raw[2] > 0.0)
        return np.array([steer, throttle, 1.0 if jump else 0.0], dtype=np.float32)

    def train_es_epoch(self, arena: CyberArena3D, car: RaycastSuspensionCar3D, generations=60, pop_size=64, rollout_steps=340, verbose=True):
        t0 = time.perf_counter()
        half = pop_size // 2
        sigma = 0.08
        top_fit = -9999.0

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        for gen in range(generations):
            eps_W1 = [np.random.randn(*self.mW1.shape).astype(np.float32) for _ in range(half)]
            eps_W2 = [np.random.randn(*self.mW2.shape).astype(np.float32) for _ in range(half)]
            eps_W3 = [np.random.randn(*self.mW3.shape).astype(np.float32) for _ in range(half)]

            fitnesses = np.zeros(pop_size, dtype=np.float32)
            closest_dist = 999.0
            solved_count = 0
            max_altitude = 0.0

            for i in range(pop_size):
                sign = 1.0 if i < half else -1.0
                idx = i if i < half else i - half

                w1 = self.mW1 + sign * sigma * eps_W1[idx]
                w2 = self.mW2 + sign * sigma * eps_W2[idx]
                w3 = self.mW3 + sign * sigma * eps_W3[idx]
                weights = (w1, self.mb1, w2, self.mb2, w3, self.mb3)

                car.reset()
                fit = 0.0
                init_dist = float(np.linalg.norm(car.start_pos - arena.goal_pos))
                min_d = init_dist

                for step in range(rollout_steps):
                    pos, euler, heading = car.get_pos_and_heading()
                    lin_vel, _ = p.getBaseVelocity(car.body)
                    max_altitude = max(max_altitude, float(pos[2]))

                    lidar_9 = car.query_3d_lidar()
                    to_goal = arena.goal_pos - pos
                    d_3d = float(np.linalg.norm(to_goal))
                    min_d = min(min_d, d_3d)

                    u_goal = to_goal / (d_3d + 1e-6)
                    ch, sh = math.cos(heading), math.sin(heading)
                    ego_gx = ch * u_goal[0] + sh * u_goal[1]
                    ego_gy = -sh * u_goal[0] + ch * u_goal[1]
                    ego_gz = u_goal[2]

                    ego_vx = ch * lin_vel[0] + sh * lin_vel[1]
                    ego_vy = -sh * lin_vel[0] + ch * lin_vel[1]

                    # 20-dimensional full state observation
                    obs = np.concatenate([
                        lidar_9,
                        [ego_gx, ego_gy, ego_gz, d_3d * 0.03, ego_vx, ego_vy, lin_vel[2], pos[2] * 0.25, euler[0], euler[1], 1.0 if pos[2] < 0.2 else 0.0]
                    ]).astype(np.float32)

                    act = self.forward(obs, weights=weights)
                    car.apply_control(act[0], act[1], bool(act[2] > 0.5), step_count=step)
                    p.stepSimulation()

                    fwd_prog = lin_vel[0] * u_goal[0] + lin_vel[1] * u_goal[1] + lin_vel[2] * u_goal[2] * 2.0
                    fit += fwd_prog * 12.0 + (lin_vel[0]**2 + lin_vel[1]**2)**0.5 * 4.0

                    # Reached Summit Sky-Deck Goal
                    if d_3d < 2.5 and pos[2] >= 3.0:
                        fit += 12000.0 + (rollout_steps - step) * 30.0
                        solved_count += 1
                        break

                    # Fell in Chasm
                    if pos[2] < -1.5:
                        fit -= 100.0
                        break

                fit += max(0.0, init_dist - min_d) * 100.0
                fitnesses[i] = fit
                closest_dist = min(closest_dist, min_d)

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

            lr, beta = 0.04, 0.85
            self.vW1 = beta * self.vW1 + lr * gW1
            self.vW2 = beta * self.vW2 + lr * gW2
            self.vW3 = beta * self.vW3 + lr * gW3
            self.mW1 += self.vW1
            self.mW2 += self.vW2
            self.mW3 += self.vW3

            self.generation += 1
            top_fit = float(np.max(fitnesses))
            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Top Fit: {top_fit:8.1f} | Closest: {closest_dist:4.1f}m | Solved: {solved_count:2d}/{pop_size} | Max Alt: {max_altitude:4.2f}m")

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        elapsed = time.perf_counter() - t0
        return elapsed, top_fit


# =====================================================================
# 4. HARDWARE OPENGL 3D CHASE VISUALIZER & VIDEO EXPORTER
# =====================================================================
class PyBulletStudioVisualizer:
    def __init__(self, arena: CyberArena3D, car: RaycastSuspensionCar3D, policy: Fast3DNeuroPolicy):
        self.arena = arena
        self.car = car
        self.policy = policy
        self.cam_mode = 0

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        video_writer = None
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD 3D Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

        self.car.reset()
        frame = 0

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        hud_id = -1
        p.addUserDebugText(
            "[C] Cycle Camera Mode | [R] Reset | [Q/ESC] Quit",
            [1.0, 1.0, 5.5],
            textColorRGB=[0.0, 0.9, 1.0],
            textSize=1.2
        )

        while p.isConnected():
            if max_frames is not None and frame >= max_frames:
                break

            keys = p.getKeyboardEvents()
            if ord('c') in keys and (keys[ord('c')] & p.KEY_WAS_TRIGGERED):
                self.cam_mode = (self.cam_mode + 1) % 2
            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                self.car.reset()
            if (ord('q') in keys and (keys[ord('q')] & p.KEY_WAS_TRIGGERED)) or (27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED)):
                break

            pos, euler, heading = self.car.get_pos_and_heading()
            lin_vel, _ = p.getBaseVelocity(self.car.body)
            speed = float(np.linalg.norm(lin_vel))

            # Query 9-beam 3D Lidar
            lidar_9 = self.car.query_3d_lidar()
            to_goal = self.arena.goal_pos - pos
            d_3d = float(np.linalg.norm(to_goal))

            u_goal = to_goal / (d_3d + 1e-6)
            ch, sh = math.cos(heading), math.sin(heading)
            ego_gx = ch * u_goal[0] + sh * u_goal[1]
            ego_gy = -sh * u_goal[0] + ch * u_goal[1]
            ego_gz = u_goal[2]
            ego_vx = ch * lin_vel[0] + sh * lin_vel[1]
            ego_vy = -sh * lin_vel[0] + ch * lin_vel[1]

            obs = np.concatenate([
                lidar_9,
                [ego_gx, ego_gy, ego_gz, d_3d * 0.03, ego_vx, ego_vy, lin_vel[2], pos[2] * 0.25, euler[0], euler[1], 1.0 if pos[2] < 0.2 else 0.0]
            ]).astype(np.float32)

            act = self.policy.forward(obs)
            is_jumping = bool(act[2] > 0.5)
            self.car.apply_control(act[0], act[1], is_jumping, step_count=frame)
            p.stepSimulation()

            # Dynamic 3D Chase Camera
            if self.cam_mode == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.8,
                    cameraYaw=-math.degrees(heading) - 90,
                    cameraPitch=-20,
                    cameraTargetPosition=[float(pos[0]), float(pos[1]), float(pos[2]) + 0.5]
                )
            else:
                p.resetDebugVisualizerCamera(
                    cameraDistance=26.0,
                    cameraYaw=45,
                    cameraPitch=-42,
                    cameraTargetPosition=[16.0, 13.0, 2.0]
                )

            # Auto-Reset upon Reaching Summit Goal or Falling in Chasm
            if (d_3d < 2.5 and pos[2] >= 3.0) or pos[2] < -1.5:
                time.sleep(0.4)
                self.car.reset()

            # Live 3D Floating Telemetry HUD
            telemetry = f"SPEED: {speed*3.6:4.1f} km/h | ALT: {pos[2]:4.2f}m | DIST: {d_3d:4.1f}m"
            if is_jumping:
                telemetry += " [THRUSTERS ACTIVE]"
            hud_id = p.addUserDebugText(
                telemetry,
                [pos[0], pos[1], pos[2] + 0.9],
                textColorRGB=[0.2, 1.0, 0.4] if not is_jumping else [1.0, 0.3, 0.8],
                textSize=1.3,
                replaceItemUniqueId=hud_id
            )

            # Video Frame Grab
            if video_writer is not None:
                view_mat = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[float(pos[0]), float(pos[1]), float(pos[2]) + 0.5],
                    distance=4.8,
                    yaw=-math.degrees(heading) - 90,
                    pitch=-20,
                    roll=0,
                    upAxisIndex=2
                )
                proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=1280 / 720, nearVal=0.1, farVal=100.0)
                _, _, rgb, _, _ = p.getCameraImage(1280, 720, view_mat, proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                video_writer.append_data(rgb[:, :, :3])

            frame += 1
            time.sleep(1.0 / 60.0)

        if video_writer is not None:
            video_writer.close()
            print(f"[+] Video saved successfully to {video_path}")


# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CyberArena: Actual 3D Spatial Labyrinth in PyBullet")
    parser.add_argument("--video", type=str, default=None, help="Save MP4 recording to path")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record when --video is set (default: 600)")
    parser.add_argument("--generations", type=int, default=60, help="Evolution generations (default: 60)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberArena: True 3D Multi-Tier Labyrinth in PyBullet Engine   ")
    print("=================================================================")

    connection_mode = p.DIRECT if args.headless else p.GUI
    p.connect(connection_mode)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 60.0)

    print("1. Constructing 3-Tier Multi-Level World (Ramps, Skyway, Chasm)...")
    arena = CyberArena3D()
    arena.build_world()
    car = RaycastSuspensionCar3D(arena.start_pos)

    print(f"2. Evolving 3D Policy via Antithetic ES ({args.generations} Generations)...")
    policy = Fast3DNeuroPolicy(in_dim=20, out_dim=3)
    elapsed, top_fit = policy.train_es_epoch(arena, car, generations=args.generations, pop_size=64, rollout_steps=340, verbose=True)
    print(f"\n   Training Finished in {elapsed:.2f}s! Top Fitness: {top_fit:.1f}")

    print("\n3. Launching 3D Studio Visualizer...")
    print("   Controls: [C] Toggle View (Chase / Aerial) | [R] Reset | [Q/ESC] Quit\n")

    max_frames = args.frames if args.video else None
    viz = PyBulletStudioVisualizer(arena, car, policy)
    viz.run(video_path=args.video, max_frames=max_frames)

    p.disconnect()


if __name__ == "__main__":
    main()

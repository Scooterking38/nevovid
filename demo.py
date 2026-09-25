# tag_arena.py
from __future__ import annotations

import os
import sys
import time
import math
import random
import argparse
from typing import Tuple, List, Dict, Optional, Set
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
<mujoco model="cyber_tag_neat">
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
# 2. NEAT (NeuroEvolution of Augmenting Topologies) CORE ENGINE
# =====================================================================
class ConnectionGene:
    __slots__ = ["in_node", "out_node", "weight", "enabled", "innovation", "is_recurrent"]

    def __init__(self, in_node: int, out_node: int, weight: float, enabled: bool, innovation: int, is_recurrent: bool = False):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
        self.is_recurrent = is_recurrent

    def copy(self) -> ConnectionGene:
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation, self.is_recurrent)


class Genome:
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connections: Dict[int, ConnectionGene] = {}  # innovation -> ConnectionGene
        self.hidden_nodes: Set[int] = set()
        self.fitness: float = 0.0

    def copy(self) -> Genome:
        g = Genome(self.in_dim, self.out_dim)
        g.connections = {k: v.copy() for k, v in self.connections.items()}
        g.hidden_nodes = set(self.hidden_nodes)
        g.fitness = self.fitness
        return g


class NEATInnovations:
    def __init__(self):
        self.current_innovation = 0
        self.connection_history: Dict[Tuple[int, int], int] = {}
        self.next_node_id = 0

    def get_innovation(self, in_node: int, out_node: int) -> int:
        pair = (in_node, out_node)
        if pair not in self.connection_history:
            self.current_innovation += 1
            self.connection_history[pair] = self.current_innovation
        return self.connection_history[pair]

    def get_new_node_id(self) -> int:
        self.next_node_id += 1
        return self.next_node_id


class PhenotypeNetwork:
    """Fast recurrent neural network compiler from a NEAT genome."""
    def __init__(self, genome: Genome):
        self.in_dim = genome.in_dim
        self.out_dim = genome.out_dim
        self.all_nodes = sorted(list(range(self.in_dim)) + list(range(self.in_dim, self.in_dim + self.out_dim)) + list(genome.hidden_nodes))
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.all_nodes)}
        self.num_nodes = len(self.all_nodes)

        # Connection matrices for 1-step feedforward & recurrent memory
        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for conn in genome.connections.values():
            if conn.enabled:
                src = self.node_to_idx[conn.in_node]
                dst = self.node_to_idx[conn.out_node]
                self.weights[src, dst] = conn.weight

        self.values = np.zeros(self.num_nodes, dtype=np.float32)
        self.prev_values = np.zeros(self.num_nodes, dtype=np.float32)

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        # Load input sensors
        self.values[:self.in_dim] = inputs

        # 2 passes of relaxation to propagate forward signals and recurrent loops
        for _ in range(2):
            self.prev_values[:] = self.values
            # Hidden & Output activation: tanh(sum(incoming))
            for i in range(self.in_dim, self.num_nodes):
                incoming = np.dot(self.prev_values, self.weights[:, i])
                self.values[i] = np.tanh(incoming)

        # Output actuators
        out_start = self.in_dim
        out_end = self.in_dim + self.out_dim
        return self.values[out_start:out_end]


class NEATPopulation:
    def __init__(self, size: int, in_dim: int, out_dim: int, tracker: NEATInnovations, role: str = "tagger"):
        self.size = size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tracker = tracker
        self.role = role
        self.tracker.next_node_id = max(self.tracker.next_node_id, in_dim + out_dim)

        self.population: List[Genome] = []
        for _ in range(size):
            g = Genome(in_dim, out_dim)
            self._init_minimal_genome(g)
            self.population.append(g)

        self.species: List[List[Genome]] = []

    def _init_minimal_genome(self, g: Genome):
        """Minimal initial connectivity: directly connect sensory rays to drive/jump."""
        angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        for k in range(16):
            dx, dy = np.cos(angles[k]), np.sin(angles[k])
            # Connection from opponent detection beam to Drive X (output 0) and Drive Y (output 1)
            in_opp = 16 + k
            in_obs = k
            out_x = self.in_dim + 0
            out_y = self.in_dim + 1
            out_jump = self.in_dim + 2

            if self.role == "tagger":
                # Chase opponent, avoid walls
                self._add_conn(g, in_opp, out_x, float(-dx * 1.8))
                self._add_conn(g, in_opp, out_y, float(-dy * 1.8))
                self._add_conn(g, in_obs, out_x, float(dx * 0.8))
                self._add_conn(g, in_obs, out_y, float(dy * 0.8))
            else:
                # Flee opponent, avoid walls, jump reflex when cornered
                self._add_conn(g, in_opp, out_x, float(dx * 2.0))
                self._add_conn(g, in_opp, out_y, float(dy * 2.0))
                self._add_conn(g, in_obs, out_x, float(dx * 1.0))
                self._add_conn(g, in_obs, out_y, float(dy * 1.0))
                self._add_conn(g, in_opp, out_jump, -1.2)

    def _add_conn(self, g: Genome, in_n: int, out_n: int, w: float, is_rec: bool = False):
        innov = self.tracker.get_innovation(in_n, out_n)
        g.connections[innov] = ConnectionGene(in_n, out_n, w, True, innov, is_rec)

    def mutate(self, g: Genome):
        # 1. Weight Mutations (80% fine tune, 10% random reset)
        for conn in g.connections.values():
            if np.random.rand() < 0.80:
                if np.random.rand() < 0.90:
                    conn.weight += float(np.random.randn() * 0.12)
                else:
                    conn.weight = float(np.random.randn() * 0.5)

        # 2. Add Connection Mutation (5% chance: connect two previously unconnected nodes)
        if np.random.rand() < 0.15:
            all_possible_nodes = list(range(self.in_dim)) + list(g.hidden_nodes)
            all_target_nodes = list(range(self.in_dim, self.in_dim + self.out_dim)) + list(g.hidden_nodes)

            src = random.choice(all_possible_nodes)
            dst = random.choice(all_target_nodes)

            # Check if connection already exists
            exists = any(c.in_node == src and c.out_node == dst for c in g.connections.values())
            if not exists:
                is_recurrent = (src in g.hidden_nodes and dst in g.hidden_nodes and src >= dst)
                self._add_conn(g, src, dst, float(np.random.randn() * 0.5), is_recurrent)

        # 3. Add Node Mutation (3% chance: split an enabled connection to sprout a new hidden neuron)
        if np.random.rand() < 0.08 and len(g.connections) > 0:
            enabled_conns = [c for c in g.connections.values() if c.enabled]
            if enabled_conns:
                conn_to_split = random.choice(enabled_conns)
                conn_to_split.enabled = False

                new_node = self.tracker.get_new_node_id()
                g.hidden_nodes.add(new_node)

                # in -> new_node (weight 1.0)
                self._add_conn(g, conn_to_split.in_node, new_node, 1.0)
                # new_node -> out (weight = old weight)
                self._add_conn(g, new_node, conn_to_split.out_node, conn_to_split.weight)

    def speciate_and_reproduce(self):
        # Sort by fitness descending
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        elites = [g.copy() for g in self.population[:4]]

        new_pop = []
        new_pop.extend(elites)

        # Truncation selection from top 30%
        top_pool = self.population[:max(4, int(self.size * 0.3))]

        while len(new_pop) < self.size:
            parent = random.choice(top_pool).copy()
            self.mutate(parent)
            new_pop.append(parent)

        self.population = new_pop


# =====================================================================
# 3. PURE 360° LIDAR ENVIRONMENT WITH LINE-OF-SIGHT OCCLUSION
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
        self.max_range = 16.0

        angles = np.linspace(0.0, 2 * np.pi, self.num_rays, endpoint=False)
        self.ray_dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float64)

        self.steps = 0
        self.max_steps = 350
        self.is_tagged = False

        self.tagger_hits: List[Tuple[float, float, float, bool]] = []
        self.avoider_hits: List[Tuple[float, float, float, bool]] = []

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)

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
        """Pure 360-degree LiDAR: 16 obstacle beams + 16 opponent beams = 32 inputs."""
        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        t_walls, t_opp, self.tagger_hits = self._cast_360_lidar(p_tag, self.tagger_bid, self.avoider_gid)
        obs_tagger = np.concatenate([t_walls, t_opp]).astype(np.float32)

        a_walls, a_opp, self.avoider_hits = self._cast_360_lidar(p_avd, self.avoider_bid, self.tagger_gid)
        obs_avoider = np.concatenate([a_walls, a_opp]).astype(np.float32)

        return obs_tagger, obs_avoider

    def step(self, act_tagger: np.ndarray, act_avoider: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool]:
        self.data.qfrc_applied[:] = 0.0

        p_tag = self.data.qpos[0:3]
        p_avd = self.data.qpos[7:10]

        # Tagger Forces
        self.data.qfrc_applied[0] = act_tagger[0] * 38.0
        self.data.qfrc_applied[1] = act_tagger[1] * 38.0
        if act_tagger[2] > 0.0 and p_tag[2] < 2.0 and abs(self.data.qvel[2]) < 0.6:
            self.data.qfrc_applied[2] = 95.0

        # Avoider Forces
        self.data.qfrc_applied[6] = act_avoider[0] * 34.0
        self.data.qfrc_applied[7] = act_avoider[1] * 34.0
        if act_avoider[2] > 0.0 and p_avd[2] < 2.0 and abs(self.data.qvel[8]) < 0.6:
            self.data.qfrc_applied[8] = 90.0

        # High damping to eliminate ice-skating momentum
        self.data.qvel[0:2] *= 0.88
        self.data.qvel[3:5] *= 0.88
        self.data.qvel[6:8] *= 0.88
        self.data.qvel[9:11] *= 0.88

        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        p_tag_after = self.data.qpos[0:3]
        p_avd_after = self.data.qpos[7:10]
        dist = np.linalg.norm(p_tag_after - p_avd_after)

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
# 4. ADVERSARIAL NEAT CO-EVOLUTION TRAINER
# =====================================================================
class NEATCoEvolutionTrainer:
    def __init__(self, env: CyberTagEnv, pop_tagger: NEATPopulation, pop_avoider: NEATPopulation):
        self.env = env
        self.pop_tagger = pop_tagger
        self.pop_avoider = pop_avoider

    def train_epoch(self, generations: int = 40, rollout_steps: int = 320, verbose: bool = True):
        t0 = time.perf_counter()

        for gen in range(generations):
            tags_count = 0

            # Round-robin competitive rollouts
            for i in range(self.pop_tagger.size):
                tagger_genome = self.pop_tagger.population[i]
                avoider_genome = self.pop_avoider.population[i]

                net_t = PhenotypeNetwork(tagger_genome)
                net_a = PhenotypeNetwork(avoider_genome)

                obs_t, obs_a = self.env.reset()
                fit_t, fit_a = 0.0, 0.0

                for _ in range(rollout_steps):
                    act_t = net_t.activate(obs_t)
                    act_a = net_a.activate(obs_a)

                    (obs_t, obs_a), (r_t, r_a), done = self.env.step(act_t, act_a)
                    fit_t += r_t
                    fit_a += r_a

                    if done:
                        if self.env.is_tagged:
                            tags_count += 1
                        break

                tagger_genome.fitness = fit_t
                avoider_genome.fitness = fit_a

            # Compute topology statistics
            avg_nodes_t = np.mean([len(g.hidden_nodes) for g in self.pop_tagger.population])
            avg_conn_t = np.mean([sum(1 for c in g.connections.values() if c.enabled) for g in self.pop_tagger.population])
            avg_nodes_a = np.mean([len(g.hidden_nodes) for g in self.pop_avoider.population])
            avg_conn_a = np.mean([sum(1 for c in g.connections.values() if c.enabled) for g in self.pop_avoider.population])

            if verbose and (gen % 5 == 0 or gen == generations - 1):
                print(f"   Gen {gen:02d}/{generations} | Tags: {tags_count:2d}/{self.pop_tagger.size} | "
                      f"Tagger Neurons: {avg_nodes_t:.1f} (Synapses: {avg_conn_t:.1f}) | "
                      f"Avoider Neurons: {avg_nodes_a:.1f} (Synapses: {avg_conn_a:.1f})")

            # Speciation & structural reproduction
            self.pop_tagger.speciate_and_reproduce()
            self.pop_avoider.speciate_and_reproduce()

        elapsed = time.perf_counter() - t0
        return elapsed


# =====================================================================
# 5. STUDIO TOP-DOWN VISUALIZER & DYNAMIC BRAIN GRAPH COMPOSITOR
# =====================================================================
class NEATStudioVisualizer:
    def __init__(self, env: CyberTagEnv, champion_tagger: Genome, champion_avoider: Genome):
        self.env = env
        self.net_t = PhenotypeNetwork(champion_tagger)
        self.net_a = PhenotypeNetwork(champion_avoider)
        self.genome_t = champion_tagger
        self.genome_a = champion_avoider

        self.width = 1280
        self.height = 720
        self.renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)

        # Overhead Camera locked directly top-down
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat = [0.0, 0.0, 0.5]
        self.camera.distance = 23.5
        self.camera.elevation = -84.0
        self.camera.azimuth = 90.0

        self.tagger_trail: List[Tuple[float, float]] = []
        self.avoider_trail: List[Tuple[float, float]] = []
        self.shockwaves: List[List[float]] = []
        self.total_tags = 0

    def world_to_screen(self, x: float, y: float, z: float = 0.0) -> Tuple[int, int]:
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        scale = 35.5
        sx = int(center_x + x * scale)
        sy = int(center_y - y * scale - z * 3.5)
        return sx, sy

    def draw_neat_brain_graph(self, draw: ImageDraw.ImageDraw, genome: Genome, net: PhenotypeNetwork, ox: int, oy: int, title: str, color_theme: Tuple[int, int, int]):
        """Draws the live evolved NEAT neural graph with active pulsing synapses on the HUD."""
        bw, bh = 220, 110
        draw.rectangle([ox, oy, ox + bw, oy + bh], fill=(16, 22, 38, 220), outline=color_theme, width=2)
        draw.text((ox + 8, oy + 6), title, fill=color_theme)
        draw.text((ox + 8, oy + 22), f"Hidden: {len(genome.hidden_nodes)} | Synapses: {len(genome.connections)}", fill=(180, 200, 230, 255))

        # Node coordinates in micro-graph
        node_pos: Dict[int, Tuple[int, int]] = {}
        # Inputs (left column)
        for i in range(min(8, genome.in_dim)):
            node_pos[i] = (ox + 16, oy + 42 + i * 8)
        # Outputs (right column)
        for i in range(genome.out_dim):
            node_pos[genome.in_dim + i] = (ox + bw - 20, oy + 50 + i * 20)
        # Hidden nodes (center cloud)
        for idx, hid in enumerate(sorted(list(genome.hidden_nodes))[:6]):
            hx = ox + 60 + (idx % 3) * 45
            hy = oy + 48 + (idx // 3) * 26
            node_pos[hid] = (hx, hy)

        # Draw synaptic connections
        for conn in genome.connections.values():
            if conn.enabled and conn.in_node in node_pos and conn.out_node in node_pos:
                p1 = node_pos[conn.in_node]
                p2 = node_pos[conn.out_node]
                w_color = color_theme if conn.weight > 0 else (255, 60, 60, 180)
                draw.line([p1, p2], fill=w_color, width=1)

        # Draw nodes
        for nid, (nx, ny) in node_pos.items():
            nc = (100, 255, 150, 255) if nid < genome.in_dim else (color_theme if nid < genome.in_dim + genome.out_dim else (255, 230, 80, 255))
            draw.ellipse([nx - 3, ny - 3, nx + 3, ny + 3], fill=nc)

    def composite_frame(self, raw_pixels: np.ndarray, dist: float, is_tagged: bool) -> np.ndarray:
        base_img = Image.fromarray(raw_pixels).convert("RGBA")
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        p_t = self.env.data.qpos[0:3]
        p_a = self.env.data.qpos[7:10]

        # 1. Update Motion Trails
        self.tagger_trail.append((p_t[0], p_t[1]))
        self.avoider_trail.append((p_a[0], p_a[1]))
        if len(self.tagger_trail) > 28:
            self.tagger_trail.pop(0)
        if len(self.avoider_trail) > 28:
            self.avoider_trail.pop(0)

        # Draw predator red trail
        for i in range(len(self.tagger_trail) - 1):
            pt1 = self.world_to_screen(self.tagger_trail[i][0], self.tagger_trail[i][1])
            pt2 = self.world_to_screen(self.tagger_trail[i + 1][0], self.tagger_trail[i + 1][1])
            alpha = int((i / len(self.tagger_trail)) * 150)
            draw.line([pt1, pt2], fill=(255, 30, 60, alpha), width=3)

        # Draw prey cyan trail
        for i in range(len(self.avoider_trail) - 1):
            pt1 = self.world_to_screen(self.avoider_trail[i][0], self.avoider_trail[i][1])
            pt2 = self.world_to_screen(self.avoider_trail[i + 1][0], self.avoider_trail[i + 1][1])
            alpha = int((i / len(self.avoider_trail)) * 150)
            draw.line([pt1, pt2], fill=(0, 220, 255, alpha), width=3)

        # 2. Draw Tagger 360° LiDAR Scan
        t_center = self.world_to_screen(p_t[0], p_t[1], p_t[2])
        for hx, hy, hz, is_opp in self.env.tagger_hits:
            hit_p = self.world_to_screen(hx, hy, hz)
            if is_opp:
                # Crimson Target Lock Laser
                draw.line([t_center, hit_p], fill=(255, 20, 60, 240), width=4)
                draw.ellipse([hit_p[0] - 6, hit_p[1] - 6, hit_p[0] + 6, hit_p[1] + 6], fill=(255, 255, 255, 240), outline=(255, 20, 60, 255))
            else:
                draw.line([t_center, hit_p], fill=(255, 50, 80, 45), width=1)

        # 3. Draw Avoider 360° LiDAR Scan
        a_center = self.world_to_screen(p_a[0], p_a[1], p_a[2])
        for hx, hy, hz, is_opp in self.env.avoider_hits:
            hit_p = self.world_to_screen(hx, hy, hz)
            if is_opp:
                # Electric Cyan Threat Alert Laser
                draw.line([a_center, hit_p], fill=(0, 230, 255, 240), width=4)
                draw.ellipse([hit_p[0] - 6, hit_p[1] - 6, hit_p[0] + 6, hit_p[1] + 6], fill=(255, 255, 255, 240), outline=(0, 230, 255, 255))
            else:
                draw.line([a_center, hit_p], fill=(0, 180, 240, 40), width=1)

        # 4. Shockwave FX on Tag Event
        if is_tagged:
            self.total_tags += 1
            self.shockwaves.append([p_t[0], p_t[1], 0.4, 1.0])

        surv_shockwaves = []
        for sw in self.shockwaves:
            sw[2] += 0.35
            sw[3] -= 0.08
            if sw[3] > 0.0:
                surv_shockwaves.append(sw)
                sw_center = self.world_to_screen(sw[0], sw[1])
                sr = int(sw[2] * 35.0)
                alpha = int(sw[3] * 220)
                draw.ellipse([sw_center[0] - sr, sw_center[1] - sr, sw_center[0] + sr, sw_center[1] + sr],
                             outline=(255, 220, 50, alpha), width=3)
        self.shockwaves = surv_shockwaves

        # 5. Draw Live Evolved NEAT Brain Graphs
        self.draw_neat_brain_graph(draw, self.genome_t, self.net_t, 25, 75, "TAGGER NEAT BRAIN", (255, 50, 70))
        self.draw_neat_brain_graph(draw, self.genome_a, self.net_a, self.width - 245, 75, "AVOIDER NEAT BRAIN", (0, 220, 255))

        # 6. Top-Down Sci-Fi Arcade Header
        draw.rectangle([0, 0, self.width, 56], fill=(12, 16, 28, 230))
        draw.line([0, 56, self.width, 56], fill=(0, 220, 255, 255), width=2)

        draw.text((25, 14), "NEAT RECURRENT STEALTH TAG [MUJOCO 3D]", fill=(0, 240, 255, 255))
        draw.text((540, 14), f"TAGS: {self.total_tags:02d}", fill=(255, 220, 40, 255))
        draw.text((680, 14), f"SEPARATION: {dist:4.2f}m", fill=(0, 255, 160, 255) if dist > 3.0 else (255, 60, 90, 255))
        draw.text((940, 14), f"ROUND TIME: {self.env.steps / 60.0:4.1f}s", fill=(200, 220, 245, 255))

        # Bottom Telemetry Dashboard
        hud_y = self.height - 48
        draw.rectangle([0, hud_y, self.width, self.height], fill=(12, 16, 28, 230))
        draw.line([0, hud_y, self.width, hud_y], fill=(0, 220, 255, 255), width=2)

        t_jump = "[JUMPING!]" if p_t[2] > 0.8 else "[GROUND]"
        a_jump = "[JUMPING!]" if p_a[2] > 0.8 else "[GROUND]"
        draw.text((30, hud_y + 12), f"TAGGER [RED]: {t_jump} | Z: {p_t[2]:4.2f}m", fill=(255, 50, 70, 255))
        draw.text((450, hud_y + 12), f"AVOIDER [CYAN]: {a_jump} | Z: {p_a[2]:4.2f}m", fill=(0, 220, 255, 255))
        draw.text((900, hud_y + 12), "STEALTH: CORNER PILLARS BREAK 360 LIDAR", fill=(200, 225, 255, 255))

        if is_tagged or len(self.shockwaves) > 0:
            draw.rectangle([self.width // 2 - 120, 70, self.width // 2 + 120, 115], fill=(255, 30, 70, 230))
            draw.text((self.width // 2 - 60, 82), "TAGGED!", fill=(255, 255, 255, 255))

        final_img = Image.alpha_composite(base_img, overlay).convert("RGB")
        return np.array(final_img, dtype=np.uint8)

    def run(self, video_path: Optional[str] = None, max_frames: Optional[int] = None):
        if video_path:
            import imageio
            print(f"[*] Recording 60 FPS HD Top-Down NEAT Tag Video to: {video_path}")
            video_writer = imageio.get_writer(video_path, fps=60, codec="libx264", quality=8)

            obs_t, obs_a = self.env.reset()
            frame_count = 0
            limit = max_frames if max_frames else 600

            while frame_count < limit:
                act_t = self.net_t.activate(obs_t)
                act_a = self.net_a.activate(obs_a)

                (obs_t, obs_a), _, done = self.env.step(act_t, act_a)

                self.renderer.update_scene(self.env.data, camera=self.camera)
                raw_pixels = self.renderer.render()

                p_t = self.env.data.qpos[0:3]
                p_a = self.env.data.qpos[7:10]
                dist = float(np.linalg.norm(p_t - p_a))

                frame = self.composite_frame(raw_pixels, dist, self.env.is_tagged)
                video_writer.append_data(frame)

                if done:
                    obs_t, obs_a = self.env.reset()

                frame_count += 1

            video_writer.close()
            print(f"[+] 3D NEAT Video saved successfully to: {video_path} ({frame_count} frames)")

        else:
            try:
                import mujoco.viewer
                print("[*] Launching MuJoCo Interactive Desktop Viewer (Top-Down)...")
                with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                    viewer.cam.elevation = -84.0
                    viewer.cam.lookat = [0, 0, 0.5]
                    viewer.cam.distance = 23.5
                    viewer.cam.azimuth = 90.0

                    obs_t, obs_a = self.env.reset()
                    while viewer.is_running():
                        step_start = time.time()

                        act_t = self.net_t.activate(obs_t)
                        act_a = self.net_a.activate(obs_a)

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
                print("    You can record an HD video: python tag_arena.py --video tag_neat.mp4")


# =====================================================================
# 6. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Top-Down 3D Multi-Agent Tag Game with NEAT Recurrent Brains")
    parser.add_argument("--video", type=str, default=None, help="Path to save MP4 video output")
    parser.add_argument("--frames", type=int, default=600, help="Frames to record (default: 600 = 10s)")
    parser.add_argument("--generations", type=int, default=30, help="NEAT co-evolution generations (default: 30)")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size per agent role (default: 32)")
    args = parser.parse_args()

    print("=================================================================")
    print("   CyberTag 3D: NEAT Recurrent Brains + Stealth LiDAR Arena      ")
    print("=================================================================")

    env = CyberTagEnv(num_lidar_rays=16)
    print("1. Initializing 3D Box Container with Line-of-Sight Occlusion...")
    print("   Enclosure : 16m x 16m Container, 4m Blast Walls")
    print("   Stealth   : 4 Massive Sentry Pillars physically block 360° LiDAR")
    print("   Tactics   : Center Hub (0.9m) + Low Hurdles (Jump over to escape!)")

    print(f"\n2. Evolving Augmented Topologies via NEAT ({args.generations} Generations)...")
    innovations = NEATInnovations()
    pop_tagger = NEATPopulation(args.pop_size, in_dim=32, out_dim=3, tracker=innovations, role="tagger")
    pop_avoider = NEATPopulation(args.pop_size, in_dim=32, out_dim=3, tracker=innovations, role="avoider")

    trainer = NEATCoEvolutionTrainer(env, pop_tagger, pop_avoider)
    elapsed = trainer.train_epoch(generations=args.generations, rollout_steps=320, verbose=True)
    print(f"\n   NEAT Co-Evolution Finished in {elapsed:.2f}s!")

    champion_tagger = pop_tagger.population[0]
    champion_avoider = pop_avoider.population[0]
    print(f"   Champion Tagger Architecture : {len(champion_tagger.hidden_nodes)} Hidden Nodes, {len(champion_tagger.connections)} Synapses")
    print(f"   Champion Avoider Architecture: {len(champion_avoider.hidden_nodes)} Hidden Nodes, {len(champion_avoider.connections)} Synapses")

    print("\n3. Launching Studio Top-Down Visualizer...")
    viz = NEATStudioVisualizer(env, champion_tagger, champion_avoider)
    viz.run(video_path=args.video, max_frames=args.frames)


if __name__ == "__main__":
    main()

# nevorl.py
from __future__ import annotations

from typing import Any, Dict, Tuple
from collections import deque
import math
import numpy as np
from lark import Lark, Transformer

# =====================================================================
# 1. GRAMMAR DEFINITION
# =====================================================================
NEVORL_GRAMMAR = r"""
    start: env_def

    env_def: "env" CNAME "{" section* "}"

    ?section: state_sec
            | action_sec
            | obs_sec
            | reset_sec
            | step_sec
            | reward_sec
            | terminal_sec

    state_sec: "state" "{" field_decl* "}"
    field_decl: CNAME ":" TYPE ";"?

    action_sec: "action" action_type ";"?
    action_type: "discrete" "(" INT ")"       -> act_discrete
               | "continuous" "(" INT ")"     -> act_continuous

    obs_sec: "observation" "{" stmt* "return" expr ";"? "}"
    reset_sec: "reset" "{" reset_stmt* "}"
    step_sec: "step" "(" CNAME ")" "{" stmt* "}"
    reward_sec: "reward" "{" stmt* "return" expr ";"? "}"
    terminal_sec: "terminal" "{" stmt* "return" expr ";"? "}"

    reset_stmt: target "=" expr ";"?           -> reset_assign

    ?stmt: let_stmt
         | assign_stmt

    let_stmt: "let" CNAME "=" expr ";"?
    assign_stmt: target "=" expr ";"?

    dot_path: CNAME ("." CNAME)+

    target: dot_path -> dot_target
          | CNAME    -> var_target

    ?expr: or_expr
    ?or_expr: and_expr (LOGIC_OR and_expr)*
    ?and_expr: comp_expr (LOGIC_AND comp_expr)*
    ?comp_expr: arith_expr (COMP_OP arith_expr)*
    ?arith_expr: term (ADD_OP term)*
    ?term: factor (MUL_OP factor)*
    ?factor: ADD_OP? atom
    ?atom: SIGNED_NUMBER                     -> num
         | dot_path                          -> dot_access
         | CNAME "[" expr "]"                -> array_index
         | CNAME "(" (expr ("," expr)*)? ")" -> func_call
         | CNAME                             -> var
         | "[" (expr ("," expr)*)? "]"       -> list_lit
         | "(" expr ")"

    LOGIC_OR.2: "or" | "||"
    LOGIC_AND.2: "and" | "&&"
    TYPE.2: "float" | "int" | "bool" | "vec2"

    COMP_OP: ">=" | "<=" | "==" | "!=" | ">" | "<"
    ADD_OP: "+" | "-"
    MUL_OP: "*" | "/"

    %import common.CNAME
    %import common.INT
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
    %ignore /\/\/[^\n]*/
"""

# =====================================================================
# 2. RUNTIME ENGINE (Defensive BFS, Swept CCD & Jump Kinematics)
# =====================================================================
_BFS_CACHE: Dict[Tuple[int, int, bytes], Tuple[np.ndarray, np.ndarray]] = {}

def get_bfs_maps(target_pt: Tuple[int, int], grid_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    key = (target_pt[0], target_pt[1], grid_map.tobytes())
    if key in _BFS_CACHE:
        return _BFS_CACHE[key]

    H, W = grid_map.shape
    tx, ty = int(target_pt[0]), int(target_pt[1])
    dist_map = np.full((H, W), 9999.0, dtype=np.float32)

    if 0 <= tx < W and 0 <= ty < H and grid_map[ty, tx] == 0:
        dist_map[ty, tx] = 0.0
        q = deque([(tx, ty)])
        while q:
            cx, cy = q.popleft()
            d = dist_map[cy, cx]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if grid_map[ny, nx] == 0 and dist_map[ny, nx] > d + 1.0:
                        dist_map[ny, nx] = d + 1.0
                        q.append((nx, ny))

    dir_map = np.zeros((H, W, 2), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if grid_map[y, x] == 1 or dist_map[y, x] >= 9000.0:
                dx = float(tx - x)
                dy = float(ty - y)
                norm = math.hypot(dx, dy) + 1e-6
                dir_map[y, x] = [dx / norm, dy / norm]
                continue

            best_dir = (0.0, 0.0)
            best_d = dist_map[y, x]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and grid_map[ny, nx] == 0:
                    if dist_map[ny, nx] < best_d:
                        best_d = dist_map[ny, nx]
                        best_dir = (float(dx), float(dy))
            norm = (best_dir[0]**2 + best_dir[1]**2)**0.5
            if norm > 0:
                dir_map[y, x] = [best_dir[0] / norm, best_dir[1] / norm]

    _BFS_CACHE[key] = (dist_map, dir_map)
    return dist_map, dir_map

def get_bfs_dist(pos: np.ndarray, target: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
    tx = int(target[0, 0]) if target.ndim == 2 else int(target[0])
    ty = int(target[0, 1]) if target.ndim == 2 else int(target[1])
    dist_map, _ = get_bfs_maps((tx, ty), grid_map)
    H, W = grid_map.shape
    ix = np.clip(pos[:, 0].astype(np.int32), 0, W - 1)
    iy = np.clip(pos[:, 1].astype(np.int32), 0, H - 1)
    raw_dist = dist_map[iy, ix]
    euc_dist = np.linalg.norm(pos - target, axis=-1) + 20.0
    return np.where(raw_dist >= 9000.0, euc_dist, raw_dist).astype(np.float32)

def get_bfs_vector(pos: np.ndarray, target: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
    tx = int(target[0, 0]) if target.ndim == 2 else int(target[0])
    ty = int(target[0, 1]) if target.ndim == 2 else int(target[1])
    _, dir_map = get_bfs_maps((tx, ty), grid_map)
    H, W = grid_map.shape
    ix = np.clip(pos[:, 0].astype(np.int32), 0, W - 1)
    iy = np.clip(pos[:, 1].astype(np.int32), 0, H - 1)
    return dir_map[iy, ix]

def fast_raycast_fan(pos: np.ndarray, heading: np.ndarray, fov: float, count: int, max_dist: float, grid_map: np.ndarray) -> np.ndarray:
    H, W = grid_map.shape
    angles = heading[:, None] + np.linspace(-fov / 2.0, fov / 2.0, count)[None, :]
    dx = np.cos(angles)
    dy = np.sin(angles)

    steps = 32
    dists = np.linspace(0.05, max_dist, steps)
    sample_x = pos[:, 0, None, None] + dx[:, :, None] * dists[None, None, :]
    sample_y = pos[:, 1, None, None] + dy[:, :, None] * dists[None, None, :]

    ix = np.clip(sample_x.astype(np.int32), 0, W - 1)
    iy = np.clip(sample_y.astype(np.int32), 0, H - 1)

    hits = grid_map[iy, ix] == 1
    has_hit = np.any(hits, axis=2)
    first_hit = np.argmax(hits, axis=2)

    hit_dists = np.where(has_hit, dists[first_hit], max_dist)
    return (hit_dists / max_dist).astype(np.float32)

def check_circle_aabb_collision(pos: np.ndarray, radius: float, grid_map: np.ndarray) -> np.ndarray:
    H, W = grid_map.shape
    x, y = pos[:, 0], pos[:, 1]
    r = radius

    oob = (x - r < 1.0) | (x + r > W - 1.0) | (y - r < 1.0) | (y + r > H - 1.0)

    min_x = np.clip(np.floor(x - r).astype(np.int32), 0, W - 1)
    max_x = np.clip(np.floor(x + r).astype(np.int32), 0, W - 1)
    min_y = np.clip(np.floor(y - r).astype(np.int32), 0, H - 1)
    max_y = np.clip(np.floor(y + r).astype(np.int32), 0, H - 1)

    cand_cells = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
    ]

    r_sq = r * r
    any_hit = oob.copy()

    for cx, cy in cand_cells:
        is_wall = grid_map[cy, cx] == 1
        closest_x = np.clip(x, cx.astype(np.float32), (cx + 1).astype(np.float32))
        closest_y = np.clip(y, cy.astype(np.float32), (cy + 1).astype(np.float32))
        dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
        hit_box = is_wall & (dist_sq < r_sq)
        any_hit |= hit_box

    return any_hit

def resolve_wall_slide(pos: np.ndarray, vel: np.ndarray, radius: float, elasticity: float, grid_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(vel, axis=-1)
    max_step_dist = 0.15
    substeps = int(np.clip(np.ceil(np.max(speed) / max_step_dist), 1, 6))

    cur_pos = pos.copy()
    cur_vel = vel.copy()
    sub_vel = cur_vel / substeps
    any_hit = np.zeros(pos.shape[0], dtype=bool)

    for _ in range(substeps):
        cand_x = cur_pos[:, 0] + sub_vel[:, 0]
        hit_x = check_circle_aabb_collision(np.column_stack([cand_x, cur_pos[:, 1]]), radius, grid_map)
        cur_pos[:, 0] = np.where(hit_x, cur_pos[:, 0], cand_x)
        cur_vel[:, 0] = np.where(hit_x, -cur_vel[:, 0] * elasticity, cur_vel[:, 0])

        cand_y = cur_pos[:, 1] + sub_vel[:, 1]
        hit_y = check_circle_aabb_collision(np.column_stack([cur_pos[:, 0], cand_y]), radius, grid_map)
        cur_pos[:, 1] = np.where(hit_y, cur_pos[:, 1], cand_y)
        cur_vel[:, 1] = np.where(hit_y, -cur_vel[:, 1] * elasticity, cur_vel[:, 1])

        sub_vel = cur_vel / substeps
        any_hit |= hit_x | hit_y

    return cur_pos, cur_vel, any_hit

def apply_car_kinematics(state: Dict[str, np.ndarray], steer: np.ndarray, throttle: np.ndarray,
                         traction: float, momentum: float, steer_rate: float, angular_drag: float,
                         elasticity: float, radius: float, altitude: np.ndarray, grid_map: np.ndarray) -> np.ndarray:
    N = state['pos'].shape[0]

    # Angular dynamics
    ang_vel = state.get('ang_vel', np.zeros(N, dtype=np.float32))
    target_turn = np.clip(steer, -1.0, 1.0) * steer_rate
    new_ang_vel = ang_vel * angular_drag + target_turn * (1.0 - angular_drag)
    new_heading = state['heading'] + new_ang_vel

    # Linear thrust & speed clamp
    thrust = np.clip(throttle, 0.0, 1.0) * 0.22
    fwd_x = np.cos(new_heading)
    fwd_y = np.sin(new_heading)
    lat_x = -fwd_y
    lat_y = fwd_x

    cur_vx = state['vel'][:, 0] * momentum + fwd_x * thrust
    cur_vy = state['vel'][:, 1] * momentum + fwd_y * thrust

    cur_speed = np.sqrt(cur_vx * cur_vx + cur_vy * cur_vy) + 1e-6
    max_speed = 0.40
    scale = np.minimum(1.0, max_speed / cur_speed)
    cur_vx *= scale
    cur_vy *= scale

    v_long = cur_vx * fwd_x + cur_vy * fwd_y
    v_lat  = cur_vx * lat_x + cur_vy * lat_y
    v_lat_retained = v_lat * (1.0 - traction)

    total_vx = fwd_x * v_long + lat_x * v_lat_retained
    total_vy = fwd_y * v_long + lat_y * v_lat_retained
    unresolved_vel = np.column_stack([total_vx, total_vy])

    is_airborne = altitude > 0.08

    # Ground agents slide along walls
    g_pos, g_vel, g_hit = resolve_wall_slide(state['pos'], unresolved_vel, radius, elasticity, grid_map)

    # Airborne agents glide over obstacles freely (outer boundary still blocks)
    air_pos = state['pos'] + unresolved_vel
    H, W = grid_map.shape
    r = radius
    x, y = air_pos[:, 0], air_pos[:, 1]
    oob = (x - r < 1.0) | (x + r > W - 1.0) | (y - r < 1.0) | (y + r > H - 1.0)

    new_pos = np.where(is_airborne[:, None], air_pos, g_pos).astype(np.float32)
    new_vel = np.where(is_airborne[:, None], unresolved_vel, g_vel).astype(np.float32)
    hit_mask = np.where(is_airborne, oob, g_hit)

    state['pos'] = new_pos
    state['vel'] = new_vel
    state['heading'] = new_heading
    if 'ang_vel' in state:
        state['ang_vel'] = new_ang_vel

    return hit_mask

def apply_rigid_kinematics(state: Dict[str, np.ndarray], steer: np.ndarray, throttle: np.ndarray,
                           speed_param: float, steer_rate: float, radius: float, grid_map: np.ndarray) -> np.ndarray:
    speed = np.clip(throttle, -1.0, 1.0) * min(speed_param, 0.38)
    new_heading = state['heading'] + np.clip(steer, -1.0, 1.0) * steer_rate
    fwd_x = np.cos(new_heading)
    fwd_y = np.sin(new_heading)
    unresolved_vel = np.column_stack([fwd_x * speed, fwd_y * speed])

    new_pos, resolved_vel, hit_mask = resolve_wall_slide(state['pos'], unresolved_vel, radius, 0.0, grid_map)
    state['pos'] = new_pos
    state['vel'] = resolved_vel
    state['heading'] = new_heading
    return hit_mask

def rotate_vec(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    ca = np.cos(angle)[:, None]
    sa = np.sin(angle)[:, None]
    rx = v[:, 0:1] * ca - v[:, 1:2] * sa
    ry = v[:, 0:1] * sa + v[:, 1:2] * ca
    return np.hstack([rx, ry]).astype(np.float32)

def _make_vec2(x, y, scope):
    n = len(scope["mask"]) if "mask" in scope else scope.get("num_envs", 1)
    x_arr = np.broadcast_to(np.asarray(x, dtype=np.float32), (n,))
    y_arr = np.broadcast_to(np.asarray(y, dtype=np.float32), (n,))
    return np.column_stack([x_arr, y_arr]).astype(np.float32)

def _vec_mul(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.ndim == 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
            return a * b[:, None]
        if b.ndim == 2 and a.ndim == 1 and b.shape[0] == a.shape[0]:
            return a[:, None] * b
    return a * b

def _vec_add(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.ndim == 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
            return a + b[:, None]
        if b.ndim == 2 and a.ndim == 1 and b.shape[0] == a.shape[0]:
            return a[:, None] + b
    return a + b

def _vec_sub(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.ndim == 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
            return a - b[:, None]
        if b.ndim == 2 and a.ndim == 1 and b.shape[0] == a.shape[0]:
            return a[:, None] - b
    return a - b

def _vec_div(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.ndim == 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
            return a / b[:, None]
    return a / b

def _vec_where(cond, a, b):
    if isinstance(cond, np.ndarray) and cond.ndim == 1:
        if (isinstance(a, np.ndarray) and a.ndim == 2) or (isinstance(b, np.ndarray) and b.ndim == 2):
            cond = cond[:, None]
    return np.where(cond, a, b)


# =====================================================================
# 3. TRANSPILER
# =====================================================================
class NevoRLTranspiler(Transformer):
    def __init__(self):
        super().__init__()
        self.env_name = ""
        self.state_schema: Dict[str, str] = {}
        self.action_space: Dict[str, Any] = {}
        self.action_var = "act"
        self.kernels: Dict[str, str] = {}

    def env_def(self, items):
        self.env_name = str(items[0])
        return self

    def field_decl(self, items):
        name, type_ = str(items[0]), str(items[1])
        self.state_schema[name] = type_
        return (name, type_)

    def state_sec(self, items):
        return ("state", self.state_schema)

    def act_discrete(self, items):
        self.action_space = {"type": "discrete", "n": int(items[0])}
        return self.action_space

    def act_continuous(self, items):
        self.action_space = {"type": "continuous", "dim": int(items[0])}
        return self.action_space

    def action_sec(self, items):
        return ("action", self.action_space)

    def dot_path(self, items):
        return [str(x) for x in items]

    def dot_target(self, items):
        return items[0]

    def var_target(self, items):
        return str(items[0])

    def dot_access(self, items):
        path = items[0]
        if len(path) == 2:
            if path[0] == "state":
                return f"state['{path[1]}']"
            else:
                idx = 0 if path[1] == "x" else (1 if path[1] == "y" else path[1])
                return f"{path[0]}[:, {idx}]"
        elif len(path) == 3 and path[0] == "state":
            idx = 0 if path[2] == "x" else (1 if path[2] == "y" else path[2])
            return f"state['{path[1]}'][:, {idx}]"
        return ".".join(path)

    def reset_assign(self, items):
        target, expr = items[0], str(items[1])
        if isinstance(target, list):
            if len(target) == 2 and target[0] == "state":
                return f"    state['{target[1]}'][mask] = {expr}"
            elif len(target) == 3 and target[0] == "state":
                idx = 0 if target[2] == "x" else 1
                return f"    state['{target[1]}'][mask, {idx}] = {expr}"
        return f"    {target} = {expr}"

    def assign_stmt(self, items):
        target, expr = items[0], str(items[1])
        if isinstance(target, list):
            if len(target) == 2 and target[0] == "state":
                return f"    state['{target[1]}'] = {expr}"
            elif len(target) == 3 and target[0] == "state":
                idx = 0 if target[2] == "x" else 1
                return f"    state['{target[1]}'][:, {idx}] = {expr}"
            return f"    {'.'.join(target)} = {expr}"
        return f"    {target} = {expr}"

    def let_stmt(self, items):
        return f"    {items[0]} = {items[1]}"

    def num(self, items):
        return str(items[0])

    def var(self, items):
        return str(items[0])

    def array_index(self, items):
        target, idx = str(items[0]), str(items[1])
        if target == self.action_var and self.action_space.get("type") == "continuous":
            return f"{target}[:, {idx}]"
        return f"{target}[{idx}]"

    def func_call(self, items):
        name = str(items[0])
        args = [str(a) for a in items[1:]]
        if name == "kinematics_car":
            if len(args) == 8:
                return f"apply_car_kinematics(state, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]}, np.zeros(num_envs, dtype=np.float32), grid_map)"
            elif len(args) >= 9:
                return f"apply_car_kinematics(state, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]}, {args[8]}, grid_map)"
        elif name == "kinematics_rigid":
            return f"apply_rigid_kinematics(state, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, grid_map)"
        elif name == "uniform":
            return f"uniform({args[0]}, {args[1]})"
        elif name == "vec2":
            return f"_make_vec2({args[0]}, {args[1]}, locals())"
        elif name == "length":
            return f"np.linalg.norm({args[0]}, axis=-1)"
        elif name == "rotate":
            return f"rotate_vec({args[0]}, {args[1]})"
        elif name == "raycast_fan":
            return f"fast_raycast_fan({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, grid_map)"
        elif name == "is_wall":
            return f"check_circle_aabb_collision({args[0]}, {args[1]}, grid_map)"
        elif name == "bfs_dist":
            return f"get_bfs_dist({args[0]}, {args[1]}, grid_map)"
        elif name == "bfs_vector":
            return f"get_bfs_vector({args[0]}, {args[1]}, grid_map)"
        elif name == "where":
            return f"_vec_where({args[0]}, {args[1]}, {args[2]})"
        elif name == "clamp":
            return f"np.clip({args[0]}, {args[1]}, {args[2]})"
        elif name in ("cos", "sin", "abs", "sqrt"):
            return f"np.{name}({args[0]})"
        return f"np.{name}({', '.join(args)})"

    def list_lit(self, items):
        parts = []
        for it in items:
            s = str(it)
            parts.append(f"({s}[:, None] if {s}.ndim == 1 else {s})")
        return f"np.hstack([{', '.join(parts)}]).astype(np.float32)"

    def or_expr(self, items):
        res = str(items[0])
        for i in range(1, len(items), 2):
            res = f"({res} | {items[i+1]})"
        return res

    def and_expr(self, items):
        res = str(items[0])
        for i in range(1, len(items), 2):
            res = f"({res} & {items[i+1]})"
        return res

    def comp_expr(self, items):
        res = str(items[0])
        for i in range(1, len(items), 2):
            res = f"({res} {items[i]} {items[i+1]})"
        return res

    def arith_expr(self, items):
        res = str(items[0])
        for i in range(1, len(items), 2):
            op = items[i]
            fn = "_vec_add" if op == "+" else "_vec_sub"
            res = f"{fn}({res}, {items[i+1]})"
        return res

    def term(self, items):
        res = str(items[0])
        for i in range(1, len(items), 2):
            op = items[i]
            fn = "_vec_mul" if op == "*" else "_vec_div"
            res = f"{fn}({res}, {items[i+1]})"
        return res

    def factor(self, items):
        if len(items) == 2:
            return f"(-{items[1]})"
        return str(items[0])

    def reset_sec(self, items):
        lines = [
            "def _reset_kernel(state, mask, num_envs, grid_map):",
            "    uniform = lambda a, b: np.random.uniform(a, b, size=len(mask)).astype(np.float32)"
        ] + [str(s) for s in items]
        self.kernels["_reset_kernel"] = "\n".join(lines)
        return self.kernels["_reset_kernel"]

    def step_sec(self, items):
        self.action_var = str(items[0])
        lines = [
            f"def _step_kernel(state, {self.action_var}, num_envs, grid_map):",
            "    uniform = lambda a, b: np.random.uniform(a, b, size=num_envs).astype(np.float32)"
        ] + [str(s) for s in items[1:]]
        self.kernels["_step_kernel"] = "\n".join(lines)
        return self.kernels["_step_kernel"]

    def reward_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _reward_kernel(state, num_envs, grid_map):"] + body + [f"    return np.asarray({ret_expr}, dtype=np.float32)"]
        self.kernels["_reward_kernel"] = "\n".join(lines)
        return self.kernels["_reward_kernel"]

    def terminal_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _terminal_kernel(state, num_envs, grid_map):"] + body + [f"    return np.asarray({ret_expr}, dtype=bool)"]
        self.kernels["_terminal_kernel"] = "\n".join(lines)
        return self.kernels["_terminal_kernel"]

    def obs_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _obs_kernel(state, num_envs, grid_map):"] + body + [f"    return {ret_expr}"]
        self.kernels["_obs_kernel"] = "\n".join(lines)
        return self.kernels["_obs_kernel"]


# =====================================================================
# 4. COMPILED VECTORIZED ENVIRONMENT
# =====================================================================
class CompiledVectorEnv:
    def __init__(self, num_envs: int, schema: Dict[str, str], act_space: Dict[str, Any], kernels: Dict[str, Any], grid_map: np.ndarray | None = None):
        self.num_envs = num_envs
        self.schema = schema
        self.action_space_info = act_space
        self.kernels = kernels
        H, W = (24, 30) if grid_map is None else grid_map.shape
        self.grid_map = np.zeros((H, W), dtype=np.uint8) if grid_map is None else grid_map

        self.state: Dict[str, np.ndarray] = {}
        for name, dtype_str in self.schema.items():
            if dtype_str == "float":
                self.state[name] = np.zeros(self.num_envs, dtype=np.float32)
            elif dtype_str == "int":
                self.state[name] = np.zeros(self.num_envs, dtype=np.int32)
            elif dtype_str == "bool":
                self.state[name] = np.zeros(self.num_envs, dtype=bool)
            elif dtype_str == "vec2":
                self.state[name] = np.zeros((self.num_envs, 2), dtype=np.float32)

        self._reset_all()

    def _reset_all(self):
        all_mask = np.arange(self.num_envs, dtype=np.int32)
        self.kernels["_reset_kernel"](self.state, all_mask, self.num_envs, self.grid_map)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._reset_all()
        obs = self.kernels["_obs_kernel"](self.state, self.num_envs, self.grid_map)
        return obs, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        self.kernels["_step_kernel"](self.state, actions, self.num_envs, self.grid_map)

        rewards = self.kernels["_reward_kernel"](self.state, self.num_envs, self.grid_map)
        if rewards.ndim == 0 or (rewards.ndim == 1 and len(rewards) == 1):
            rewards = np.full(self.num_envs, rewards, dtype=np.float32)

        terminated = self.kernels["_terminal_kernel"](self.state, self.num_envs, self.grid_map)
        if terminated.ndim == 0:
            terminated = np.full(self.num_envs, terminated, dtype=bool)

        truncated = np.zeros(self.num_envs, dtype=bool)
        obs = self.kernels["_obs_kernel"](self.state, self.num_envs, self.grid_map)

        dones = np.where(terminated | truncated)[0]
        infos: Dict[str, Any] = {}

        if len(dones) > 0:
            infos["final_observation"] = obs[dones].copy()
            infos["_final_observation"] = dones

            self.kernels["_reset_kernel"](self.state, dones, self.num_envs, self.grid_map)
            reset_obs = self.kernels["_obs_kernel"](self.state, self.num_envs, self.grid_map)
            obs[dones] = reset_obs[dones]

        return obs, rewards, terminated, truncated, infos


class NevoRLCompiler:
    def __init__(self):
        self.parser = Lark(NEVORL_GRAMMAR, parser="lalr")

    def compile_source(self, source_code: str) -> type:
        tree = self.parser.parse(source_code)
        transpiler = NevoRLTranspiler()

        for node in tree.find_data("step_sec"):
            transpiler.action_var = str(node.children[0])
        for node in tree.find_data("act_continuous"):
            transpiler.action_space = {"type": "continuous", "dim": int(node.children[0])}
        for node in tree.find_data("act_discrete"):
            transpiler.action_space = {"type": "discrete", "n": int(node.children[0])}

        transpiler.transform(tree)

        exec_env = {
            "np": np,
            "fast_raycast_fan": fast_raycast_fan,
            "check_circle_aabb_collision": check_circle_aabb_collision,
            "resolve_wall_slide": resolve_wall_slide,
            "apply_car_kinematics": apply_car_kinematics,
            "apply_rigid_kinematics": apply_rigid_kinematics,
            "get_bfs_dist": get_bfs_dist,
            "get_bfs_vector": get_bfs_vector,
            "rotate_vec": rotate_vec,
            "_make_vec2": _make_vec2,
            "_vec_mul": _vec_mul,
            "_vec_add": _vec_add,
            "_vec_sub": _vec_sub,
            "_vec_div": _vec_div,
            "_vec_where": _vec_where
        }
        for name, code in transpiler.kernels.items():
            exec(code, exec_env)

        schema = transpiler.state_schema
        act_space = transpiler.action_space

        class GeneratedEnv(CompiledVectorEnv):
            def __init__(self, num_envs: int = 64, grid_map: np.ndarray | None = None):
                super().__init__(num_envs=num_envs, schema=schema, act_space=act_space, kernels=exec_env, grid_map=grid_map)

        GeneratedEnv.__name__ = transpiler.env_name
        return GeneratedEnv

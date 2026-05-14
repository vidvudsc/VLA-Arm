from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


ACTION_DIM = 3
ACTION_LABELS = ["shoulder_velocity", "elbow_velocity", "magnet_command"]


@dataclass(frozen=True)
class ArmConfig:
    world_size: int = 256
    base_x: float = 128.0
    base_y: float = 148.0
    link1: float = 78.0
    link2: float = 58.0
    joint_step: float = 0.075
    object_radius: float = 8.0
    bowl_radius: float = 13.0
    pick_radius: float = 10.0
    place_radius: float = 14.0
    min_spawn_radius: float = 36.0
    max_spawn_radius: float = 120.0
    max_steps: int = 150


@dataclass
class ObjectSpec:
    x: float
    y: float


@dataclass
class BowlSpec:
    x: float
    y: float


@dataclass
class ArmState:
    shoulder: float
    elbow: float
    obj: ObjectSpec
    bowl: BowlSpec
    holding: bool = False
    placed: bool = False
    step_count: int = 0


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def forward_kinematics(state: ArmState, cfg: ArmConfig) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    base = (cfg.base_x, cfg.base_y)
    joint = (
        cfg.base_x + cfg.link1 * math.cos(state.shoulder),
        cfg.base_y + cfg.link1 * math.sin(state.shoulder),
    )
    ee_angle = state.shoulder + state.elbow
    end = (
        joint[0] + cfg.link2 * math.cos(ee_angle),
        joint[1] + cfg.link2 * math.sin(ee_angle),
    )
    return base, joint, end


def object_position(state: ArmState, cfg: ArmConfig) -> tuple[float, float]:
    if state.holding:
        return forward_kinematics(state, cfg)[2]
    return state.obj.x, state.obj.y


def state_vector(state: ArmState, cfg: ArmConfig) -> np.ndarray:
    # Realistic proprioception only: joint angles and gripper/holding state.
    return np.array(
        [
            math.sin(state.shoulder),
            math.cos(state.shoulder),
            math.sin(state.elbow),
            math.cos(state.elbow),
            1.0 if state.holding else 0.0,
        ],
        dtype=np.float32,
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_to_object(state: ArmState, cfg: ArmConfig) -> float:
    return distance(forward_kinematics(state, cfg)[2], object_position(state, cfg))


def distance_to_bowl(state: ArmState, cfg: ArmConfig) -> float:
    return distance(forward_kinematics(state, cfg)[2], (state.bowl.x, state.bowl.y))


def is_success(state: ArmState, cfg: ArmConfig) -> bool:
    return state.placed and distance((state.obj.x, state.obj.y), (state.bowl.x, state.bowl.y)) <= cfg.place_radius


def clone_state(state: ArmState) -> ArmState:
    return ArmState(
        shoulder=state.shoulder,
        elbow=state.elbow,
        obj=ObjectSpec(state.obj.x, state.obj.y),
        bowl=BowlSpec(state.bowl.x, state.bowl.y),
        holding=state.holding,
        placed=state.placed,
        step_count=state.step_count,
    )


def apply_action(state: ArmState, action: np.ndarray | list[float] | tuple[float, ...], cfg: ArmConfig) -> ArmState:
    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.shape[0] != ACTION_DIM:
        raise ValueError(f"expected action dim {ACTION_DIM}, got {action_arr.shape}")

    new = clone_state(state)
    new.step_count += 1
    new.shoulder = wrap_angle(new.shoulder + float(np.clip(action_arr[0], -1.0, 1.0)) * cfg.joint_step)
    new.elbow = wrap_angle(new.elbow + float(np.clip(action_arr[1], -1.0, 1.0)) * cfg.joint_step)

    magnet = float(np.clip(action_arr[2], -1.0, 1.0))
    if magnet > 0.45 and not new.holding and not new.placed and distance_to_object(new, cfg) <= cfg.pick_radius:
        new.holding = True
    elif magnet < -0.45 and new.holding:
        end = forward_kinematics(new, cfg)[2]
        new.obj.x, new.obj.y = end
        new.holding = False
        if distance(end, (new.bowl.x, new.bowl.y)) <= cfg.place_radius:
            new.obj.x, new.obj.y = new.bowl.x, new.bowl.y
            new.placed = True
    return new


def inverse_kinematics(target_x: float, target_y: float, cfg: ArmConfig) -> tuple[float, float]:
    dx = target_x - cfg.base_x
    dy = target_y - cfg.base_y
    dist_to_target = max(1e-6, math.hypot(dx, dy))
    dist_to_target = min(
        cfg.link1 + cfg.link2 - 1e-3,
        max(abs(cfg.link1 - cfg.link2) + 1e-3, dist_to_target),
    )
    cos_elbow = (dist_to_target * dist_to_target - cfg.link1 * cfg.link1 - cfg.link2 * cfg.link2) / (2 * cfg.link1 * cfg.link2)
    cos_elbow = min(1.0, max(-1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(dy, dx) - math.atan2(cfg.link2 * math.sin(elbow), cfg.link1 + cfg.link2 * math.cos(elbow))
    return wrap_angle(shoulder), wrap_angle(elbow)


def expert_action(state: ArmState, cfg: ArmConfig) -> np.ndarray:
    if is_success(state, cfg):
        return np.zeros(ACTION_DIM, dtype=np.float32)
    if state.holding and distance_to_bowl(state, cfg) <= cfg.place_radius:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if not state.holding and distance_to_object(state, cfg) <= cfg.pick_radius:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    target_x, target_y = (state.bowl.x, state.bowl.y) if state.holding else object_position(state, cfg)
    desired_shoulder, desired_elbow = inverse_kinematics(target_x, target_y, cfg)
    ds = wrap_angle(desired_shoulder - state.shoulder)
    de = wrap_angle(desired_elbow - state.elbow)
    return np.array(
        [
            float(np.clip(ds / cfg.joint_step, -1.0, 1.0)),
            float(np.clip(de / cfg.joint_step, -1.0, 1.0)),
            1.0 if state.holding else 0.0,
        ],
        dtype=np.float32,
    )


def sample_reachable_point(rng: random.Random, cfg: ArmConfig) -> tuple[float, float]:
    margin = max(cfg.object_radius, cfg.bowl_radius) + 4.0
    for _ in range(500):
        radius = rng.uniform(cfg.min_spawn_radius, cfg.max_spawn_radius)
        angle = rng.uniform(-math.pi, math.pi)
        x = cfg.base_x + radius * math.cos(angle)
        y = cfg.base_y + radius * math.sin(angle)
        if margin <= x <= cfg.world_size - margin and margin <= y <= cfg.world_size - margin:
            return x, y
    raise RuntimeError("failed to sample a visible reachable point")


def make_scene(seed: int, cfg: ArmConfig) -> ArmState:
    rng = random.Random(seed)
    obj = ObjectSpec(*sample_reachable_point(rng, cfg))
    min_separation = cfg.object_radius + cfg.bowl_radius + 34.0
    for _ in range(500):
        bowl = BowlSpec(*sample_reachable_point(rng, cfg))
        if distance((obj.x, obj.y), (bowl.x, bowl.y)) > min_separation:
            break
    else:
        raise RuntimeError("failed to sample a non-overlapping object/bowl pair")

    start_target = obj if rng.random() < 0.75 else bowl
    shoulder, elbow = inverse_kinematics(start_target.x, start_target.y, cfg)
    shoulder = wrap_angle(shoulder + rng.uniform(-1.6, 1.6))
    elbow = wrap_angle(elbow + rng.uniform(-1.3, 1.3))
    return ArmState(shoulder=shoulder, elbow=elbow, obj=obj, bowl=bowl)

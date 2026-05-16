from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .env import ArmConfig, apply_action, distance, distance_to_bowl, distance_to_object, forward_kinematics, is_success, make_scene, state_vector
from .model import PolicyConfig, VLAArmPolicy
from .render import image_to_tensor, render_state


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_policy(path: str, device: torch.device) -> tuple[VLAArmPolicy, ArmConfig]:
    ckpt = torch.load(path, map_location="cpu")
    policy_cfg = PolicyConfig(**ckpt.get("policy_config", {}))
    arm_cfg = ArmConfig(**ckpt.get("arm_config", {}))
    model = VLAArmPolicy(policy_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, arm_cfg


@torch.no_grad()
def policy_action_chunk(model: VLAArmPolicy, state, cfg: ArmConfig, device: torch.device):
    image = image_to_tensor(render_state(state, cfg)).unsqueeze(0).to(device)
    robot = torch.from_numpy(state_vector(state, cfg)).unsqueeze(0).to(device)
    return model(image, robot).squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def policy_action(model: VLAArmPolicy, state, cfg: ArmConfig, device: torch.device):
    return policy_action_chunk(model, state, cfg, device)[0]


def rollout(
    model: VLAArmPolicy,
    cfg: ArmConfig,
    seed: int,
    device: torch.device,
    render_dir: Path | None = None,
    render_every: int = 5,
    temporal_ensemble: bool = False,
    ensemble_decay: float = 0.01,
    reset_ensemble_on_gripper_change: bool = True,
) -> dict[str, object]:
    state = make_scene(seed, cfg)
    frames = []
    pending_chunks = []
    render_stride = max(1, render_every)
    picked_once = False
    released_once = False
    min_object_distance = distance_to_object(state, cfg)
    min_bowl_distance = distance_to_bowl(state, cfg)
    min_bowl_distance_while_holding = float("inf")
    for step in range(cfg.max_steps):
        if render_dir is not None and step % render_stride == 0:
            frames.append(render_state(state, cfg))
        if temporal_ensemble:
            action_chunk = policy_action_chunk(model, state, cfg, device)
            pending_chunks.append(action_chunk)
            weighted_actions = []
            weights = []
            for age, chunk in enumerate(reversed(pending_chunks)):
                if age < len(chunk):
                    weighted_actions.append(chunk[age])
                    weights.append(math.exp(-ensemble_decay * age))
            action = sum(w * a for w, a in zip(weights, weighted_actions)) / max(sum(weights), 1e-9)
            pending_chunks = pending_chunks[-len(action_chunk) :]
        else:
            action = policy_action(model, state, cfg, device)
        was_holding = state.holding
        state = apply_action(state, action, cfg)
        tip = forward_kinematics(state, cfg)[2]
        obj_pos = (state.obj.x, state.obj.y)
        bowl_pos = (state.bowl.x, state.bowl.y)
        if not was_holding and state.holding:
            picked_once = True
            if temporal_ensemble and reset_ensemble_on_gripper_change:
                pending_chunks.clear()
        if was_holding and not state.holding:
            released_once = True
            if temporal_ensemble and reset_ensemble_on_gripper_change:
                pending_chunks.clear()
        if not state.holding and not state.placed:
            min_object_distance = min(min_object_distance, distance(tip, obj_pos))
        min_bowl_distance = min(min_bowl_distance, distance(tip, bowl_pos))
        if state.holding:
            min_bowl_distance_while_holding = min(min_bowl_distance_while_holding, distance(tip, bowl_pos))
        if is_success(state, cfg):
            break
    if render_dir is not None and frames:
        frames.append(render_state(state, cfg))
        path = render_dir / f"episode_{seed}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return {
        "success": is_success(state, cfg),
        "steps": state.step_count,
        "holding": state.holding,
        "placed": state.placed,
        "picked_once": picked_once,
        "released_once": released_once,
        "min_object_distance": min_object_distance,
        "min_bowl_distance": min_bowl_distance,
        "min_bowl_distance_while_holding": None
        if math.isinf(min_bowl_distance_while_holding)
        else min_bowl_distance_while_holding,
        "final_object_distance": distance_to_object(state, cfg),
        "final_bowl_distance": distance_to_bowl(state, cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLA-Arm rollout success")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--render_dir", default="")
    parser.add_argument("--render_every", type=int, default=5)
    parser.add_argument("--temporal_ensemble", action="store_true")
    parser.add_argument("--ensemble_decay", type=float, default=0.01)
    parser.add_argument("--no_reset_ensemble_on_gripper_change", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg = load_policy(args.checkpoint, device)
    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
    results = [
        rollout(
            model,
            cfg,
            args.seed + idx,
            device,
            render_dir if idx < 4 else None,
            render_every=args.render_every,
            temporal_ensemble=args.temporal_ensemble,
            ensemble_decay=args.ensemble_decay,
            reset_ensemble_on_gripper_change=not args.no_reset_ensemble_on_gripper_change,
        )
        for idx in range(args.episodes)
    ]
    success = sum(bool(item["success"]) for item in results) / max(1, len(results))
    avg_steps = sum(int(item["steps"]) for item in results) / max(1, len(results))
    picked = sum(bool(item["picked_once"]) for item in results) / max(1, len(results))
    released = sum(bool(item["released_once"]) for item in results) / max(1, len(results))
    min_object = sum(float(item["min_object_distance"]) for item in results) / max(1, len(results))
    final_object = sum(float(item["final_object_distance"]) for item in results) / max(1, len(results))
    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "success_rate": success,
                "picked_rate": picked,
                "released_rate": released,
                "avg_min_object_distance": min_object,
                "avg_final_object_distance": final_object,
                "avg_steps": avg_steps,
            },
            indent=2,
        )
    )
    for item in results[:5]:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()

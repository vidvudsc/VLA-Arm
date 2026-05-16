from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from vla_arm.env import ArmConfig, apply_action, distance_to_bowl, distance_to_object, forward_kinematics, is_success, make_scene
from vla_arm.eval import load_policy, pick_device, policy_action_chunk
from vla_arm.render import render_state


def choose_action(
    action_chunk: np.ndarray,
    pending_chunks: list[np.ndarray],
    ensemble_decay: float,
    ensemble_gripper: bool,
) -> np.ndarray:
    pending_chunks.append(action_chunk)
    weighted_actions = []
    weights = []
    for age, chunk in enumerate(reversed(pending_chunks)):
        if age < len(chunk):
            weighted_actions.append(chunk[age])
            weights.append(math.exp(-ensemble_decay * age))
    action = sum(w * a for w, a in zip(weights, weighted_actions)) / max(sum(weights), 1e-9)
    if not ensemble_gripper:
        action[2] = action_chunk[0, 2]
    del pending_chunks[:-len(action_chunk)]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one VLA-Arm rollout step-by-step.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--temporal_ensemble", action="store_true")
    parser.add_argument("--ensemble_decay", type=float, default=0.01)
    parser.add_argument("--ensemble_gripper", action="store_true")
    parser.add_argument("--render_dir", default="")
    parser.add_argument("--render_every", type=int, default=1)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg = load_policy(args.checkpoint, device)
    cfg = ArmConfig(**{**cfg.__dict__, "max_steps": min(cfg.max_steps, args.max_steps)})
    state = make_scene(args.seed, cfg)
    pending_chunks: list[np.ndarray] = []
    frames = []
    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    picked = False
    released = False
    print("step | obj_px | bowl_px | shoulder | elbow | magnet | holding | events")
    for step in range(cfg.max_steps):
        if render_dir is not None and step % max(1, args.render_every) == 0:
            frames.append(render_state(state, cfg))
        action_chunk = policy_action_chunk(model, state, cfg, device)
        if args.temporal_ensemble:
            action = choose_action(action_chunk, pending_chunks, args.ensemble_decay, args.ensemble_gripper)
        else:
            action = action_chunk[0]

        obj_dist = distance_to_object(state, cfg)
        bowl_dist = distance_to_bowl(state, cfg)
        before_holding = state.holding
        state = apply_action(state, action, cfg)

        events = []
        if not before_holding and state.holding:
            picked = True
            events.append("pickup")
            pending_chunks.clear()
        if before_holding and not state.holding:
            released = True
            events.append("release")
            pending_chunks.clear()
        if is_success(state, cfg):
            events.append("success")

        print(
            f"{step:4d} | {obj_dist:6.1f} | {bowl_dist:7.1f} | "
            f"{float(action[0]):8.3f} | {float(action[1]):5.3f} | {float(action[2]):6.3f} | "
            f"{int(state.holding):7d} | {','.join(events)}"
        )
        if is_success(state, cfg):
            break

    if render_dir is not None and frames:
        frames.append(render_state(state, cfg))
        frames[0].save(render_dir / f"trace_{args.seed}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

    tip = forward_kinematics(state, cfg)[2]
    print(
        "summary:",
        {
            "success": is_success(state, cfg),
            "picked": picked,
            "released": released,
            "holding_end": state.holding,
            "placed": state.placed,
            "tip": tuple(round(v, 2) for v in tip),
            "obj_px": round(distance_to_object(state, cfg), 2),
            "bowl_px": round(distance_to_bowl(state, cfg), 2),
        },
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

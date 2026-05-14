from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .env import ArmConfig, apply_action, is_success, make_scene, state_vector
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
def policy_action(model: VLAArmPolicy, state, cfg: ArmConfig, device: torch.device):
    image = image_to_tensor(render_state(state, cfg)).unsqueeze(0).to(device)
    robot = torch.from_numpy(state_vector(state, cfg)).unsqueeze(0).to(device)
    return model(image, robot).squeeze(0).detach().cpu().numpy()


def rollout(model: VLAArmPolicy, cfg: ArmConfig, seed: int, device: torch.device, render_dir: Path | None = None) -> dict[str, object]:
    state = make_scene(seed, cfg)
    frames = []
    for step in range(cfg.max_steps):
        if render_dir is not None and step % 5 == 0:
            frames.append(render_state(state, cfg))
        action = policy_action(model, state, cfg, device)
        state = apply_action(state, action, cfg)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLA-Arm rollout success")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--render_dir", default="")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg = load_policy(args.checkpoint, device)
    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
    results = [rollout(model, cfg, args.seed + idx, device, render_dir if idx < 4 else None) for idx in range(args.episodes)]
    success = sum(bool(item["success"]) for item in results) / max(1, len(results))
    avg_steps = sum(int(item["steps"]) for item in results) / max(1, len(results))
    print(json.dumps({"episodes": args.episodes, "success_rate": success, "avg_steps": avg_steps}, indent=2))
    for item in results[:5]:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()

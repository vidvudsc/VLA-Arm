from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ExpertTransitionDataset
from .env import ACTION_LABELS, ArmConfig
from .eval import rollout
from .model import PolicyConfig, VLAArmPolicy, count_parameters


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def weighted_action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_dim_weight: float,
    magnet_dim_weight: float,
    magnet_event_weight: float,
) -> torch.Tensor:
    sq = (pred - target).pow(2)
    dim_weight = torch.ones_like(sq)
    dim_weight[:, :2] = joint_dim_weight
    dim_weight[:, 2] = magnet_dim_weight
    event_weight = torch.where(target[:, 2].abs() > 0.5, magnet_event_weight, 1.0).unsqueeze(1)
    weights = dim_weight * event_weight
    return (sq * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def validation_loss(
    model: VLAArmPolicy,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    joint_dim_weight: float,
    magnet_dim_weight: float,
    magnet_event_weight: float,
) -> float:
    model.eval()
    iterator = iter(loader)
    total = 0.0
    seen = 0
    for _ in range(batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = move_batch(batch, device)
        pred = model(batch["image"], batch["robot"])
        loss = weighted_action_mse(pred, batch["action"], joint_dim_weight, magnet_dim_weight, magnet_event_weight)
        total += float(loss.detach().cpu())
        seen += 1
    model.train()
    return total / max(1, seen)


@torch.no_grad()
def rollout_eval(
    model: VLAArmPolicy,
    cfg: ArmConfig,
    device: torch.device,
    episodes: int,
    seed: int,
    render_dir: Path | None,
    gif_episodes: int,
) -> dict[str, float]:
    model.eval()
    results = []
    for idx in range(episodes):
        episode_render_dir = render_dir if render_dir is not None and idx < gif_episodes else None
        results.append(rollout(model, cfg, seed + idx, device, episode_render_dir))
    model.train()
    success_rate = sum(bool(item["success"]) for item in results) / max(1, len(results))
    avg_steps = sum(int(item["steps"]) for item in results) / max(1, len(results))
    held_rate = sum(bool(item["holding"]) for item in results) / max(1, len(results))
    placed_rate = sum(bool(item["placed"]) for item in results) / max(1, len(results))
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "held_rate": held_rate,
        "placed_rate": placed_rate,
    }


def save_checkpoint(model: VLAArmPolicy, policy_cfg: PolicyConfig, arm_cfg: ArmConfig, out_dir: Path, step: int) -> None:
    ckpt = {
        "model": model.state_dict(),
        "policy_config": asdict(policy_cfg),
        "arm_config": asdict(arm_cfg),
        "step": step,
    }
    torch.save(ckpt, out_dir / "policy_last.pt")
    torch.save(ckpt, out_dir / f"policy_step_{step}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLA-Arm with synthetic expert behavior cloning")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--dataset_len", type=int, default=100_000)
    parser.add_argument("--val_len", type=int, default=2048)
    parser.add_argument("--cache_samples", type=int, default=0, help="Pre-render this many train samples as uint8 tensors; 0 disables cache.")
    parser.add_argument("--cache_val_samples", type=int, default=1024, help="Pre-render this many validation samples as uint8 tensors; 0 disables cache.")
    parser.add_argument("--event_sample_prob", type=float, default=0.25, help="Probability of sampling a pickup/release transition when a rollout contains one.")
    parser.add_argument("--recovery_noise_prob", type=float, default=0.35, help="Probability of perturbing a sampled train state before relabeling with the expert.")
    parser.add_argument("--recovery_noise_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=16)
    parser.add_argument("--rollout_episodes", type=int, default=12)
    parser.add_argument("--rollout_seed", type=int, default=50_000)
    parser.add_argument("--gif_episodes", type=int, default=2)
    parser.add_argument("--joint_dim_weight", type=float, default=2.0)
    parser.add_argument("--magnet_dim_weight", type=float, default=1.0)
    parser.add_argument("--magnet_event_weight", type=float, default=3.0)
    parser.add_argument("--out_dir", default="runs/v0")
    args = parser.parse_args()

    device = pick_device(args.device)
    arm_cfg = ArmConfig()
    policy_cfg = PolicyConfig(image_size=arm_cfg.world_size)
    train_dataset = ExpertTransitionDataset(
        args.dataset_len,
        cfg=arm_cfg,
        seed=1234,
        cache_samples=args.cache_samples,
        event_sample_prob=args.event_sample_prob,
        recovery_noise_prob=args.recovery_noise_prob,
        recovery_noise_steps=args.recovery_noise_steps,
    )
    val_dataset = ExpertTransitionDataset(
        args.val_len,
        cfg=arm_cfg,
        seed=900_000,
        cache_samples=args.cache_val_samples,
        event_sample_prob=args.event_sample_prob,
        recovery_noise_prob=0.0,
        recovery_noise_steps=0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    iterator = iter(train_loader)
    model = VLAArmPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"params: {count_parameters(model):,}")
    print(f"continuous action: {ACTION_LABELS}")
    print(f"train samples: {len(train_dataset):,} | val samples: {len(val_dataset):,}")
    print(json.dumps({"policy": asdict(policy_cfg), "arm": asdict(arm_cfg)}, indent=2))

    start = time.time()
    ema_loss = None
    model.train()
    for step in tqdm(range(1, args.steps + 1), dynamic_ncols=True):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        pred = model(batch["image"], batch["robot"])
        loss = weighted_action_mse(
            pred,
            batch["action"],
            args.joint_dim_weight,
            args.magnet_dim_weight,
            args.magnet_event_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            joint_error = (pred[:, :2] - batch["action"][:, :2]).abs().mean()
            grip_error = (pred[:, 2] - batch["action"][:, 2]).abs().mean()
            magnet_active = (batch["action"][:, 2].abs() > 0.5).float().mean()
        loss_value = float(loss.detach().cpu())
        joint_score = float((1.0 - joint_error.detach().cpu()).clamp(0.0, 1.0))
        grip_value = float(grip_error.detach().cpu())
        event_value = float(magnet_active.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_value

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:5d} | loss {loss_value:.4f} | ema {ema_loss:.4f} | "
                f"joint_score {joint_score:.3f} | grip_err {grip_value:.3f} | "
                f"mag_active {event_value:.3f} | {step / max(elapsed, 1e-9):.2f} step/s"
            )

        if step % args.eval_every == 0 or step == args.steps:
            val = validation_loss(
                model,
                val_loader,
                device,
                args.val_batches,
                args.joint_dim_weight,
                args.magnet_dim_weight,
                args.magnet_event_weight,
            )
            gif_dir = out_dir / "policy_gifs" / f"step_{step:06d}" if args.gif_episodes > 0 else None
            if gif_dir is not None:
                gif_dir.mkdir(parents=True, exist_ok=True)
            roll = rollout_eval(
                model,
                arm_cfg,
                device,
                args.rollout_episodes,
                args.rollout_seed + step * 100,
                gif_dir,
                args.gif_episodes,
            )
            print(
                f"eval {step:5d} | val_loss {val:.4f} | success {roll['success_rate']:.3f} | "
                f"placed {roll['placed_rate']:.3f} | holding_end {roll['held_rate']:.3f} | "
                f"avg_steps {roll['avg_steps']:.1f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, policy_cfg, arm_cfg, out_dir, step)


if __name__ == "__main__":
    main()

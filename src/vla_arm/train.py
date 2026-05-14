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


def log_line(message: str) -> None:
    tqdm.write(message)
    import sys

    sys.stdout.flush()


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
    loss_type: str = "mse",
    huber_delta: float = 0.25,
) -> torch.Tensor:
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    error = pred - target
    if loss_type == "huber":
        abs_error = error.abs()
        delta = max(huber_delta, 1e-6)
        base_loss = torch.where(abs_error <= delta, 0.5 * error.pow(2) / delta, abs_error - 0.5 * delta)
    elif loss_type == "l1":
        base_loss = error.abs()
    else:
        base_loss = error.pow(2)
    dim_weight = torch.ones_like(base_loss)
    dim_weight[..., :2] = joint_dim_weight
    dim_weight[..., 2] = magnet_dim_weight
    event_weight = torch.where(target[..., 2].abs() > 0.5, magnet_event_weight, 1.0).unsqueeze(-1)
    weights = dim_weight * event_weight
    return (base_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_dim_weight: float,
    magnet_dim_weight: float,
    magnet_event_weight: float,
    joint_direction_weight: float,
    loss_type: str = "mse",
    huber_delta: float = 0.25,
) -> torch.Tensor:
    regression_loss = weighted_action_mse(
        pred,
        target,
        joint_dim_weight,
        magnet_dim_weight,
        magnet_event_weight,
        loss_type,
        huber_delta,
    )
    if joint_direction_weight <= 0.0:
        return regression_loss
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)

    target_joint = target[..., :2].reshape(-1, 2)
    pred_joint = pred[..., :2].reshape(-1, 2)
    active = target_joint.norm(dim=1) > 0.05
    if not bool(active.any()):
        return regression_loss

    pred_unit = torch.nn.functional.normalize(pred_joint[active], dim=1)
    target_unit = torch.nn.functional.normalize(target_joint[active], dim=1)
    direction_loss = (1.0 - (pred_unit * target_unit).sum(dim=1)).mean()
    return regression_loss + joint_direction_weight * direction_loss


@torch.no_grad()
def validation_loss(
    model: VLAArmPolicy,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    joint_dim_weight: float,
    magnet_dim_weight: float,
    magnet_event_weight: float,
    joint_direction_weight: float,
    loss_type: str,
    huber_delta: float,
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
        loss = action_loss(
            pred,
            batch["action"],
            joint_dim_weight,
            magnet_dim_weight,
            magnet_event_weight,
            joint_direction_weight,
            loss_type,
            huber_delta,
        )
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
    render_every: int,
    temporal_ensemble: bool,
    ensemble_decay: float,
) -> dict[str, float]:
    model.eval()
    results = []
    for idx in range(episodes):
        episode_render_dir = render_dir if render_dir is not None and idx < gif_episodes else None
        results.append(
            rollout(
                model,
                cfg,
                seed + idx,
                device,
                episode_render_dir,
                render_every=render_every,
                temporal_ensemble=temporal_ensemble,
                ensemble_decay=ensemble_decay,
            )
        )
    model.train()
    success_rate = sum(bool(item["success"]) for item in results) / max(1, len(results))
    avg_steps = sum(int(item["steps"]) for item in results) / max(1, len(results))
    held_rate = sum(bool(item["holding"]) for item in results) / max(1, len(results))
    placed_rate = sum(bool(item["placed"]) for item in results) / max(1, len(results))
    picked_rate = sum(bool(item["picked_once"]) for item in results) / max(1, len(results))
    released_rate = sum(bool(item["released_once"]) for item in results) / max(1, len(results))
    min_object_distance = sum(float(item["min_object_distance"]) for item in results) / max(1, len(results))
    final_object_distance = sum(float(item["final_object_distance"]) for item in results) / max(1, len(results))
    bowl_when_holding = [
        float(item["min_bowl_distance_while_holding"])
        for item in results
        if item["min_bowl_distance_while_holding"] is not None
    ]
    min_bowl_distance_while_holding = sum(bowl_when_holding) / len(bowl_when_holding) if bowl_when_holding else None
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "held_rate": held_rate,
        "placed_rate": placed_rate,
        "picked_rate": picked_rate,
        "released_rate": released_rate,
        "min_object_distance": min_object_distance,
        "final_object_distance": final_object_distance,
        "min_bowl_distance_while_holding": min_bowl_distance_while_holding,
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
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--mlp_ratio", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--dataset_len", type=int, default=100_000)
    parser.add_argument("--val_len", type=int, default=2048)
    parser.add_argument("--cache_samples", type=int, default=0, help="Pre-render this many train samples as uint8 tensors; 0 disables cache.")
    parser.add_argument("--cache_val_samples", type=int, default=1024, help="Pre-render this many validation samples as uint8 tensors; 0 disables cache.")
    parser.add_argument("--action_chunk_size", type=int, default=8, help="Number of future expert actions predicted from each current observation.")
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
    parser.add_argument(
        "--eval_seed_mode",
        choices=("fixed", "step"),
        default="fixed",
        help="Use fixed rollout scenes at every checkpoint, or vary them by step for broader sampling.",
    )
    parser.add_argument("--gif_episodes", type=int, default=2)
    parser.add_argument("--joint_dim_weight", type=float, default=2.0)
    parser.add_argument("--magnet_dim_weight", type=float, default=1.0)
    parser.add_argument("--magnet_event_weight", type=float, default=3.0)
    parser.add_argument("--joint_direction_weight", type=float, default=0.0)
    parser.add_argument("--loss_type", choices=("mse", "huber", "l1"), default="mse")
    parser.add_argument("--huber_delta", type=float, default=0.25)
    parser.add_argument("--gif_render_every", type=int, default=5)
    parser.add_argument("--temporal_ensemble", action="store_true", help="Average overlapping predicted action chunks during rollout eval.")
    parser.add_argument("--ensemble_decay", type=float, default=0.01)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--out_dir", default="runs/v0")
    args = parser.parse_args()

    device = pick_device(args.device)
    arm_cfg = ArmConfig()
    policy_cfg = PolicyConfig(
        image_size=arm_cfg.world_size,
        patch_size=args.patch_size,
        hidden=args.hidden,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        action_chunk_size=args.action_chunk_size,
    )
    train_dataset = ExpertTransitionDataset(
        args.dataset_len,
        cfg=arm_cfg,
        seed=1234,
        cache_samples=args.cache_samples,
        event_sample_prob=args.event_sample_prob,
        recovery_noise_prob=args.recovery_noise_prob,
        recovery_noise_steps=args.recovery_noise_steps,
        action_chunk_size=args.action_chunk_size,
    )
    val_dataset = ExpertTransitionDataset(
        args.val_len,
        cfg=arm_cfg,
        seed=900_000,
        cache_samples=args.cache_val_samples,
        event_sample_prob=args.event_sample_prob,
        recovery_noise_prob=0.0,
        recovery_noise_steps=0,
        action_chunk_size=args.action_chunk_size,
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

    log_line(f"device: {device}")
    log_line(f"params: {count_parameters(model):,}")
    log_line(f"continuous action: {ACTION_LABELS}")
    log_line(f"train samples: {len(train_dataset):,} | val samples: {len(val_dataset):,}")
    log_line(json.dumps({"policy": asdict(policy_cfg), "arm": asdict(arm_cfg)}, indent=2))

    start = time.time()
    ema_loss = None
    model.train()
    for step in tqdm(range(1, args.steps + 1), dynamic_ncols=True, disable=args.no_progress_bar):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        pred = model(batch["image"], batch["robot"])
        loss = action_loss(
            pred,
            batch["action"],
            args.joint_dim_weight,
            args.magnet_dim_weight,
            args.magnet_event_weight,
            args.joint_direction_weight,
            args.loss_type,
            args.huber_delta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            joint_error = (pred[..., :2] - batch["action"][..., :2]).abs().mean()
            grip_error = (pred[..., 2] - batch["action"][..., 2]).abs().mean()
            magnet_active = (batch["action"][..., 2].abs() > 0.5).float().mean()
        loss_value = float(loss.detach().cpu())
        joint_score = float((1.0 - joint_error.detach().cpu()).clamp(0.0, 1.0))
        grip_value = float(grip_error.detach().cpu())
        event_value = float(magnet_active.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_value

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            log_line(
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
                args.joint_direction_weight,
                args.loss_type,
                args.huber_delta,
            )
            gif_dir = out_dir / "policy_gifs" / f"step_{step:06d}" if args.gif_episodes > 0 else None
            if gif_dir is not None:
                gif_dir.mkdir(parents=True, exist_ok=True)
            rollout_seed = args.rollout_seed if args.eval_seed_mode == "fixed" else args.rollout_seed + step * 100
            roll = rollout_eval(
                model,
                arm_cfg,
                device,
                args.rollout_episodes,
                rollout_seed,
                gif_dir,
                args.gif_episodes,
                args.gif_render_every,
                args.temporal_ensemble,
                args.ensemble_decay,
            )
            log_line(
                f"eval {step:5d} | val_loss {val:.4f} | success {roll['success_rate']:.3f} | "
                f"picked {roll['picked_rate']:.3f} | placed {roll['placed_rate']:.3f} | "
                f"holding_end {roll['held_rate']:.3f} | min_obj {roll['min_object_distance']:.1f}px | "
                f"avg_steps {roll['avg_steps']:.1f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, policy_cfg, arm_cfg, out_dir, step)


if __name__ == "__main__":
    main()

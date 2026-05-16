#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence

from vla_arm.eval import load_policy, pick_device, rollout


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    if path.stem.endswith("last"):
        return 10**12
    return -1


def gif_summary_frames(path: Path, frames_per_gif: int) -> list[Image.Image]:
    image = Image.open(path)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"No frames in {path}")
    if frames_per_gif <= 1:
        idxs = [len(frames) - 1]
    else:
        idxs = [round(i * (len(frames) - 1) / (frames_per_gif - 1)) for i in range(frames_per_gif)]
    return [frames[idx].copy() for idx in idxs]


def make_contact_sheet(rows: list[tuple[str, Path]], out_path: Path, frames_per_gif: int) -> None:
    rendered_rows = []
    for label, gif_path in rows:
        frames = gif_summary_frames(gif_path, frames_per_gif)
        for idx, frame in enumerate(frames):
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 130, 20), fill=(0, 0, 0))
            draw.text((5, 5), f"{label} / {idx + 1}", fill=(255, 255, 0))
        rendered_rows.append(frames)

    cell_w, cell_h = rendered_rows[0][0].size
    canvas = Image.new("RGB", (frames_per_gif * cell_w, len(rendered_rows) * cell_h), (18, 18, 18))
    for row_idx, frames in enumerate(rendered_rows):
        for col_idx, frame in enumerate(frames):
            canvas.paste(frame, (col_idx * cell_w, row_idx * cell_h))
    canvas.save(out_path)


def mean_bool(results: list[dict[str, object]], key: str) -> float:
    return sum(bool(item[key]) for item in results) / max(1, len(results))


def mean_float(results: list[dict[str, object]], key: str) -> float:
    return sum(float(item[key]) for item in results) / max(1, len(results))


def mean_optional_float(results: list[dict[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in results if item[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-seed inference across VLA-Arm checkpoints.")
    parser.add_argument("--run_dir", default="runs/v4_chunk8")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument("--frames_per_gif", type=int, default=4)
    parser.add_argument("--temporal_ensemble", action="store_true")
    parser.add_argument("--ensemble_decay", type=float, default=0.01)
    parser.add_argument("--ensemble_gripper", action="store_true", help="Also average magnet commands during temporal ensembling.")
    parser.add_argument("--no_reset_ensemble_on_gripper_change", action="store_true")
    parser.add_argument("--include_last", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = sorted(run_dir.glob("policy_step_*.pt"), key=checkpoint_step)
    if args.include_last:
        last = run_dir / "policy_last.pt"
        if last.exists():
            checkpoints.append(last)
    if not checkpoints:
        raise SystemExit(f"No policy checkpoints found in {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "fixed_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    metrics = []
    sheet_rows = []
    for ckpt_path in checkpoints:
        label = ckpt_path.stem.replace("policy_", "")
        model, cfg = load_policy(str(ckpt_path), device)
        ckpt_dir = out_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for episode in range(args.episodes):
            render_dir = ckpt_dir if episode == 0 else None
            results.append(
                rollout(
                    model,
                    cfg,
                    args.seed + episode,
                    device,
                    render_dir=render_dir,
                    render_every=args.render_every,
                    temporal_ensemble=args.temporal_ensemble,
                    ensemble_decay=args.ensemble_decay,
                    ensemble_gripper=args.ensemble_gripper,
                    reset_ensemble_on_gripper_change=not args.no_reset_ensemble_on_gripper_change,
                )
            )
        success = mean_bool(results, "success")
        placed = mean_bool(results, "placed")
        holding = mean_bool(results, "holding")
        picked = mean_bool(results, "picked_once")
        released = mean_bool(results, "released_once")
        avg_steps = sum(int(item["steps"]) for item in results) / max(1, len(results))
        row = {
            "checkpoint": str(ckpt_path),
            "label": label,
            "episodes": args.episodes,
            "success_rate": success,
            "picked_rate": picked,
            "released_rate": released,
            "placed_rate": placed,
            "holding_rate": holding,
            "avg_min_object_distance": mean_float(results, "min_object_distance"),
            "avg_final_object_distance": mean_float(results, "final_object_distance"),
            "avg_min_bowl_distance": mean_float(results, "min_bowl_distance"),
            "avg_min_bowl_distance_while_holding": mean_optional_float(results, "min_bowl_distance_while_holding"),
            "avg_final_bowl_distance": mean_float(results, "final_bowl_distance"),
            "avg_steps": avg_steps,
        }
        metrics.append(row)
        print(json.dumps(row))
        gif_path = ckpt_dir / f"episode_{args.seed}.gif"
        if gif_path.exists():
            sheet_rows.append((label, gif_path))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    if sheet_rows:
        sheet_path = out_dir / "contact_sheet.png"
        make_contact_sheet(sheet_rows, sheet_path, args.frames_per_gif)
        print(sheet_path)
    print(metrics_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vla_arm.env import ArmConfig, apply_action, expert_action, is_success, make_scene
from vla_arm.render import render_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Render VLA-Arm continuous pick-and-place examples")
    parser.add_argument("--out_dir", default="runs/examples")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gif_every", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ArmConfig()

    for idx in range(args.count):
        state = make_scene(args.seed + idx, cfg)
        frames = [render_state(state, cfg)]
        pick_step = None
        release_step = None
        for step in range(cfg.max_steps):
            was_holding = state.holding
            action = expert_action(state, cfg)
            state = apply_action(state, action, cfg)
            if state.holding and not was_holding and pick_step is None:
                pick_step = step + 1
            if not state.holding and was_holding and release_step is None:
                release_step = step + 1
            if (step + 1) % args.gif_every == 0:
                frames.append(render_state(state, cfg))
            if is_success(state, cfg):
                break

        frames.append(render_state(state, cfg))
        png_path = out_dir / f"scene_{idx:03d}.png"
        gif_path = out_dir / f"expert_{idx:03d}.gif"
        frames[0].save(png_path)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=65, loop=0)
        print(
            png_path,
            "|",
            gif_path,
            "| success:",
            is_success(state, cfg),
            "| steps:",
            state.step_count,
            "| pick:",
            pick_step,
            "| release:",
            release_step,
        )


if __name__ == "__main__":
    main()

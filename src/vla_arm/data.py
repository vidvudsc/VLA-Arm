from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .env import ArmConfig, apply_action, expert_action, is_success, make_scene, state_vector
from .render import image_to_tensor, image_to_uint8_tensor, render_state


class ExpertTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        length: int = 100_000,
        cfg: ArmConfig | None = None,
        seed: int = 1234,
        rollout_prefix_max: int = 120,
        cache_samples: int = 0,
        event_sample_prob: float = 0.35,
        recovery_noise_prob: float = 0.35,
        recovery_noise_steps: int = 5,
    ):
        self.cfg = cfg or ArmConfig()
        self.seed = int(seed)
        self.rollout_prefix_max = int(rollout_prefix_max)
        self.event_sample_prob = float(event_sample_prob)
        self.recovery_noise_prob = float(recovery_noise_prob)
        self.recovery_noise_steps = int(recovery_noise_steps)
        self.cache: list[dict[str, torch.Tensor]] | None = None
        if cache_samples > 0:
            self.length = int(min(length, cache_samples))
            self.cache = [self._make_sample(idx, cache_uint8=True) for idx in range(self.length)]
        else:
            self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache is not None:
            item = self.cache[int(idx) % len(self.cache)]
            return {
                "image": item["image"].float() / 255.0,
                "robot": item["robot"],
                "action": item["action"],
            }
        return self._make_sample(idx, cache_uint8=False)

    def _make_sample(self, idx: int, cache_uint8: bool) -> dict[str, torch.Tensor]:
        seed = self.seed + int(idx)
        rng = random.Random(seed)
        state = make_scene(seed, self.cfg)
        trajectory = []
        for _ in range(self.rollout_prefix_max + 1):
            action = expert_action(state, self.cfg)
            trajectory.append((state, action))
            if is_success(state, self.cfg):
                break
            state = apply_action(state, action, self.cfg)

        # Oversample only state-changing gripper decisions: first pickup and final release.
        # Holding movement now has magnet target +1, but it is not rare enough to need special sampling.
        event_indices = []
        for i, (candidate_state, action) in enumerate(trajectory):
            magnet = float(action[2])
            if magnet > 0.5 and not candidate_state.holding:
                event_indices.append(i)
            elif magnet < -0.5 and candidate_state.holding:
                event_indices.append(i)
        if event_indices and rng.random() < self.event_sample_prob:
            chosen = rng.choice(event_indices)
        else:
            chosen = rng.randrange(len(trajectory))
        state, action = trajectory[chosen]

        if rng.random() < self.recovery_noise_prob and not is_success(state, self.cfg):
            for _ in range(rng.randint(1, max(1, self.recovery_noise_steps))):
                noisy_action = np.array(
                    [
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                        1.0 if state.holding else 0.0,
                    ],
                    dtype=np.float32,
                )
                state = apply_action(state, noisy_action, self.cfg)
            action = expert_action(state, self.cfg)

        rendered = render_state(state, self.cfg)
        image = image_to_uint8_tensor(rendered) if cache_uint8 else image_to_tensor(rendered)
        robot = torch.from_numpy(state_vector(state, self.cfg))
        return {
            "image": image,
            "robot": robot,
            "action": torch.from_numpy(action),
        }

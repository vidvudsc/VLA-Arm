from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .env import ArmConfig, apply_action, expert_action, is_success, make_scene, state_vector
from .render import image_to_tensor, image_to_uint8_tensor, render_state


def expert_action_chunk(state, cfg: ArmConfig, action_chunk_size: int, first_action: np.ndarray | None = None) -> np.ndarray:
    actions = []
    chunk_state = state
    for chunk_idx in range(action_chunk_size):
        if chunk_idx == 0 and first_action is not None:
            action = first_action
        else:
            action = expert_action(chunk_state, cfg)
        actions.append(action)
        if not is_success(chunk_state, cfg):
            chunk_state = apply_action(chunk_state, action, cfg)
    return np.stack(actions).astype(np.float32)


def render_sample(state, cfg: ArmConfig, action_chunk: np.ndarray, cache_uint8: bool) -> dict[str, torch.Tensor]:
    rendered = render_state(state, cfg)
    image = image_to_uint8_tensor(rendered) if cache_uint8 else image_to_tensor(rendered)
    robot = torch.from_numpy(state_vector(state, cfg))
    return {
        "image": image,
        "robot": robot,
        "action": torch.from_numpy(action_chunk),
    }


class ExpertTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        length: int = 100_000,
        cfg: ArmConfig | None = None,
        seed: int = 1234,
        rollout_prefix_max: int = 120,
        cache_samples: int = 0,
        event_sample_prob: float = 0.35,
        release_event_multiplier: int = 1,
        recovery_noise_prob: float = 0.35,
        recovery_noise_steps: int = 5,
        action_chunk_size: int = 1,
    ):
        self.cfg = cfg or ArmConfig()
        self.seed = int(seed)
        self.rollout_prefix_max = int(rollout_prefix_max)
        self.event_sample_prob = float(event_sample_prob)
        self.release_event_multiplier = max(1, int(release_event_multiplier))
        self.recovery_noise_prob = float(recovery_noise_prob)
        self.recovery_noise_steps = int(recovery_noise_steps)
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")
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
                event_indices.extend([i] * self.release_event_multiplier)
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

        action_chunk = expert_action_chunk(state, self.cfg, self.action_chunk_size, first_action=action)
        return render_sample(state, self.cfg, action_chunk, cache_uint8)


class ExpertEpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        length: int = 100_000,
        episode_count: int = 200,
        cfg: ArmConfig | None = None,
        seed: int = 1234,
        cache_samples: int = 0,
        event_sample_prob: float = 0.25,
        release_event_multiplier: int = 1,
        recovery_noise_prob: float = 0.0,
        recovery_noise_steps: int = 5,
        action_chunk_size: int = 32,
    ):
        self.cfg = cfg or ArmConfig()
        self.seed = int(seed)
        self.episode_count = int(episode_count)
        self.event_sample_prob = float(event_sample_prob)
        self.release_event_multiplier = max(1, int(release_event_multiplier))
        self.recovery_noise_prob = float(recovery_noise_prob)
        self.recovery_noise_steps = int(recovery_noise_steps)
        self.action_chunk_size = int(action_chunk_size)
        if self.episode_count < 1:
            raise ValueError("episode_count must be at least 1")
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1")

        self.episodes = [self._make_episode(self.seed + episode_idx) for episode_idx in range(self.episode_count)]
        self.event_locations = [
            (episode_idx, step_idx)
            for episode_idx, episode in enumerate(self.episodes)
            for step_idx in episode["event_indices"]
        ]
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

    def _make_episode(self, seed: int) -> dict[str, object]:
        state = make_scene(seed, self.cfg)
        states = []
        actions = []
        event_indices = []
        for _ in range(self.cfg.max_steps + 1):
            action = expert_action(state, self.cfg)
            states.append(state)
            actions.append(action)
            magnet = float(action[2])
            if magnet > 0.5 and not state.holding:
                event_indices.append(len(states) - 1)
            elif magnet < -0.5 and state.holding:
                event_indices.extend([len(states) - 1] * self.release_event_multiplier)
            if is_success(state, self.cfg):
                break
            state = apply_action(state, action, self.cfg)
        return {"states": states, "actions": actions, "event_indices": event_indices}

    def _choose_location(self, rng: random.Random) -> tuple[int, int]:
        if self.event_locations and rng.random() < self.event_sample_prob:
            return rng.choice(self.event_locations)
        episode_idx = rng.randrange(len(self.episodes))
        episode = self.episodes[episode_idx]
        step_idx = rng.randrange(len(episode["states"]))
        return episode_idx, step_idx

    def _make_sample(self, idx: int, cache_uint8: bool) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed * 1_000_003 + int(idx))
        episode_idx, step_idx = self._choose_location(rng)
        episode = self.episodes[episode_idx]
        state = episode["states"][step_idx]
        action = episode["actions"][step_idx]

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
            action_chunk = expert_action_chunk(state, self.cfg, self.action_chunk_size, first_action=action)
        else:
            actions = []
            chunk_state = state
            for offset in range(self.action_chunk_size):
                action_idx = step_idx + offset
                if action_idx < len(episode["actions"]):
                    chunk_action = episode["actions"][action_idx]
                else:
                    chunk_action = expert_action(chunk_state, self.cfg)
                actions.append(chunk_action)
                if not is_success(chunk_state, self.cfg):
                    chunk_state = apply_action(chunk_state, chunk_action, self.cfg)
            action_chunk = np.stack(actions).astype(np.float32)

        return render_sample(state, self.cfg, action_chunk, cache_uint8)

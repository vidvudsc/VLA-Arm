# VLA-Arm

A tiny local VLA playground for a fixed-base 2D robot arm.

Current task:

1. Move the arm tip to the red object.
2. Turn on the magnetic gripper to pick it up.
3. Move the held object to the bowl.
4. Turn off the magnet to release it into the bowl.

There is no language input in this version. The model is not given hidden object or bowl coordinates. It receives only:

```text
RGB image
joint-angle proprioception
magnetic gripper / holding state
```

The policy is a small transformer over 16x16 image patches plus one robot-state token. It predicts a short chunk of continuous 3D actions:

```text
shoulder velocity in [-1, 1]
elbow velocity in [-1, 1]
magnet command in [-1, 1]
```

With `--action_chunk_size 8`, the model predicts:

```text
current RGB image + current robot state -> next 8 actions
```

Rollout can use temporal ensembling over overlapping chunks with `--temporal_ensemble`.
By default it smooths only joint velocities; the magnet command comes from the
current observation because pickup/release are thresholded state changes. Use
`--ensemble_gripper` only if you explicitly want to smooth magnet commands too.

The magnet command is persistent:

```text
 0 = idle/off
+1 = keep magnet on / hold object
-1 = release object
```

Positive magnet command picks when the tip is close to the object. While holding, the expert keeps the magnet target at `+1` until it is time to release into the bowl.

## Quick Start

```bash
git clone https://github.com/vidvudsc/VLA-Arm.git
cd VLA-Arm
python -m pip install -e .

python scripts/make_examples.py --out_dir runs/examples

python -m vla_arm.train \
  --device mps \
  --steps 2000 \
  --batch_size 64 \
  --dataset_mode episode \
  --episode_count 200 \
  --val_episode_count 40 \
  --action_chunk_size 8 \
  --temporal_ensemble \
  --eval_every 250 \
  --eval_seed_mode fixed \
  --val_batches 16 \
  --rollout_episodes 12 \
  --gif_episodes 2 \
  --cache_val_samples 1024 \
  --out_dir runs/v0

python -m vla_arm.eval \
  --checkpoint runs/v0/policy_last.pt \
  --device mps \
  --episodes 100 \
  --temporal_ensemble \
  --render_dir runs/v0/eval_renders
```

## What Counts As Success

Success means the object has been released inside the bowl. This is behavior cloning from a synthetic expert, not reinforcement learning.

## Training Notes

This is not a pretrained transformer yet. The policy transformer starts from random weights and learns by supervised behavior cloning from the synthetic expert.

Dataset modes:

```text
transition:
  generate one expert rollout per sample, then train on one sampled transition

episode:
  prebuild full synthetic expert episodes, then sample action chunks from those episodes
```

`episode` mode is closer to LeRobot/ACT-style imitation learning because the training distribution comes from a fixed bank of coherent demonstrations.

The training loop tracks:

- weighted train loss
- validation loss on held-out synthetic seeds
- rollout success rate
- pickup/release rate and closest object/bowl distances
- placement rate
- policy GIFs from live rollouts

By default, training uses `--eval_seed_mode fixed`, so every checkpoint is evaluated and rendered on the same rollout seeds. This makes GIFs comparable across training time. Use `--eval_seed_mode step` when you want each checkpoint to sample different rollout scenes.

Pickup/release transitions are still brief, so the loss upweights the magnet output dimension and the dataset deliberately samples those transition moments some of the time via `--event_sample_prob`.

Optional caching:

```bash
--cache_samples 4096
```

pre-renders that many train samples as uint8 tensors. This can help if PIL rendering becomes the bottleneck on MPS, but it uses RAM.

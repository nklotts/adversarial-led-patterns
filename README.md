# Dynamic Physical Adversarial LED Patterns via Reinforcement Learning

Reference implementation for the NeurIPS 2026 paper *Dynamic Physical Adversarial LED Patterns
via Reinforcement Learning*.

**[Paper](https://nklotts.github.io/adversarial-led-patterns/assets/zyryanov-lotts-neurips2026-dynamic-led-adversarial.pdf)**
· **[Project page](https://nklotts.github.io/adversarial-led-patterns/)**

Konstantin Zyryanov<sup>1</sup>, Nikita Lotts<sup>2</sup>
<sup>1</sup>Department of Physics, Lomonosov Moscow State University ·
<sup>2</sup>Faculty of Computer Science, HSE University

---

## Overview

Automated person detection is widely deployed in video surveillance, access control and scene
monitoring, which makes two questions concrete: how reliable are these systems against
perturbations that adapt to the scene, and what recourse does someone who has not consented to
being tracked actually have? Physical adversarial attacks are the standard instrument for the
first question, and almost all of them optimize a single pattern once and then print it, so the
perturbation cannot respond when illumination, distance or viewing angle change.

This repository implements **closed-loop physical adversarial control**: a policy drives a
wearable RGB LED array in response to what its onboard cameras observe, while receiving only
black-box detector confidence during training.

The control loop runs between two machines:

1. A **training host** with a GPU holds the policy and the target detector. It receives the
   onboard camera frames, selects an LED pattern, captures the illuminated garment through a
   separate detector camera, and computes the reward from the detector confidence.
2. A **LED controller** — a single-board computer worn with the garment — captures the onboard
   frames, renders the pattern it receives on the LED matrix, and acknowledges the render.

The two exchange length-prefixed messages over TCP, so one step of the loop is: frames → pattern →
render acknowledgement → detector frame → reward → policy update.

## Scope of this release

This repository releases the method, not a deployable attack. In line with the responsible-release
statement in the paper:

- **No trained policies or checkpoints are published**, and none optimized against a specific
  target.
- **GPIO rendering and onboard camera capture are not implemented.** Both are specific to the LED
  panels and camera modules in use; `server.py` raises `NotImplementedError` for the hardware
  paths and ships a simulation mode instead.
- With simulation enabled, the full protocol and the full training loop run on a single machine
  against synthetic onboard frames, which is sufficient to inspect and extend the method.

A GPU-attached camera is still required to produce a real detector signal. Without one, `train.py`
exits during environment construction.

## Repository layout

```
config.py              Every experiment parameter, in one class
train.py               Training entry point (runs on the training host)
server.py              LED controller service (runs on the single-board computer)

env/
  jacket_env.py        Closed-loop environment: state, reward, LED expansion, transport

models/
  actor.py             Stochastic policy, sigmoid-squashed to the brightness ceiling
  critic.py            Twin Q-networks for clipped double Q-learning
  sac_agent.py         SAC update rule, gradient clipping, checkpointing

utils/
  replay_buffer.py     Fixed-capacity transition buffer
  socket_utils.py      Length-prefixed pickle transport over TCP
  visualization.py     Live OpenCV views of the detector, the pattern and the cameras

docs/                  Source of the project page published with GitHub Pages
```

## Installation

Python 3.10 or newer is required.

The CUDA build of PyTorch is not distributed on PyPI, so install it first, matching the CUDA
toolkit on the training host:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements.txt
```

The detector weights (`yolov8n.pt`) are downloaded automatically by `ultralytics` on first use and
are deliberately not committed.

## Running

Start the LED controller. On the garment's single-board computer, or on the training host itself
for a self-contained run:

```bash
python server.py
```

Then start training on the host:

```bash
python train.py
```

Set `SERVER_HOST` in `config.py` to the controller's address. The default, `127.0.0.1`, keeps both
processes on one machine.

## Configuration

Every parameter lives in `config.TrainingConfig`. The full configuration is echoed to the console
at startup and written into the header of each run log, so a log file alone identifies the run.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `BOARD_H`, `BOARD_W`, `N_BOARDS` | 16, 16, 39 | LED panel geometry; the array holds `BOARD_H · BOARD_W · N_BOARDS` = 9,984 diodes |
| `SUPERPIXEL_SIZE` | 4 | One RGB value drives an `S × S` block of diodes; must divide the panel dimensions |
| `MAX_BRIGHTNESS` | 10 | Per-channel brightness ceiling, out of 255 |
| `DETECTION_THRESHOLD` | 0.9 | Detection confidence the attack is scored against |
| `REWARD_A`, `REWARD_B`, `REWARD_C` | 10, 100, 10 | Weights of the detection, diversity and temporal reward terms |
| `DETECTION_AVG_FRAMES` | 3 | Detector frames averaged per step to damp measurement noise |
| `STATE_NOISE_STD`, `ACTION_NOISE_STD` | 0.05, 0.02 | Exploration noise on the observation and on the action |
| `NUM_EPISODES` | 2000 | Episodes per run; roughly 10–12 s each, so 5–6 hours |

### Action space

Acting on every diode independently gives an action dimension of `3 · N_LED = 29,952`, which
destabilizes the critic. The policy instead emits one RGB triplet per superpixel, and the
environment expands it deterministically over the array. At the defaults this is 624 superpixels,
an action dimension of 1,872 and a 16× reduction against per-diode control. The expansion accounts
for the serpentine wiring of the panels — even rows run left to right, odd rows right to left — so
each superpixel stays spatially contiguous on the garment.

There is no separate intensity channel: brightness is carried by the norm of the RGB triplet, so
colour and brightness are optimized jointly by a single action.

### Reward

```
r = REWARD_A · (DETECTION_THRESHOLD − C_det)
  + REWARD_B · Var(a)
  + REWARD_C · ‖a − a_prev‖²
```

The first term drives detector confidence down; the second and third reward pattern diversity and
temporal variation, which counteract the collapse of policy entropy analyzed in the paper.

## Experimental protocol

The numbers reported in the paper follow this protocol.

| | |
| --- | --- |
| Episodes per run | 2,000 |
| Time per episode | ≈10–12 s, dominated by LED rendering, capture and detector inference rather than gradient updates |
| Wall-clock per run | ≈5–6 h |
| Independent runs per configuration | 15, reported as mean ± standard deviation over run-level averages with 95 % Student-*t* intervals |
| Frames per policy evaluation | 300 |
| Total experimental wall-clock | ≈380–460 h |
| Hardware | Orange Pi 5 Pro for LED control; NVIDIA GTX 1060, 3 GB VRAM, for training and detector inference |
| Checkpoint selection | Lowest mean detector confidence on a held-out validation sequence; the final checkpoint is never used by default |
| Scene variation | An operator varies pose, viewpoint, distance and illumination on a fixed randomization schedule, identical across every compared method |

The bottleneck is the physical interaction loop, not compute.

## Output

| Path | Contents |
| --- | --- |
| `checkpoints/sac_episode_XXXXX.pth` | Networks, optimizer state, entropy coefficient and the reward history, every `SAVE_MODEL_EVERY` episodes |
| `logs/training_YYYYMMDD_HHMMSS.log` | The run configuration followed by per-episode CSV: reward, detection confidence, 100-episode moving averages and the three losses |
| `dataset/episode_XXXXX/` | Onboard frames, detector frame and `metadata.json`, written only when `SAVE_DATASET` is enabled |

All three directories are ignored by git.

## Citation

```bibtex
@inproceedings{zyryanov2026dynamic,
  title     = {Dynamic Physical Adversarial {LED} Patterns via Reinforcement Learning},
  author    = {Zyryanov, Konstantin and Lotts, Nikita},
  booktitle = {Advances in Neural Information Processing Systems 40 (NeurIPS 2026)},
  year      = {2026}
}
```

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE). The paper
itself is distributed under CC BY 4.0 through OpenReview.

## Intended use

This code is released to support robustness evaluation of person detectors under physically
realizable, adaptive perturbations. It is intended for research on adversarial robustness and
defenses. Using it to evade surveillance or safety-critical perception systems in deployment is
outside its intended use.

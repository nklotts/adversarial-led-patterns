"""Training entry point for adversarial LED pattern optimization with SAC.

Exploration uses two independent noise sources, both configured in
``config.TrainingConfig``:

1. State noise (``STATE_NOISE_STD``) is added to the observation *before* it
   reaches the actor, which widens the range of patterns the policy proposes.
   The clean observation, not the perturbed one, is written to the replay
   buffer.
2. Action noise (``ACTION_NOISE_STD``) is added to the actor output, giving a
   fine-grained perturbation around the policy mean before the pattern is
   rendered on the garment.

Run from the repository root:

    python train.py
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import TrainingConfig as cfg
from env import AdversarialJacketEnv
from models import SACAgent

LOG_HEADER = ('episode,reward,detection,avg100_reward,avg100_detection,'
              'critic_loss,actor_loss,alpha_loss\n')


# =============================================================================
# Logging
# =============================================================================
def _config_lines():
    """Yield ``KEY = value`` lines for every public configuration entry."""
    for key, val in sorted(vars(cfg).items()):
        if not key.startswith('_'):
            yield f"{key} = {val}"


def _print_config():
    """Echo the full configuration to the console."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    for line in _config_lines():
        print(f"  {line}")
    print("=" * 70 + "\n")


def _write_config_header(log_file: str):
    """Write the configuration and the CSV header to a fresh log file."""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 70 + "\n")
        for line in _config_lines():
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
        f.write(LOG_HEADER)


def _save_dataset_step(episode, step, raw_images, info, action, detection, reward, env):
    """Persist the frames and metadata of a single training step."""
    ep_dir = os.path.join(cfg.DATASET_DIR, f'episode_{episode:05d}')
    os.makedirs(ep_dir, exist_ok=True)

    cv2.imwrite(os.path.join(ep_dir, 'onboard_front.jpg'), raw_images[0])
    cv2.imwrite(os.path.join(ep_dir, 'onboard_back.jpg'), raw_images[1])
    cv2.imwrite(os.path.join(ep_dir, 'detector_frame.jpg'), info['frame'])

    if cfg.SAVE_VISUALIZATION and env.last_visualization is not None:
        cv2.imwrite(os.path.join(ep_dir, 'detector_overlay.jpg'), env.last_visualization)

    metadata = {
        'episode':              episode,
        'step':                 step,
        'detection_confidence': float(detection),
        'reward':               float(reward),
        'action_mean':          float(action.mean()),
        'action_std':           float(action.std()),
        'superpixel_size':      cfg.SUPERPIXEL_SIZE,
    }
    with open(os.path.join(ep_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Training loop
# =============================================================================
def train():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)
    if cfg.SAVE_DATASET:
        os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    _print_config()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # -- Environment --
    env = AdversarialJacketEnv(cfg)
    if not env.connect_to_controller():
        print("[ERROR] Could not reach the LED controller. Aborting.")
        return

    state_dim  = env.state_dim
    action_dim = env.action_dim

    # -- Agent --
    print("[INFO] Initializing the SAC agent...")
    agent = SACAgent(state_dim, action_dim, device, cfg)
    print("[INFO] SAC agent ready.")

    episode_rewards    = []
    episode_detections = []
    last_critic_loss   = None
    last_actor_loss    = None
    last_alpha_loss    = None

    log_file = os.path.join(
        cfg.LOGS_DIR,
        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    _write_config_header(log_file)

    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70 + "\n")

    state, raw_images = env.reset()
    if state is None:
        print("[ERROR] Could not read the initial state. Aborting.")
        return

    for episode in range(1, cfg.NUM_EPISODES + 1):
        print(f"\n[Episode {episode:05d}/{cfg.NUM_EPISODES}] " + "-" * 40)

        episode_reward    = 0.0
        episode_detection = 0.0

        for step in range(cfg.MAX_STEPS_PER_EPISODE):

            # -- State noise, applied before the actor sees the observation --
            noisy_state = state + np.random.normal(
                0, cfg.STATE_NOISE_STD, size=state.shape)

            action = agent.select_action(noisy_state)

            # -- Action noise, applied after the actor emits the pattern --
            if cfg.ACTION_NOISE_STD > 0:
                action = action + np.random.normal(
                    0, cfg.ACTION_NOISE_STD, size=action.shape)
                action = np.clip(action, 0.0, cfg.MAX_BRIGHTNESS / 255.0)

            reward, done, info = env.step(action)

            if done and not info:
                print("[ERROR] Environment step failed.")
                break

            episode_detection  = info['detection_confidence']
            episode_reward    += reward

            next_state, raw_images = env.reset()
            if next_state is None:
                print("[ERROR] Could not read the next state.")
                done = True
                break

            # The clean state is stored, not the perturbed one.
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done:
                break

            if len(agent.replay_buffer) >= cfg.LEARNING_STARTS:
                c_loss, a_loss, al_loss = agent.update(cfg.BATCH_SIZE)
                if c_loss is not None:
                    last_critic_loss = c_loss
                    last_actor_loss  = a_loss
                    last_alpha_loss  = al_loss
                    if cfg.VERBOSE:
                        print(f"  [update] critic={c_loss:.4f}  "
                              f"actor={a_loss:.4f}  alpha={al_loss:.4f}")

            if cfg.SAVE_DATASET and raw_images is not None:
                _save_dataset_step(episode, step, raw_images, info, action,
                                   episode_detection, reward, env)

            state = next_state
            print(f"  [step {step + 1}] reward={reward:.4f}  "
                  f"detection={episode_detection:.4f}")

        episode_rewards.append(episode_reward)
        episode_detections.append(episode_detection)
        avg_reward    = np.mean(episode_rewards[-100:])
        avg_detection = np.mean(episode_detections[-100:])

        print(f"  reward={episode_reward:.4f}  detection={episode_detection:.4f}  "
              f"avg100_reward={avg_reward:.4f}  "
              f"avg100_detection={avg_detection:.4f}  "
              f"buffer={len(agent.replay_buffer)}")

        if episode % cfg.SAVE_MODEL_EVERY == 0:
            path = os.path.join(cfg.MODELS_DIR, f'sac_episode_{episode:05d}.pth')
            agent.save(path, episode, episode_rewards, episode_detections)
            print(f"[INFO] Checkpoint written: {path}")

        critic_str = f"{last_critic_loss:.4f}" if last_critic_loss is not None else ""
        actor_str  = f"{last_actor_loss:.4f}"  if last_actor_loss  is not None else ""
        alpha_str  = f"{last_alpha_loss:.4f}"  if last_alpha_loss  is not None else ""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{episode},{episode_reward:.4f},{episode_detection:.4f},"
                    f"{avg_reward:.4f},{avg_detection:.4f},"
                    f"{critic_str},{actor_str},{alpha_str}\n")

    env.close()
    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)


if __name__ == '__main__':
    train()

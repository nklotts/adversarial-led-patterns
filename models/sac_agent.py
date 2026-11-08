"""Soft Actor-Critic agent.

Ties together the actor, the twin critics, the replay buffer and the update
rule. Gradient-norm clipping is applied to both the actor and the critics to
contain the Q-drift that noisy physical rewards tend to induce.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.replay_buffer import ReplayBuffer

from .actor import Actor
from .critic import Critic


class SACAgent:
    """Soft Actor-Critic agent.

    Args:
        state_dim: dimensionality of the state.
        action_dim: dimensionality of the action.
        device: torch device the networks live on.
        cfg: a :class:`config.TrainingConfig` instance.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg):
        self.device    = device
        self.gamma     = cfg.GAMMA
        self.tau       = cfg.TAU
        self.grad_clip = cfg.GRAD_CLIP_NORM

        max_action = cfg.MAX_BRIGHTNESS / 255.0

        self.actor         = Actor(state_dim, action_dim, cfg.HIDDEN_DIMS, max_action).to(device)
        self.critic        = Critic(state_dim, action_dim, cfg.HIDDEN_DIMS).to(device)
        self.critic_target = Critic(state_dim, action_dim, cfg.HIDDEN_DIMS).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=cfg.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        self.target_entropy  = -action_dim
        self.log_alpha       = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.ALPHA_LR)

        self.replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            t      = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(t, deterministic)
        return action.cpu().numpy()[0]

    def update(self, batch_size: int):
        """Sample a batch and update every network.

        Returns:
            ``(critic_loss, actor_loss, alpha_loss)``, or ``(None, None, None)``
            while the replay buffer holds fewer than ``batch_size`` transitions.
        """
        if len(self.replay_buffer) < batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # -- Critic update --
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # -- Actor update --
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # -- Entropy coefficient update --
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -- Soft update of the target critic --
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def save(self, path: str, episode: int,
             episode_rewards: list, episode_detections: list):
        """Write a checkpoint that is sufficient to resume training."""
        torch.save({
            'episode':                     episode,
            'actor_state_dict':            self.actor.state_dict(),
            'critic_state_dict':           self.critic.state_dict(),
            'critic_target_state_dict':    self.critic_target.state_dict(),
            'actor_optimizer_state_dict':  self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha':                   self.log_alpha,
            'episode_rewards':             episode_rewards,
            'episode_detections':          episode_detections,
        }, path)

    def load(self, path: str) -> dict:
        """Restore networks and optimizers; returns the full checkpoint dict."""
        # weights_only=False is required because a checkpoint also carries the
        # optimizer state and the reward history, not just tensors. Only load
        # checkpoints you produced yourself.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        return checkpoint

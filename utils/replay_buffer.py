"""Experience replay buffer for off-policy SAC training."""

import random
from collections import deque


class ReplayBuffer:
    """Fixed-capacity circular buffer of ``(s, a, r, s', done)`` transitions.

    Args:
        capacity: maximum number of transitions retained; the oldest are
            evicted first once the buffer is full.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)

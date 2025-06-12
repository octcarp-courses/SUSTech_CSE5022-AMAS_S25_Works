import random

from collections import namedtuple, deque

# state: 1 x obs_dim
# action: 1 x 1
# reward: 1 x 1
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int = 10_000) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Not enough {len(self.memory)} samples for batch size: {batch_size}"
            )
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

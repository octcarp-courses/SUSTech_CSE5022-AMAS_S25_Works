import math
from dataclasses import dataclass, asdict


@dataclass
class BaseAgentConfig:
    # env info
    obs_dim: int = None
    # For DQN, this is num_actions. For CQL, this is total joint discrete actions.
    act_dim: int = None
    hidden_dim: int = 128
    # training
    batch_size: int = 128
    lr: float = 1e-4
    grad_clip_value: float = 100
    # gamma: discount factor
    gamma: float = 0.99
    # epsilon: exploration probability
    eps_start: float = 0.9
    eps_decay: float = 0.95
    eps_min: float = 0.01
    # replay memory
    mem_size: int = 10_000

    def validate(self) -> None:
        assert self.obs_dim is not None, "obs_dim must be set"
        assert self.act_dim is not None, "act_dim must be set"

    def to_dict(self) -> dict:
        return asdict(self)

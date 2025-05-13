import torch
from torch import nn

from ._network import DQN
from ._memory import ReplayMemory

from dataclasses import dataclass, asdict


@dataclass
class BaseAgentConfig:
    # env info
    obs_dim: int = None
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
        assert (self.obs_dim is not None)
        assert (self.act_dim is not None)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseAgent:
    def __init__(self, sid: str, config: BaseAgentConfig, act_sampler: callable, device=None):
        self.sid: str = sid
        self.config: BaseAgentConfig = config
        self.config.validate()
        self.act_sampler: callable = act_sampler  # action sampler function
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # DQN
        self.replay_memory: ReplayMemory = ReplayMemory(capacity=self.config.mem_size)
        self.eps: float = self.config.eps_start

        self.policy_net: DQN = DQN(config.obs_dim, config.act_dim, config.hidden_dim).to(self.device)
        self.target_net: DQN = DQN(config.obs_dim, config.act_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # opt & loss criterion
        self.opt: torch.optim.Optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.lr,
                                                            amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: torch.Tensor, **kwargs):
        return self._select_action_eps(state, dqn=self.policy_net, **kwargs)

    def select_action_greedy(self, state: torch.Tensor, dqn: nn.Module, **kwargs):
        return self._select_action_eps(state, dqn=dqn, eps=0, **kwargs)

    def _select_action_eps(self, state, dqn, eps=-1, **kwargs):
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def memorize(self, *args) -> None:
        self.replay_memory.push(*args)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_eps(self) -> None:
        self.eps = max(self.config.eps_min, self.eps * self.config.eps_decay)

import torch
import torch.nn as nn


def _init_weights(layer: nn.Module, nonlinearity: str) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DQN(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden_dims: list[int]) -> None:
        super(DQN, self).__init__()
        self.obs_dim: int = n_obs
        self.act_dim: int = n_act
        self.hidden_dims: list[int] = hidden_dims

        # model
        layers: list[nn.Module] = []
        input_dim = self.obs_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, self.act_dim))

        self.network: nn.Sequential = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # re-initialize learnable parameters
        for layer in self.network:
            _init_weights(layer, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.network(x)
        return res

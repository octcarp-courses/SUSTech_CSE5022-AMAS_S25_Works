from dataclasses import dataclass

import torch
from torch import nn
import numpy as np

from ._base import Transition, BaseAgentConfig, BaseAgent, DQN


@dataclass
class IqlAgentConfig(BaseAgentConfig):
    pass


class IqlAgent(BaseAgent):
    def __init__(
        self, sid: str, config: IqlAgentConfig, act_sampler: callable, device=None
    ) -> None:
        super().__init__(sid, config, act_sampler, device)
        self.config: IqlAgentConfig = config

    def _select_action_eps(
        self, state: torch.Tensor, dqn: DQN, eps: float = -1, **kwargs
    ) -> torch.Tensor:
        """
        input shape: 1 x obs_dim
        output shape: 1 x 1
        """
        if eps == -1:
            eps = self.eps
        if np.random.random() < eps:
            sample_res = torch.tensor(
                [[self.act_sampler()]], device=self.device, dtype=torch.long
            )
            return sample_res
        with torch.no_grad():
            q_values: torch.Tensor = dqn(state)
            sel_res = q_values.argmax(dim=1).reshape(1, 1)
        return sel_res

    def train(self) -> None:
        if len(self.replay_memory) < self.config.batch_size:
            return

        # sample a batch of transitions
        transitions = self.replay_memory.sample(self.config.batch_size)
        # batch of transitions => 1 transition with batch-array values
        batch = Transition(*zip(*transitions))

        # mask of non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )  # BS' x 1(bool)
        non_final_nxt_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )  # BS' x obs_dim

        state_batch = torch.cat(batch.state)  # BS x obs_dim
        action_batch = torch.cat(batch.action)  # BS x 1
        reward_batch = torch.cat(batch.reward)  # BS x 1

        # Q(s_t)
        q_values_batch: torch.Tensor = self.policy_net(state_batch)  # BS x act_dim
        # select the q values of the corresponding actions to get Q(s_t, a)
        # torch.gather: pick values from q_values_batch and place them at dim=1 (second dimension)
        # values are the q values indexed by action_batch, for example, if action=1 in batch=0,
        # then pick q_values_batch[0][1] and place it to res[0][0]
        state_action_q_values = q_values_batch.gather(1, action_batch)  # BS x 1

        # max_a Q(s_{t+1}, a)
        next_state_best_q_values = torch.zeros(
            self.config.batch_size, device=self.device
        )  # BS
        with torch.no_grad():
            # BS' x obs_dim ==target_net==> BS' x act_dim ==max(1).values==> BS'
            # use the mask to fill the BS' values into their positions
            next_state_best_q_values[non_final_mask] = (
                self.target_net(non_final_nxt_states).max(1).values
            )
        # V(s_{t+1})
        expected_state_action_q_values = reward_batch + (
            self.config.gamma
            * next_state_best_q_values.reshape(self.config.batch_size, 1)
        )  # BS x 1

        # loss
        loss = self.criterion(state_action_q_values, expected_state_action_q_values)

        # optimize
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(
            self.policy_net.parameters(), self.config.grad_clip_value
        )
        self.opt.step()

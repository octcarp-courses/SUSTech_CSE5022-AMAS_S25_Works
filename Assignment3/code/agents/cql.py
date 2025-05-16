import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from ._base import Transition, BaseAgent, BaseAgentConfig


@dataclass
class CqlAgentConfig(BaseAgentConfig):
    # obs/act dim for each agent
    obs_dims: dict[str, int] = None
    act_dims: dict[str, int] = None

    def validate(self) -> None:
        assert self.obs_dims is not None, "obs_dims must be set for CqlAgentConfig"
        assert self.act_dims is not None, "act_dims must be set for CqlAgentConfig"
        # obs_dim and act_dim (for the joint DQN) should be set before validation.
        # This is typically done by calling infer_joint_space().
        super().validate()

    def infer_joint_space(self) -> "CqlAgentConfig":
        self.obs_dim = sum(self.obs_dims.values())
        self.act_dim = math.prod(self.act_dims.values())
        return self


class CqlAgent(BaseAgent):
    def __init__(
        self, sid: str, config: CqlAgentConfig, act_sampler: callable, device=None
    ) -> None:
        super().__init__(sid, config, act_sampler, device)
        self.config: CqlAgentConfig = config

    def agent_keys(self) -> list[str]:
        return list(self.config.act_dims.keys())

    def n_agents(self) -> int:
        return len(self.agent_keys())

    def get_masked_joint_obs(
        self, observations: torch.Tensor | dict, done_agents: dict[str, bool] = None
    ) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            # transformation already finished
            return observations
        return torch.concat(
            [
                (
                    torch.zeros(self.config.obs_dims[cur_agent], device=self.device)
                    if cur_done
                    else torch.tensor(
                        observations[cur_agent], dtype=torch.float32, device=self.device
                    )
                )
                for cur_agent, cur_done in done_agents.items()
            ]
        ).reshape(1, -1)

    def decode_joint_action(self, joint_action: int) -> dict[str, int | None]:
        """
        Assume each agent's action is {0, 1, ...}.
        Then the first action (joint_action=0) is all agents having action=0,
        the second action (joint_action=1) is all agents except the last one (action=1) having action=0.

        output: dict[str, int | None]
        """
        actions = []
        for act_dim in reversed(self.config.act_dims.values()):
            actions.append(joint_action % act_dim)
            joint_action //= act_dim
        return dict(zip(self.agent_keys(), reversed(actions)))

    def encode_joint_action(self, actions: dict[str, int | None]) -> int:
        """
        Reverse method of decode_joint_action.
        If an agent's action is None, randomly choose one for joint action index calculation (actually not ideal).
        """
        res = 0
        multiplier = 1
        for agent_key, act_dim in reversed(self.config.act_dims.items()):
            action = (
                actions[agent_key]
                if actions[agent_key] is not None
                else random.randint(0, act_dim - 1)
            )
            res += action * multiplier
            multiplier *= act_dim
        return res

    def get_masked_actions(
        self, joint_action: int, done_agents=None
    ) -> dict[str, int | None]:
        """
        Decode the joint action into a list of actions (as tensor).
        Then, apply the mask of finished/done agents if provided.

        output: dict[str, int | None]
        """
        actions = self.decode_joint_action(joint_action)
        if done_agents:
            actions = {
                agent: None if done_agents[agent] else actions[agent]
                for agent in self.agent_keys()
            }
        return actions

    def _select_action_eps(
        self,
        state: torch.Tensor,
        dqn: nn.Module,
        eps: float = -1,
        done_agents: dict[str, bool] = None,
    ) -> dict[str, int | None]:
        """
        input shape: 1 x obs_dim
        [NOTE] output: dict[str, int | None]
        """
        if eps == -1:
            eps = self.eps
        if random.random() < eps:
            # NOTE: here the sampler outputs a dict, rather than one single value
            return self.act_sampler()

        with torch.no_grad():
            q_values: torch.Tensor = dqn(state)
            joint_act = q_values.argmax(dim=1).item()
        return self.get_masked_actions(joint_act, done_agents)

    def train(self) -> None:
        if len(self.replay_memory) < self.config.batch_size:
            return

        # sample a batch of transitions
        transitions = self.replay_memory.sample(self.config.batch_size)
        # batch of transitions => 1 transition with batch-array values
        batch = Transition(*zip(*transitions))

        # Q(s_t)
        state_batch = torch.cat(batch.state)  # BS x obs_dim
        action_batch = torch.tensor(
            [[self.encode_joint_action(cur_actions)] for cur_actions in batch.action],
            device=self.device,
            dtype=torch.long,
        )  # BS x 1
        q_values_batch: torch.Tensor = self.policy_net(state_batch)  # BS x act_dim
        state_action_q_values = q_values_batch.gather(1, action_batch)  # BS x 1

        # max_a Q(s_{t+1}, a)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )  # BS' x 1(bool)
        non_final_nxt_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )  # BS' x obs_dim
        reward_batch = torch.cat(batch.reward)  # BS x 1
        next_state_best_q_values = torch.zeros(
            self.config.batch_size, device=self.device
        )  # BS
        with torch.no_grad():
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

from itertools import count
import torch

from agent import IqlAgent, DQN
from pettingzoo.sisl import pursuit_v4

from agent import BaseAgent
from .common import get_agent_wise_cumulative_rewards


def update_agent_dqns(cur_agents:dict[str, IqlAgent], best_mean: float) -> float:
    with torch.no_grad():
        cur_eval_res = eval_agent(
            cur_agents,
            dqns={cur_agent.sid: cur_agent.policy_net for cur_agent in cur_agents.values()},
            n_episodes=10
        )
    avg_eval_res = get_agent_wise_cumulative_rewards(cur_eval_res)
    all_avg_eval_res = sum(avg_eval_res.values()) / len(avg_eval_res)
    # print(f'{all_avg_eval_res} vs {best_mean} = {all_avg_eval_res > best_mean}')
    if all_avg_eval_res > best_mean:
        best_mean = all_avg_eval_res
        for cur_agent in cur_agents.values():
            cur_agent.update_target_network()
    return best_mean


def eval_agent(
    dqn_agents: dict[str, BaseAgent],
    dqns: dict[str, DQN],
    n_episodes: int = 1,
    render: bool = False,
    max_cycles: int = 50,
    env_name: str = "pursuit",
) -> dict[str, list[float]]:
    cumulative_rewards = {dqn_agent.sid: [] for dqn_agent in dqn_agents.values()}
    eval_device = list(dqn_agents.values())[0].device
    num_agents = len(dqn_agents)
    for i in range(n_episodes):
        eval_env = pursuit_v4.parallel_env(
            n_pursuers=num_agents,
            max_cycles=max_cycles,
            render_mode=("human" if render else None),
        )
        states, info = eval_env.reset()
        dones = {dqn_agent_key: False for dqn_agent_key in dqn_agents.keys()}
        states = {
            agent_key: torch.tensor(
                state, dtype=torch.float32, device=eval_device
            ).reshape(
                1, -1
            )  # 1 x 18
            for agent_key, state in states.items()
        }
        rewards = {dqn_agent.sid: 0.0 for dqn_agent in dqn_agents.values()}
        cur_cumulative_rewards = {
            dqn_agent.sid: 0.0 for dqn_agent in dqn_agents.values()
        }
        for t in count():
            actions = {}
            for cur_agent in dqn_agents.values():
                if dones[cur_agent.sid]:
                    continue
                action = cur_agent.select_action_greedy(
                    states[cur_agent.sid], dqns[cur_agent.sid]
                )  # 1 x 1
                actions[cur_agent.sid] = action.item()
            observations, rewards, terminations, truncations, infos = eval_env.step(
                actions
            )
            # update rewards
            for cur_agent in dqn_agents.values():
                if dones[cur_agent.sid]:
                    continue
                cur_cumulative_rewards[cur_agent.sid] += rewards[cur_agent.sid]
            dones = {
                agent_key: (terminated or truncations[agent_key])
                for agent_key, terminated in terminations.items()
            }
            done = all(dones.values())
            if done:
                eval_env.close()
                break
            else:
                next_states = {
                    agent_key: torch.tensor(
                        observation, dtype=torch.float32, device=eval_device
                    ).reshape(1, -1)
                    # 1 x 18
                    for agent_key, observation in observations.items()
                }
            states = next_states
        # append the cumulative rewards for this round
        for cur_agent in dqn_agents.values():
            cumulative_rewards[cur_agent.sid].append(
                cur_cumulative_rewards[cur_agent.sid]
            )
    return cumulative_rewards

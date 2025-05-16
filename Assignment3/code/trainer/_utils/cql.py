from itertools import count

import torch

from agents import CqlAgent, DQN
from pettingzoo.sisl import pursuit_v4

from .common import get_agent_wise_cumulative_rewards


def update_agent_dqns(central_agent: CqlAgent, best_mean: float) -> float:
    with torch.no_grad():
        cur_eval_res = eval_agent(
            central_agent, dqn=central_agent.policy_net, n_episodes=10
        )
    avg_eval_res = get_agent_wise_cumulative_rewards(cur_eval_res)
    all_avg_eval_res = sum(avg_eval_res.values()) / len(avg_eval_res)
    # print(f'{all_avg_eval_res} vs {best_mean} = {all_avg_eval_res > best_mean}')
    if all_avg_eval_res > best_mean:
        best_mean = all_avg_eval_res
        central_agent.update_target_network()
    return best_mean


def eval_agent(
    cql_agent: CqlAgent,
    dqn: DQN,
    n_episodes: int = 1,
    render: bool = False,
    max_cycles: int = 50,
    env_name: str = "pursuit",
) -> dict[str, list[float]]:
    cumulative_rewards = dict([(agent_key, []) for agent_key in cql_agent.agent_keys()])
    eval_device = cql_agent.device
    num_agents = len(cql_agent.agent_keys())
    for i in range(n_episodes):
        eval_env = pursuit_v4.parallel_env(
            n_pursuers=num_agents,
            max_cycles=max_cycles,
            render_mode=("human" if render else None),
        )
        states, info = eval_env.reset()
        dones = {agent_key: False for agent_key in cql_agent.agent_keys()}
        states = cql_agent.get_masked_joint_obs(
            observations=states, done_agents=dones
        )  # 1 x (n_agents*obs_dim)
        rewards = {agent_key: 0.0 for agent_key in cql_agent.agent_keys()}
        cur_cumulative_rewards = {
            agent_key: 0.0 for agent_key in cql_agent.agent_keys()
        }
        for t in count():
            actions = cql_agent.select_action_greedy(
                states, dqn, done_agents=dones
            )  # agent_key => int | None
            observations, rewards, terminations, truncations, infos = eval_env.step(
                actions
            )
            # update rewards
            for agent_key in cql_agent.agent_keys():
                if dones[agent_key]:
                    continue
                cur_cumulative_rewards[agent_key] += rewards[agent_key]
            dones = {
                agent_key: (terminated or truncations[agent_key])
                for agent_key, terminated in terminations.items()
            }
            done = all(dones.values())
            if done:
                eval_env.close()
                break
            else:
                next_states = cql_agent.get_masked_joint_obs(
                    observations=observations, done_agents=dones
                )  # 1 x (n_agents*obs_dim)
            states = next_states
        # append the cumulative rewards for this round
        for agent_key in cql_agent.agent_keys():
            cumulative_rewards[agent_key].append(cur_cumulative_rewards[agent_key])
    return cumulative_rewards

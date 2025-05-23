from itertools import count
import torch
from pettingzoo import ParallelEnv
from pettingzoo.sisl import pursuit_v4

from agents import CqlAgent, DQN
from ._utils import (
    get_agent_wise_cumulative_rewards,
    plot_episodes,
    save_episode_acc_to_csv,
)


def update_agent_dqns(central_agent: CqlAgent, best_mean: float) -> float:
    with torch.no_grad():
        cur_eval_res = eval_agent(
            cql_agent=central_agent, dqn=central_agent.policy_net, n_episodes=10
        )
    avg_eval_res = get_agent_wise_cumulative_rewards(cur_eval_res)
    all_avg_eval_res = sum(avg_eval_res.values()) / len(avg_eval_res)
    # print(f'{all_avg_eval_res} vs {best_mean} = {all_avg_eval_res > best_mean}')
    if all_avg_eval_res > best_mean:
        print(f"{all_avg_eval_res:.5f} vs best: {best_mean:.5f}, update Target DQN")
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


def trainer(
    env: ParallelEnv,
    central_agent: CqlAgent,
    num_episodes: int = 100,
    max_episode_lengths: int = 25,
    dqn_update_freq: int = 25,
) -> None:
    total_steps: int = 0
    best_mean: float = float("-inf")
    episode_means: list[float] = []
    episode_avg_returns_per_agent: dict[str, list[float]] = {
        agent_key: [] for agent_key in central_agent.agent_keys()
    }
    device = central_agent.device
    for episode in range(num_episodes):
        # re-initialize the environment
        states, infos = env.reset()
        dones: dict[str, bool] = {
            agent_key: False for agent_key in central_agent.agent_keys()
        }
        states = central_agent.get_masked_joint_obs(
            observations=states, done_agents=dones
        )  # 1 x (n_agents*obs_dim)
        for t in count():
            actions = central_agent.select_action(
                states, done_agents=dones
            )  # agent_key => int | None
            observations, rewards, terminations, truncations, infos = env.step(actions)
            dones = {
                agent_key: (terminated or truncations[agent_key])
                for agent_key, terminated in terminations.items()
            }
            done = all(dones.values()) or (t >= max_episode_lengths - 1)
            # aggregated reward => sum(rewards)
            aggr_reward = sum(
                [reward for reward in rewards.values() if reward is not None]
            )
            aggr_reward_t = torch.tensor([[aggr_reward]], device=device)  # 1 x 1
            # update memory per agent
            if all(terminations.values()):
                next_states = None
            else:
                next_states = central_agent.get_masked_joint_obs(
                    observations=observations, done_agents=dones
                )
            central_agent.memorize(states, actions, next_states, aggr_reward_t)
            # enter next state
            states = next_states
            # optimize model
            trained = central_agent.train()
            # update target dqn if better results
            if total_steps % dqn_update_freq == 0:
                best_mean = update_agent_dqns(
                    central_agent=central_agent, best_mean=best_mean
                )
            # update eps
            central_agent.update_eps()
            # increase total number of experienced steps
            total_steps += 1
            # episode ends
            if done:
                break
        # post update target network
        best_mean = update_agent_dqns(central_agent=central_agent, best_mean=best_mean)
        # evaluate how well the current policy_net is after this episode
        with torch.no_grad():
            cur_policy_eval_res = eval_agent(
                central_agent, dqn=central_agent.policy_net, n_episodes=10
            )
        cur_policy_agent_wise_mean = get_agent_wise_cumulative_rewards(
            cur_policy_eval_res
        )
        cur_policy_mean = sum(cur_policy_agent_wise_mean.values()) / len(
            cur_policy_agent_wise_mean
        )
        for agent_key in central_agent.agent_keys():
            episode_avg_returns_per_agent[agent_key].append(
                cur_policy_agent_wise_mean[agent_key]
            )
        print(f"Episode {episode}\tAvg return = {cur_policy_mean:.4f};")
        episode_means.append(cur_policy_mean)
        save_episode_acc_to_csv(episode_means, "pu_cql")
        plot_episodes(episode_means)
    plot_episodes(episode_means, clear_after=False)

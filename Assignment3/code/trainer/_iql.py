from itertools import count

import torch
from pettingzoo import ParallelEnv
from pettingzoo.sisl import pursuit_v4

from agents import IqlAgent, DQN
from envs import foraging
from ._utils import (
    get_agent_wise_cumulative_rewards,
    plot_episodes,
    save_episode_acc_to_csv,
)


def update_agent_dqns(
    cur_agents: dict[str, IqlAgent],
    best_mean: float,
    env_name: str = "pursuit",
) -> float:
    with torch.no_grad():
        cur_eval_res = eval_agent(
            dqn_agents=cur_agents,
            dqns={
                cur_agent.sid: cur_agent.policy_net for cur_agent in cur_agents.values()
            },
            n_episodes=10,
            env_name=env_name,
        )
    avg_eval_res = get_agent_wise_cumulative_rewards(cur_eval_res)
    all_avg_eval_res = sum(avg_eval_res.values()) / len(avg_eval_res)
    # print(f'{all_avg_eval_res} vs {best_mean} = {all_avg_eval_res > best_mean}')
    if all_avg_eval_res > best_mean:
        print(f"{all_avg_eval_res:.5f} vs best: {best_mean:.5f}, update Target DQN")
        best_mean = all_avg_eval_res
        for cur_agent in cur_agents.values():
            cur_agent.update_target_network()
    return best_mean


def eval_agent(
    dqn_agents: dict[str, IqlAgent],
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
        if env_name == "foraging":
            eval_env = foraging.parallel_env(
                n_foragers=num_agents,
            )
        else:
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
            ).reshape(1, -1)
            for agent_key, state in states.items()
        }

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


def trainer(
    env: ParallelEnv,
    cur_agents: dict[str, IqlAgent],
    num_episodes: int = 100,
    max_episode_lengths: int = 100,
    dqn_update_freq: int = 25,
    env_name: str = "pursuit",
) -> None:
    total_steps: int = 0
    best_mean: float = float("-inf")
    device = list(cur_agents.values())[0].device
    episode_means: list[float] = []
    episode_avg_returns_per_agent: dict[str, list[float]] = {
        cur_agent.sid: [] for cur_agent in cur_agents.values()
    }
    for episode in range(num_episodes):
        # re-initialize the environment
        states, infos = env.reset()
        dones: dict[str, bool] = {
            cur_agent_key: False for cur_agent_key in cur_agents.keys()
        }
        states = {
            agent_key: torch.tensor(state, dtype=torch.float32, device=device).reshape(
                1, -1
            )
            for agent_key, state in states.items()
        }
        for t in count():
            actions = {}
            for cur_agent in cur_agents.values():
                if dones[cur_agent.sid]:
                    continue
                action = cur_agent.select_action(states[cur_agent.sid])
                actions[cur_agent.sid] = action
            observations, rewards, terminations, truncations, infos = env.step(
                {agent_key: action.item() for agent_key, action in actions.items()}
            )
            dones = {
                agent_key: (terminated or truncations[agent_key])
                for agent_key, terminated in terminations.items()
            }
            done = all(dones.values()) or (t >= max_episode_lengths - 1)
            if done:
                break
            rewards_t = {
                cur_agent.sid: torch.tensor([[rewards[cur_agent.sid]]], device=device)
                for cur_agent in cur_agents.values()
            }

            # update memory per agent
            for cur_agent in cur_agents.values():
                if terminations[cur_agent.sid]:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observations[cur_agent.sid], dtype=torch.float32, device=device
                    ).reshape(1, -1)
                # memorize
                cur_agent.memorize(
                    states[cur_agent.sid],
                    actions[cur_agent.sid],
                    next_state,
                    rewards_t[cur_agent.sid],
                )
                # enter next state
                states[cur_agent.sid] = next_state
                # optimize model
                cur_agent.train()
            # update target dqn if better results
            if total_steps % dqn_update_freq == 0:
                best_mean = update_agent_dqns(cur_agents, best_mean, env_name)
            # update eps
            for cur_agent in cur_agents.values():
                cur_agent.update_eps()
            # increase total number of experienced steps
            total_steps += 1
            # episode ends
            if done:
                break
        # post update target network
        best_mean = update_agent_dqns(cur_agents, best_mean, env_name)
        # evaluate how well the current policy_net is after this episode
        with torch.no_grad():
            cur_policy_eval_res = eval_agent(
                dqn_agents=cur_agents,
                dqns={
                    cur_agent.sid: cur_agent.policy_net
                    for cur_agent in cur_agents.values()
                },
                n_episodes=10,
                env_name=env_name,
            )
        cur_policy_agent_wise_mean = get_agent_wise_cumulative_rewards(
            cur_policy_eval_res
        )
        cur_policy_mean = sum(cur_policy_agent_wise_mean.values()) / len(
            cur_policy_agent_wise_mean
        )
        for cur_agent in cur_agents.values():
            episode_avg_returns_per_agent[cur_agent.sid].append(
                cur_policy_agent_wise_mean[cur_agent.sid]
            )
        print(f"Episode {episode}\tAvg return = {cur_policy_mean:.4f};")
        episode_means.append(cur_policy_mean)
        save_episode_acc_to_csv(episode_means, f"{env_name}_iql")
        plot_episodes(episode_means)
    plot_episodes(episode_means, clear_after=False)

from itertools import count
import torch
from pettingzoo import ParallelEnv

from agent import CqlAgent
from ._utils.common import get_agent_wise_cumulative_rewards
from ._utils.cql import eval_agent, update_agent_dqns
from ._utils.plot import plot_episodes


def trainer(env: ParallelEnv,
            central_agent: CqlAgent,
            num_episodes: int = 100,
            max_episode_lengths: int = 25,
            dqn_update_freq: int = 5,
            ):
    total_steps = 0
    best_mean = float('-inf')
    episode_means: list[float] = []
    episode_avg_returns_per_agent = {agent_key: [] for agent_key in central_agent.agent_keys()}
    for i in range(num_episodes):
        # re-initialize the environment
        states, infos = env.reset()
        device = central_agent.device
        dones = {agent_key: False for agent_key in central_agent.agent_keys()}
        states = central_agent.get_masked_joint_obs(observations=states, done_agents=dones)  # 1 x (n_agents*obs_dim)
        for t in count():
            actions = central_agent.select_action(states, done_agents=dones)  # agent_key => int | None
            observations, rewards, terminations, truncations, infos = env.step(actions)
            dones = {
                agent_key: (terminated or truncations[agent_key])
                for agent_key, terminated in terminations.items()
            }
            done = all(dones.values()) or (t >= max_episode_lengths - 1)
            # aggregated reward => sum(rewards)
            aggr_reward = sum([reward for reward in rewards.values() if reward is not None])
            aggr_reward_t = torch.tensor([[aggr_reward]], device=device)  # 1 x 1
            # update memory per agent
            if all(terminations.values()):
                next_states = None
            else:
                next_states = central_agent.get_masked_joint_obs(observations=observations, done_agents=dones)
            central_agent.memorize(states, actions, next_states, aggr_reward_t)
            # enter next state
            states = next_states
            # optimize model
            trained = central_agent.train()
            # update target dqn if better results
            if total_steps % dqn_update_freq == 0:
                best_mean =  update_agent_dqns(central_agent=central_agent, best_mean=best_mean)
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
                central_agent,
                dqn=central_agent.policy_net,
                n_episodes=10
            )
        cur_policy_agent_wise_mean = get_agent_wise_cumulative_rewards(cur_policy_eval_res)
        cur_policy_mean = sum(cur_policy_agent_wise_mean.values()) / len(cur_policy_agent_wise_mean)
        for agent_key in central_agent.agent_keys():
            episode_avg_returns_per_agent[agent_key].append(cur_policy_agent_wise_mean[agent_key])
        # print(episode_avg_returns_per_agent)
        # print('Episode %d:\tavg return = %.2f; avg return per agent: %s' % (i, cur_policy_mean, cur_policy_agent_wise_mean))
        episode_means.append(cur_policy_mean)
        plot_episodes(episode_means)
    plot_episodes(episode_means, clear_after=False)

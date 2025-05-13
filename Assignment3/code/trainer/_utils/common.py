def get_agent_wise_cumulative_rewards(
    cumulative_rewards: dict[str, list[float]],
) -> dict[str, float]:
    return {
        agent_key: sum(agent_episode_cumulative_rewards)
        / len(agent_episode_cumulative_rewards)
        for agent_key, agent_episode_cumulative_rewards in cumulative_rewards.items()
    }

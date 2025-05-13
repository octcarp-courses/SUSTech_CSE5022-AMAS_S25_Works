from pettingzoo.sisl import pursuit_v4
from agent import (
    IqlAgent,
    IqlAgentConfig,
    CqlAgent,
    CqlAgentConfig,
)
import logging


def main() -> None:
    logging.info("Starting pursuit environment")
    env = pursuit_v4.parallel_env(render_mode="human")
    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    obs_dims: dict[int, object] = {
        agent_key: len(observation) for agent_key, observation in observations.items()
    }
    act_dims: dict[int, object] = {
        agent_key: int(env.action_space(agent_key).n)
        for agent_key in observations.keys()
    }
    cur_agents = {
        agent_key: IqlAgent(
            sid=agent_key,
            config=IqlAgentConfig(
                obs_dim=obs_dims[agent_key],
                act_dim=act_dims[agent_key],
                hidden_dim=128,
                batch_size=64,
                lr=1e-4,
                grad_clip_value=0.5,
                gamma=0.95,
                eps_start=0.9,
                eps_decay=0.95,
                eps_min=0.05,
                mem_size=1_000,
            ),
            act_sampler=env.action_space(agent_key).sample,
        )
        for agent_key in observations.keys()
    }
    print(cur_agents)
    list(cur_agents.values())[0].target_net

    env.close()
    logging.info("Environment closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

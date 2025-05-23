import functools

import numpy as np
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.utils.env import ObsType


def env(**kwargs):
    foraging_env = RawEnv(**kwargs)
    foraging_env = wrappers.AssertOutOfBoundsWrapper(foraging_env)
    foraging_env = wrappers.OrderEnforcingWrapper(foraging_env)
    return foraging_env


parallel_env = parallel_wrapper_fn(env)

from ._render import ForagingRenderer

ACTION_MAP: dict[int, tuple[int, int]] = {
    0: (0, 0),  # Stay
    1: (-1, 0),  # Up
    2: (1, 0),  # Down
    3: (0, -1),  # Left
    4: (0, 1),  # Right
}


class RawEnv(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "foraging",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(
        self,
        x_size: int = 7,
        y_size: int = 7,
        n_foragers: int = 3,
        n_crops: int = 10,
        forager_levels: list[int] | None = None,
        crop_levels: list[int] | None = None,
        max_cycles: int = 100,
        render_mode: str | None = None,
    ) -> None:
        EzPickle.__init__(self)

        if forager_levels is not None and n_foragers != len(forager_levels):
            forager_levels = None
        if crop_levels is not None and n_crops != len(crop_levels):
            crop_levels = None

        self._renderer: ForagingRenderer | None = None
        self.np_random_seed = None
        self.x_size = x_size
        self.y_size = y_size
        self.n_foragers = n_foragers
        self.n_crops = n_crops

        self.forager_levels_config = forager_levels
        self.crop_levels_config = crop_levels
        self.max_level_param: int = 3

        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.possible_agents = [f"forager_{i}" for i in range(n_foragers)]
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        # Internal state variables, initialized in reset()
        self.agent_positions: dict[str, tuple[int, int]] = {}
        self.agent_levels: dict[str, int] = {}
        self.crop_positions: list[tuple[int, int]] = []
        self.crop_levels: list[int] = []
        self.crop_removed: list[bool] = []  # True if harvested

        self._agent_selector: AgentSelector = AgentSelector(
            self.possible_agents
        )  # Will be reinitialized in reset
        self._actions_this_turn: dict[str, int] = {}
        self.current_step: int = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> Box:
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=10, shape=(2, self.x_size, self.y_size), dtype=np.int8)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        # We can seed the action space to make the environment deterministic.
        return Discrete(len(ACTION_MAP), seed=self.np_random_seed)

    def reset(self, seed=None, options=None) -> ObsType:
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        # Unlike gymnasium's Env, the environment is responsible for setting the random seed explicitly.
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self._actions_this_turn.clear()
        self.current_step = 0

        occupied_cells = set()

        def _get_random_level(num: int, max_level: int = 4) -> list[int]:
            max_level = 4 if max_level <= 0 else max_level
            return (
                np.random.randint(0, max_level + 1, size=num).astype(np.int8).tolist()
            )

        def _get_valid_pos(x_size: int, y_size: int) -> tuple[int, int]:
            while True:
                pos_i = (np.random.randint(x_size), np.random.randint(y_size))
                if pos_i not in occupied_cells:
                    return pos_i

        self.agent_positions.clear()
        self.agent_levels.clear()

        f_levels = (
            _get_random_level(self.n_foragers, self.max_level_param)
            if self.forager_levels_config is None
            else self.forager_levels_config
        )
        for i, agent_id in enumerate(self.possible_agents):
            pos = _get_valid_pos(self.x_size, self.y_size)
            self.agent_positions[agent_id] = pos
            occupied_cells.add(pos)
            self.agent_levels[agent_id] = f_levels[i]

        # Initialize crops
        self.crop_positions.clear()
        self.crop_levels.clear()
        self.crop_removed = [False] * self.n_crops
        c_levels = (
            _get_random_level(
                self.n_crops,
                self.max_level_param + 1,
            )
            if self.crop_levels_config is None
            else self.crop_levels_config
        )
        for i in range(self.n_crops):
            pos = _get_valid_pos(self.x_size, self.y_size)
            self.crop_positions.append(pos)
            occupied_cells.add(pos)
        self.crop_levels = c_levels
        return self.observe(self.agents[0])

    def close(self) -> None:
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def observe(self, agent_id: str) -> ObsType:
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up-to-date possible)
        at any time after reset() is called.
        """
        obs_type_grid = np.zeros((self.x_size, self.y_size), dtype=np.int8)
        obs_level_grid = np.zeros((self.x_size, self.y_size), dtype=np.int8)

        for i, crop_pos_list in enumerate(self.crop_positions):
            if not self.crop_removed[i]:
                x, y = crop_pos_list
                obs_type_grid[x, y] = 3
                obs_level_grid[x, y] = self.crop_levels[i]

        for other_agent_id, pos_list in self.agent_positions.items():
            if other_agent_id not in self.agent_levels:
                continue
            x, y = pos_list
            level = self.agent_levels[other_agent_id]
            if other_agent_id == agent_id:
                obs_type_grid[x, y] = 1
            else:
                obs_type_grid[x, y] = 2
            obs_level_grid[x, y] = level

        return np.stack([obs_type_grid, obs_level_grid], axis=0).astype(np.int8)

    def _move_all_agents(self) -> None:
        active_crop_locations = set()
        for i, pos in enumerate(self.crop_positions):
            if not self.crop_removed[i]:
                active_crop_locations.add(tuple(pos))
        for agent_id in self.agents:
            if self._is_invalid_agent(agent_id):
                continue

            agent_action = self._actions_this_turn[agent_id]
            if agent_action == 0:
                continue

            dx, dy = ACTION_MAP[agent_action]
            x, y = self.agent_positions[agent_id]
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < self.x_size and 0 <= new_y < self.y_size:
                if (new_x, new_y) not in active_crop_locations:
                    self.agent_positions[agent_id] = (new_x, new_y)

    def _is_invalid_agent(self, agent_id: str) -> bool:
        return self.terminations[agent_id] or self.truncations[agent_id]

    def step(self, action: int) -> None:
        agent = self.agent_selection
        self.rewards[agent] = 0.0
        if self._is_invalid_agent(agent):
            self._was_dead_step(action)
            self._actions_this_turn[agent] = 0
            return
        self._actions_this_turn[agent] = action
        if self._agent_selector.is_last():
            current_step_rewards = {
                agent_id: -0.01
                for agent_id in self.agents
                if not (self._is_invalid_agent(agent))
            }

            self._move_all_agents()
            for crop_idx in range(self.n_crops):
                if self.crop_removed[crop_idx]:
                    continue
                crop_pos = self.crop_positions[crop_idx]
                crop_level = self.crop_levels[crop_idx]

                adj_agent_level: int = 0
                adj_agent_ids = []

                for agent_id in self.agents:
                    if self._is_invalid_agent(agent):
                        continue

                    agent_pos = self.agent_positions[agent_id]
                    if (
                        abs(agent_pos[0] - crop_pos[0])
                        + abs(agent_pos[1] - crop_pos[1])
                        == 1
                    ):
                        adj_agent_level += self.agent_levels[agent_id]
                        adj_agent_ids.append(agent_id)

                if adj_agent_level >= crop_level >= 0:
                    self.crop_removed[crop_idx] = True
                    base_harvest_reward = 10.0
                    level_bonus = float(crop_level)
                    total_crop_reward = base_harvest_reward + level_bonus

                    if adj_agent_ids:
                        shared_reward = total_crop_reward / len(adj_agent_ids)
                        for a_id in adj_agent_ids:
                            current_step_rewards[a_id] = (
                                current_step_rewards.get(a_id, 0.0) + shared_reward
                            )
                    # Optional: Small global reward for all agents when a crop is harvested
                    # for ag_id in self.agents:
                    #    if not (self.terminations[ag_id] or self.truncations[ag_id]):
                    #        current_step_rewards[ag_id] = current_step_rewards.get(ag_id, 0.0) + 0.5

            for agent_id in self.agents:
                if not self._is_invalid_agent(agent):
                    self.rewards[agent_id] = current_step_rewards.get(agent_id, 0.0)

            self.current_step += 1
            all_crops_harvested = all(self.crop_removed)

            episode_is_over = False
            if all_crops_harvested:
                for agent_id in self.agents:
                    self.terminations[agent_id] = True
                    self.rewards[agent_id] += 20.0
                episode_is_over = True

            if not episode_is_over and self.current_step >= self.max_cycles:
                for agent_id in self.agents:
                    if not self.terminations[agent_id]:
                        self.truncations[agent_id] = True
                episode_is_over = True

            if episode_is_over:
                self.agents = []

            self._actions_this_turn = {}

        for agent_id in self.possible_agents:
            self._cumulative_rewards[agent_id] = self.rewards[agent_id]
            self.infos[agent_id] = {}

        self.agent_selection = self._agent_selector.next()
        self.render()

    def render(self) -> None:
        if self.render_mode is None:
            # gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return None

        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = ForagingRenderer(
                    render_fps=self.metadata["render_fps"],
                    display_name=self.metadata["name"],
                )

            self._renderer.draw_frame(
                xs=self.x_size,
                ys=self.y_size,
                agent_positions=list(self.agent_positions.values()),
                agent_levels=list(self.agent_levels.values()),
                crop_positions=self.crop_positions,
                crop_levels=self.crop_levels,
                crop_removed=self.crop_removed,
            )

        return None

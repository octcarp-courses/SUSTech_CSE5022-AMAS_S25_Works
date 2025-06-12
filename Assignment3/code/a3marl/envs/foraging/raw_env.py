import functools

import numpy as np
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.utils.env import ObsType

from .render import ForagingRenderer


def env(**kwargs):
    foraging_env = RawEnv(**kwargs)
    foraging_env = wrappers.AssertOutOfBoundsWrapper(foraging_env)
    foraging_env = wrappers.OrderEnforcingWrapper(foraging_env)
    return foraging_env


parallel_env = parallel_wrapper_fn(env)

ACTION_MAP: dict[int, tuple[int, int]] = {
    0: (0, 0),  # Stay
    1: (-1, 0),  # Up
    2: (1, 0),  # Down
    3: (0, -1),  # Left
    4: (0, 1),  # Right
}

AGENT_TYPE: int = 1
OTHER_AGENT_TYPE: int = 2
CROP_TYPE: int = 3
PADDING_TYPE: int = -1


class RawEnv(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "foraging",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        x_size: int = 10,
        y_size: int = 8,
        n_foragers: int = 3,
        n_crops: int = 10,
        obs_radius: int = 3,
        forager_levels: list[int] | None = None,
        crop_levels: list[int] | None = None,
        max_cycles: int = 100,
        reward_idx: int = 0,
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
        self.obs_radius = obs_radius

        self.forager_levels_config = forager_levels
        self.crop_levels_config = crop_levels
        self.max_level_param: int = 3

        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.reward_idx = reward_idx

        self.possible_agents = [f"forager_{i}" for i in range(n_foragers)]
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        self.agent_positions: dict[str, tuple[int, int]] = {}
        self.agent_levels: dict[str, int] = {}
        self.crop_positions: list[tuple[int, int]] = []
        self.crop_levels: list[int] = []
        self.crop_removed: list[bool] = []  # True if harvested

        self._agent_selector: AgentSelector = AgentSelector(self.possible_agents)
        self._actions_this_turn: dict[str, int] = {}
        self.current_step: int = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> Box:
        local_dim = 2 * self.obs_radius + 1
        return Box(low=0, high=10, shape=(2, local_dim, local_dim), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> Discrete:
        return Discrete(len(ACTION_MAP), seed=self.np_random_seed)

    def reset(self, seed=None, options=None) -> ObsType:
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
            else self.forager_levels_config.copy()
        )
        for i, a_id in enumerate(self.possible_agents):
            pos = _get_valid_pos(self.x_size, self.y_size)
            self.agent_positions[a_id] = pos
            occupied_cells.add(pos)
            self.agent_levels[a_id] = f_levels[i]

        self.crop_positions.clear()
        self.crop_levels.clear()
        self.crop_removed = [False] * self.n_crops
        c_levels = (
            _get_random_level(
                self.n_crops,
                self.max_level_param + 1,
            )
            if self.crop_levels_config is None
            else self.crop_levels_config.copy()
        )
        for i in range(self.n_crops):
            pos = _get_valid_pos(self.x_size, self.y_size)
            self.crop_positions.append(pos)
            occupied_cells.add(pos)
        self.crop_levels = c_levels
        return self.observe(self.agents[0])

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # alternate： full view of the environment
    # def observe(self, a_id: str) -> ObsType:
    #     obs_type_grid = np.zeros((self.x_size, self.y_size), dtype=np.int8)
    #     obs_level_grid = np.zeros((self.x_size, self.y_size), dtype=np.int8)
    #
    #     for i, crop_pos in enumerate(self.crop_positions):
    #         if not self.crop_removed[i]:
    #             x, y = crop_pos
    #             obs_type_grid[x, y] = self.CROP_TYPE
    #             obs_level_grid[x, y] = self.crop_levels[i]
    #
    #     for other_a_id, pos_list in self.agent_positions.items():
    #         if other_a_id not in self.agent_levels:
    #             continue
    #         x, y = pos_list
    #         level = self.agent_levels[other_a_id]
    #         if other_a_id == a_id:
    #             obs_type_grid[x, y] = self.AGENT_TYPE
    #         else:
    #             obs_type_grid[x, y] = self.OTHER_AGENT_TYPE
    #         obs_level_grid[x, y] = level
    #
    #     return np.stack([obs_type_grid, obs_level_grid], axis=0).astype(np.int8)

    def observe(self, a_id: str) -> ObsType:
        g_obs_type = np.zeros((self.x_size, self.y_size), dtype=np.int8)
        g_obs_level = np.zeros((self.x_size, self.y_size), dtype=np.int8)

        for i, crop_pos in enumerate(self.crop_positions):
            if not self.crop_removed[i]:
                x, y = crop_pos
                if 0 <= x < self.x_size and 0 <= y < self.y_size:
                    g_obs_type[x, y] = CROP_TYPE
                    g_obs_level[x, y] = self.crop_levels[i]

        for other_a_id, pos in self.agent_positions.items():
            x, y = pos
            level = self.agent_levels[other_a_id]

            if 0 <= x < self.x_size and 0 <= y < self.y_size:
                if other_a_id == a_id:
                    g_obs_type[x, y] = AGENT_TYPE
                else:
                    g_obs_type[x, y] = OTHER_AGENT_TYPE
                g_obs_level[x, y] = level

        a_x, a_y = self.agent_positions[a_id]

        g_xmin = a_x - self.obs_radius
        g_xmax = a_x + self.obs_radius + 1
        g_ymin = a_y - self.obs_radius
        g_ymax = a_y + self.obs_radius + 1

        g_xmin_actual = max(0, g_xmin)
        g_xmax_actual = min(self.x_size, g_xmax)
        g_ymin_actual = max(0, g_ymin)
        g_ymax_actual = min(self.y_size, g_ymax)

        extracted_type_patch = g_obs_type[
            g_xmin_actual:g_xmax_actual, g_ymin_actual:g_ymax_actual
        ]
        extracted_level_patch = g_obs_level[
            g_xmin_actual:g_xmax_actual, g_ymin_actual:g_ymax_actual
        ]

        pad_x_before = max(0, -g_xmin)
        pad_x_after = max(0, g_xmax - self.x_size)
        pad_y_before = max(0, -g_ymin)
        pad_y_after = max(0, g_ymax - self.y_size)

        local_obs_type = np.pad(
            extracted_type_patch,
            ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
            mode="constant",
            constant_values=PADDING_TYPE,
        )

        local_obs_level = np.pad(
            extracted_level_patch,
            ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
            mode="constant",
            constant_values=PADDING_TYPE,
        )

        return np.stack([local_obs_type, local_obs_level], axis=0).astype(np.int8)

    def _move_all_agents(self) -> None:
        active_crop_locations = set()
        for i, pos in enumerate(self.crop_positions):
            if not self.crop_removed[i]:
                active_crop_locations.add(tuple(pos))
        for a_id in self.agents:
            if self._is_invalid_agent(a_id):
                continue

            agent_action = self._actions_this_turn[a_id]
            if agent_action == 0:
                continue

            dx, dy = ACTION_MAP[agent_action]
            x, y = self.agent_positions[a_id]
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < self.x_size and 0 <= new_y < self.y_size:
                if (new_x, new_y) not in active_crop_locations:
                    self.agent_positions[a_id] = (new_x, new_y)

    def _is_invalid_agent(self, a_id: str) -> bool:
        return self.terminations[a_id] or self.truncations[a_id]

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
                a_id: -0.1
                for a_id in self.agents
                if not (self._is_invalid_agent(agent))
            }

            self._move_all_agents()
            for crop_idx in range(self.n_crops):
                if self.crop_removed[crop_idx]:
                    continue
                crop_pos = self.crop_positions[crop_idx]
                crop_level = self.crop_levels[crop_idx]

                adj_agent_levels: list[int] = []
                adj_a_ids: list[str] = []

                for a_id in self.agents:
                    if self._is_invalid_agent(agent):
                        continue

                    agent_pos = self.agent_positions[a_id]
                    if (
                        abs(agent_pos[0] - crop_pos[0])
                        + abs(agent_pos[1] - crop_pos[1])
                        == 1
                    ):
                        adj_agent_levels.append(self.agent_levels[a_id])
                        adj_a_ids.append(a_id)
                level_sum = np.sum(adj_agent_levels)
                if level_sum >= crop_level >= 0:

                    def reward0() -> float:
                        return 6.0

                    def reward1() -> float:
                        return 2 * float(crop_level)

                    self.crop_removed[crop_idx] = True
                    total_crop_reward = reward0() if self.reward_idx == 0 else reward1()

                    if adj_a_ids:
                        for a_id in adj_a_ids:
                            local_reward = total_crop_reward * (
                                self.agent_levels[a_id] / level_sum
                            )
                            current_step_rewards[a_id] = (
                                current_step_rewards.get(a_id, 0.0) + local_reward
                            )

                    if self.reward_idx == 1:
                        for a_id in self.agents:
                            if not (self.terminations[a_id] or self.truncations[a_id]):
                                current_step_rewards[a_id] = (
                                    current_step_rewards.get(a_id, 0.0) + 0.5
                                )

            for a_id in self.agents:
                if not self._is_invalid_agent(agent):
                    self.rewards[a_id] = current_step_rewards.get(a_id, 0.0)

            self.current_step += 1
            all_crops_harvested = all(self.crop_removed)

            episode_is_over = False
            if all_crops_harvested:
                for a_id in self.agents:
                    self.terminations[a_id] = True
                    self.rewards[a_id] += 10.0
                episode_is_over = True

            if not episode_is_over and self.current_step >= self.max_cycles:
                for a_id in self.agents:
                    if not self.terminations[a_id]:
                        self.truncations[a_id] = True
                episode_is_over = True

            if episode_is_over:
                self.agents = []

            self._actions_this_turn = {}

        for a_id in self.possible_agents:
            self._cumulative_rewards[a_id] = self.rewards[a_id]
            self.infos[a_id] = {}

        self.agent_selection = self._agent_selector.next()
        self.render()

    def render(self) -> None:
        if self.render_mode is None:
            return

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
                obs_radius=self.obs_radius,
            )

        return

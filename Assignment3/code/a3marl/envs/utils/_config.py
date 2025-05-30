from dataclasses import dataclass

from pettingzoo import ParallelEnv


@dataclass
class EnvConfig:
    name_abbr: str
    env_creator: callable
    env_kwargs: dict[str, object]

    def get_env(self, **override_kwargs) -> ParallelEnv:
        if override_kwargs is not None:
            final_kwargs = self.env_kwargs.copy()
            final_kwargs.update(override_kwargs)
            return self.env_creator(**final_kwargs)
        else:
            return self.env_creator(**self.env_kwargs)

import pandas as pd

_FOLDER = "./data/temp"


def _save_init() -> None:
    import os

    if not os.path.exists(_FOLDER):
        os.makedirs(_FOLDER)
        print(f"Created directory: {_FOLDER}")


def save_episode_acc_to_csv(
    episode_means: list[float], file_name: str = "episode_means"
) -> None:
    path: str = f"{_FOLDER}/{file_name}_lastest.csv"
    df = pd.DataFrame(episode_means, columns=["Episode Mean"])

    df.to_csv(path, index=False)

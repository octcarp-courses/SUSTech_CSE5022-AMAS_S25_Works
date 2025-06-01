import pandas as pd

_FOLDER = "./data/temp"


def _save_init() -> None:
    import os

    if not os.path.exists(_FOLDER):
        os.makedirs(_FOLDER)
        print(f"Created directory: {_FOLDER}")


def save_episode_ret_to_csv(
    episode_means: list[float],
    file_name: str = "episode_means",
    file_folder: str = _FOLDER,
) -> None:
    path: str = f"{file_folder}/{file_name}_latest.csv"
    df = pd.DataFrame(
        {
            "Episode": range(len(episode_means)),
            "Mean Return": episode_means,
        }
    )

    df.to_csv(path, index=False)


def load_episode_ret_from_csv(
    file_path: str = "./data/temp/latest.csv",
) -> list[float]:
    df = pd.read_csv(file_path)
    episode_mean_list: list[float] = df["Mean Return"].astype(float).tolist()
    return episode_mean_list

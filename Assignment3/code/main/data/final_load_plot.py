import os
import pandas as pd
from a3marl.utils import plot_episodes

_CSV_FOLDER: str = "./final/"
_IMG_FOLDER: str = "./img_out/"


def load_data_and_plot(
    file_name: str = "episode_means",
    plot_title: str = "Episode Means",
) -> None:
    file_path = os.path.join(_CSV_FOLDER, f"{file_name}.csv")
    df = pd.read_csv(file_path)
    episode_mean_list: list[float] = df["Episode Mean"].astype(float).tolist()
    plot_episodes(
        episode_mean_list,
        title=plot_title,
        save_path=os.path.join(_IMG_FOLDER, f"{file_name}.png"),
        clear_after=False,
    )


def main() -> None:
    files_to_plot: dict[str, str] = {
        "pu_iql": "Pursuit IQL",
        "pu_cql": "Pursuit CQL",
        "fo_iql_r0": "Foraging IQL Reward 0",
        "fo_iql_r1": "Foraging IQL Reward 1",
        "latest": "Latest",
    }
    for file_name, plot_title in files_to_plot.items():
        load_data_and_plot(
            file_name=file_name,
            plot_title=plot_title,
        )


if __name__ == "__main__":
    main()

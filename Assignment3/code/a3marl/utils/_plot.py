import numpy as np
from matplotlib import pyplot as plt


def plot_episodes(
    episode_returns: list[float],
    title: str = "Training...",
    save_path: str = None,
    clear_after: bool = True,
) -> None:
    durations_np = np.array(episode_returns, dtype=np.float32)

    plt.clf()
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Return")
    plt.plot(durations_np)
    # take 10 episode averages and plot them too
    window_size: int = 10
    if len(durations_np) >= window_size:
        means_calculated = np.convolve(
            durations_np, np.ones(window_size) / window_size, mode="valid"
        )
        padding_value = durations_np[0]
        padding = np.full((window_size - 1,), padding_value, dtype=np.float32)
        means_to_plot = np.concatenate((padding, means_calculated))
        plt.plot(means_to_plot)
    plt.tight_layout()
    # plt.pause(0.001)

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved {title} to {save_path}")

    plt.show()
    if clear_after:
        plt.clf()

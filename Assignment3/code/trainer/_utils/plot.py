import torch
from matplotlib import pyplot as plt


def plot_episodes(durations: list[float], clear_after: bool = True) -> None:
    plt.figure(1)
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Avg. Return")
    plt.plot(durations_t.numpy())
    # take 10 episode averages and plot them too

    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        padding = torch.full((9,), durations_t[0], dtype=torch.float)
        means = torch.cat((padding, means))
        plt.plot(means.numpy())
    plt.show()
    # plt.pause(0.001)

    if clear_after:
        plt.clf()

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

TRACK_FILE: str = "track.csv"
GRAPH_FILE: str = "graph.csv"
GRAPH_THRESHOLD: float = 0.1

CSV_DIR: str = "./csv_wait/"
PNG_DIR: str = "./csv_img/"


def draw_track(filename: str) -> None:
    data = pd.read_csv(os.path.join(CSV_DIR, filename))
    ticks = data["tick"]
    track_counts = data["track_count"]

    plt.figure(figsize=(8, 6))
    plt.plot(ticks, track_counts)
    plt.title("Tick vs Track Count", fontsize=14)
    plt.xlabel("Tick", fontsize=12)
    plt.ylabel("Track Count", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PNG_DIR, "tick_count.png"))
    plt.close()


def draw_graph(filename: str) -> None:
    df = pd.read_csv(os.path.join(CSV_DIR, filename))
    df["value"] = df["value"].round(3)

    batch_groups = df.groupby("batch")

    def custom_layout(G: nx.Graph) -> dict:
        pos = {}
        for i, node in enumerate([1, 2, 3, 4]):
            pos[node] = (i, 0)
        for i, node in enumerate([5, 6, 7, 8]):
            pos[node] = (i, 1)
        return pos

    for batch, group in batch_groups:
        filtered_group = group[group["value"] > GRAPH_THRESHOLD]

        G = nx.Graph()

        for _, row in filtered_group.iterrows():
            G.add_edge(row["fromId"], row["toId"], weight=row["value"])

        plt.figure(figsize=(8, 6))
        pos = custom_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=500,
            node_color="lightblue",
            font_size=12,
            font_weight="bold",
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title(f"Graph for Batch {batch}")
        plt.tight_layout()
        plt.savefig(os.path.join(PNG_DIR, f"graph_batch_{batch}.png"))
        plt.close()


def main() -> None:
    print(f"Search {TRACK_FILE}, {GRAPH_FILE} from: {CSV_DIR}")

    for filename in os.listdir(CSV_DIR):
        if filename == TRACK_FILE:
            draw_track(filename)
            print(f"Track output image is saved in the {PNG_DIR}")
        elif filename == GRAPH_FILE:
            draw_graph(filename)
            print(f"Graph output image is saved in the {PNG_DIR}")


if __name__ == "__main__":
    main()

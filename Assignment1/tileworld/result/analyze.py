import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR : str = './csv_wait/'
PNG_DIR : str = './csv_img/'
PLOT_SCORE : bool = True
PLOT_ENERGY : bool = True

def plot_over_time(data: pd.DataFrame, value_column: str, ylabel: str, filename: str, png_dir: str) -> None:
    plt.figure(figsize=(10, 6))

    for robot_id in data['robot_id'].unique():
        robot_data = data[data['robot_id'] == robot_id]
        plt.plot(robot_data['tick'], robot_data[value_column], label=f'Robot {robot_id}')
    
    plt.title(f'Robot {ylabel} Over Time - {filename}')
    plt.xlabel('Tick')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    output_file = os.path.join(png_dir, f"{value_column}_{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {value_column} plot for {filename} as PNG.")

def plot_score_over_time(data: pd.DataFrame, filename: str, png_dir: str) -> None:
    plot_over_time(data, 'score', 'Score', filename, png_dir)

def plot_energy_level_over_time(data: pd.DataFrame, filename: str, png_dir: str) -> None:
    plot_over_time(data, 'energy', 'Energy Level', filename, png_dir)

def main() -> None:
    print(f"Loading CSV files from: {CSV_DIR}")

    for filename in os.listdir(CSV_DIR):
        if filename.endswith('.csv'):
            file_path = os.path.join(CSV_DIR, filename)
            data = pd.read_csv(file_path)

            if PLOT_SCORE:
                plot_score_over_time(data, filename, PNG_DIR)
            if PLOT_ENERGY:    
                plot_energy_level_over_time(data, filename, PNG_DIR)

    print(f"Output images are saved in the {PNG_DIR}")


if __name__ == "__main__":
    main()

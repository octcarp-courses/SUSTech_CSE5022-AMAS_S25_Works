import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR : str = './csv_wait/'
PNG_DIR : str = './csv_img/'
PLOT_SCORE : bool = False
PLOT_ENERGY : bool = True

def plot_score_over_time(data: pd.DataFrame, filename: str, png_dir: str) -> None:
    plt.figure(figsize=(10, 6))

    for robot_id in data['robot_id'].unique():
        robot_data = data[data['robot_id'] == robot_id]
        plt.plot(robot_data['tick'], robot_data['score'], label=f'Robot {robot_id}')
    
    plt.title(f'Robot Score Changes Over Time - {filename}')
    plt.xlabel('Tick')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    score_output_file = os.path.join(png_dir, f"score_{os.path.splitext(filename)[0]}.png")
    plt.savefig(score_output_file)
    plt.close()
    print(f"Saved score plot for {filename} as PNG.")

def plot_energy_level_over_time(data: pd.DataFrame, filename: str, png_dir: str) -> None:
    plt.figure(figsize=(10, 6))

    for robot_id in data['robot_id'].unique():
        robot_data = data[data['robot_id'] == robot_id]
        plt.plot(robot_data['tick'], robot_data['energy'], label=f'Robot {robot_id}')
    
    plt.title(f'Robot Energy Level Over Time - {filename}')
    plt.xlabel('Tick')
    plt.ylabel('Energy Level')
    plt.legend()
    plt.grid(True)

    energy_output_file = os.path.join(png_dir, f"energy_{os.path.splitext(filename)[0]}.png")
    plt.savefig(energy_output_file)
    plt.close()
    print(f"Saved energy level plot for {filename} as PNG.")

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

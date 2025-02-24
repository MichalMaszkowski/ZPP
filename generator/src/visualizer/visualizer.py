import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from tqdm import tqdm
import os


def visualize_simulation(simulation: pd.DataFrame, number_of_frames: int = 258):
    """
    Visualizes the simulation of nuclei movement and ERK values over time.

        Args:
        - simulation (pd.DataFrame): The DataFrame containing the simulation data.
        - number_of_frames (int): The number of frames to simulate.

        Returns:
        - None
    """
    marker = 'ERKKTR_ratio'

    min_value = np.log(simulation[marker].min())
    max_value = np.log(simulation[marker].max())

    output_dir = "../../data/temp_frames"
    os.makedirs(output_dir, exist_ok=True)
    frames = []

    colors = ['darkblue', 'blue', 'turquoise', 'yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('DarkBlueToYellow', colors)

    for t in tqdm(range(1, number_of_frames + 1), desc='Creating frames'):
        current_frame = simulation[simulation['Image_Metadata_T'] == t]
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            current_frame['objNuclei_Location_Center_X'],
            current_frame['objNuclei_Location_Center_Y'],
            s=16,
            c=np.log(current_frame[marker]),
            cmap=custom_cmap,
            vmin=min_value,
            vmax=max_value
        )

        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.xlabel('Center X')
        plt.ylabel('Center Y')
        plt.title(f'Simulating Movement Time: {t}')
        plt.grid(False)

        cbar = plt.colorbar(sc, label=f'Intensity ({marker})')

        filename = f"../../data/temp_frames/frame_{t:03d}.png"
        plt.savefig(filename, transparent=False)
        frames.append(filename)
        plt.close()

    with imageio.get_writer("../../data/simulation.gif", mode='I', duration=1) as writer:
        for frame in frames:
            image = imageio.imread(frame)[..., :3]
            writer.append_data(image)

    for frame in frames:
        os.remove(frame)

    os.rmdir(output_dir)






from typing import Tuple
import numpy as np
import torch
import os

import src.utils.utils as utils
import src.visualizer.visualizer as visualizer
import src.transformations.transformations as transformations

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


@torch.no_grad()
def load_experiment_data_to_tensor(experiments: Tuple[int] = (1, 2, 3, 4, 5, 6),
                                   maintain_experiment_visualization: bool = False) -> torch.Tensor:
    df = utils.unpack_and_read('../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz')
    if not os.path.exists("../../data/experiments"):
        os.makedirs("../../data/experiments")

    df['ERKKTR_ratio'] = np.clip(df['ERKKTR_ratio'], 0.4, 2.7)
    df = df[df['Exp_ID'].isin(experiments)]
    videos_count = len(df[['Exp_ID', 'Image_Metadata_Site']].drop_duplicates())
    experiments_tensor = torch.zeros(videos_count, 258, 3, 256, 256, device=DEVICE, dtype=torch.float16)

    i = 0
    for experiment in experiments:
        df_experiment = df[df['Exp_ID'] == experiment]
        fields_of_view = np.sort(df_experiment['Image_Metadata_Site'].unique())

        for field_of_view in fields_of_view:
            df_fov = df_experiment[df_experiment['Image_Metadata_Site'] == field_of_view]
            frames_count = df_fov['Image_Metadata_T'].max() + 1
            visualizer.visualize_simulation(df_fov, number_of_frames=frames_count,
                                            path=f"../../data/experiments/experiment_{experiment}_fov_{field_of_view}.gif")

            fov_tensor = ((transformations.transform_gif_to_tensor(
                        f"../../data/experiments/experiment_{experiment}_fov_{field_of_view}.gif"))
                        .squeeze(0))

            if fov_tensor.shape[0] < 258:
                padding = torch.zeros(258 - fov_tensor.shape[0], 3, 256, 256, device=DEVICE)
                fov_tensor = torch.cat((fov_tensor, padding), dim=0)

            experiments_tensor[i] = fov_tensor

            if not maintain_experiment_visualization:
                os.remove(f"../../data/experiments/experiment_{experiment}_fov_{field_of_view}.gif")
            i += 1

    if not maintain_experiment_visualization:
        os.rmdir("../../data/experiments")

    torch.save(experiments_tensor, "../../data/experiments_tensor_exp_1.pt")

    return experiments_tensor


# Example usage of the function:
# Here I create a tensor containing the data of experiment 1.
if __name__ == "__main__":
    load_experiment_data_to_tensor((2,))


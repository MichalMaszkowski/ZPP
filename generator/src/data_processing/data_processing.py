from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
                                   maintain_experiment_visualization: bool = False):
    """
    Transforms tabular data to images and load them to a tensor of shape (B, S, C, H, W)
    Tensor is not normalized and saved as float16 for memory efficiency
    Clipping 'ERKKTR_ratio' to [0.4, 2.7] to avoid outliers
    Saves each field of view as a separate tensor file

    Args:
    - experiments Tuple[int]: Experiments to include in tensor. Default (1, 2, 3, 4, 5, 6) - All experiments
    - maintain_experiment_visualization (bool): If True, keeps the visualizations of the experiments. Default False
    """
    df = utils.unpack_and_read('../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz')
    if not os.path.exists("../../data/experiments"):
        os.makedirs("../../data/experiments")

    df['ERKKTR_ratio'] = np.clip(df['ERKKTR_ratio'], 0.4, 2.7)
    df = df[df['Exp_ID'].isin(experiments)]

    for experiment in experiments:
        df_experiment = df[df['Exp_ID'] == experiment]
        fields_of_view = np.sort(df_experiment['Image_Metadata_Site'].unique())
        experiments_tensor = torch.zeros(1, 258, 3, 256, 256, device=DEVICE, dtype=torch.float16)

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

            experiments_tensor[0] = fov_tensor

            if not maintain_experiment_visualization:
                os.remove(f"../../data/experiments/experiment_{experiment}_fov_{field_of_view}.gif")

            torch.save(experiments_tensor,
                       f"../../data/tensors_to_load/experiments_tensor_exp_{experiment}_fov_{field_of_view}.pt")

    if not maintain_experiment_visualization:
        os.rmdir("../../data/experiments")


class TensorDataset(Dataset):
    def __init__(self, data_folder: str = "../../data/tensors_to_load/",
                 load_to_ram: bool = False):
        """
        Args:
            data_folder (str): Path to the folder containing tensor files with multiple batches.
            load_to_ram (bool): If True, loads all tensors into RAM. Otherwise, loads lazily from disk.
        """
        self.data_folder = data_folder
        self.file_names = sorted(os.listdir(data_folder))
        self.file_names = [file for file in self.file_names if 'experiments_tensor' in file]
        self.load_to_ram = load_to_ram
        self.data_len = len(self.file_names)

        if self.load_to_ram:
            self.data = []
            for f_name in self.file_names:
                file_path = os.path.join(data_folder, f_name)
                batches = torch.load(file_path)
                self.data.extend(batches)

            self.data = torch.stack(self.data)
        else:
            self.data = self.file_names

    def __len__(self):
        """
        Calculate the total number of batches in all files.
        """
        return self.data_len

    def __getitem__(self, idx):
        """
        Get a batch by index.
        """
        if self.load_to_ram:
            return self.data[idx][:-1], self.data[idx][1:]
        else:
            file_idx = self.file_names[idx]
            file_path = os.path.join(self.data_folder, file_idx)
            batches = torch.load(file_path)
            item = batches[0]

            return item[:-1], item[1:]


# Example usage of the function:
# Here I create a tensor containing the data from experiment 1.
if __name__ == "__main__":
    # load_experiment_data_to_tensor((1,))

    #my_tensor = torch.load("../../data/tensors_to_load/experiments_tensor_exp_1_fov_1.pt")
    dataset = TensorDataset(load_to_ram=True)
    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

    for source, target in loader:
        print(source.shape, target.shape)



import torch
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from tqdm import tqdm
from typing import Tuple, List
from enum import Enum
import random
import os

import src.utils.utils as utils
import src.visualizer.visualizer as visualizer

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class ExtractionMethod(Enum):
    RANDOM = "random"
    GRID = "grid"

    def __str__(self):
        return self.value

class Rotation(Enum):
    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270

    def __str__(self):
        return f"{self.value}°"

class Generator:            
    def __init__(self, df_first_frame: pd.DataFrame, number_of_frames: int = 259):
        """
        Initializes the Generator object with the first frame of data and the number of frames to simulate.
        :param df_first_frame:
        :param number_of_frames:
        """
        self.df_first_frame = df_first_frame
        self.number_of_frames = number_of_frames

    def generate_next_move(self, current_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates the next movement of nuclei in the frame by adding random noise to their X and Y coordinates.

        Args:
        - current_frame (pd.DataFrame): The DataFrame representing the current frame of data, which includes the positions.

        Returns:
        - pd.DataFrame: The updated DataFrame with modified X and Y coordinates for the nuclei.
        """
        current_frame['objNuclei_Location_Center_X'] += np.random.normal(0, 0.43, size=current_frame[
            'objNuclei_Location_Center_X'].shape)
        current_frame['objNuclei_Location_Center_Y'] += np.random.normal(0, 0.43, size=current_frame[
            'objNuclei_Location_Center_Y'].shape)
        return current_frame

    def generate_next_ERK(self, points: pd.DataFrame, adjacency_matrix: torch.tensor, T: int) -> pd.DataFrame:
        """
        Simulates the next ERK values for the given nuclei based on adjacency and previous ERK values.

        Args:
        - points (pd.DataFrame): The DataFrame containing the current positions and ERK values.
        - adjacency_matrix (torch.tensor): A tensor representing the adjacency matrix, indicating which nuclei are neighbors.
        - T (int): The current time point/frame number.

        Returns:
        - pd.DataFrame: The updated DataFrame with new ERK values.
        """
        points = points.copy()
        points = points.sort_values(by=['track_id'])
        points_ERK = torch.tensor(points['ERKKTR_ratio'].values, device=DEVICE, dtype=torch.float32)

        mean_before = torch.mean(points_ERK)
        std_before = torch.std(points_ERK)

        max_neighbor = torch.max(adjacency_matrix * points_ERK, dim=1).values

        mask = max_neighbor < 1.2
        new_ERK = torch.where(mask, points_ERK, 0.005 * max_neighbor + 0.995 * points_ERK)

        mean_after = torch.mean(new_ERK)
        std_after = torch.std(new_ERK)

        noise_mean = mean_before - mean_after
        noise_std = torch.sqrt(abs(std_after ** 2 - std_before ** 2))

        new_ERK = torch.clamp(
            new_ERK + torch.normal(mean=float(noise_mean), std=float(noise_std), size=new_ERK.shape, device=DEVICE),
            min=0.4, max=2.7
        )

        sampled_values_df = pd.DataFrame({
            'track_id': points['track_id'].values,
            'ERKKTR_ratio': new_ERK.cpu().numpy()
        })
        points_filtered = points.drop(columns=['ERKKTR_ratio'], errors='ignore')
        sampled_values_df = sampled_values_df.merge(points_filtered, on='track_id')

        sampled_values_df['Image_Metadata_T'] = T
        sampled_values_df['track_id'] = sampled_values_df['track_id'].astype(int)
        sampled_values_df = sampled_values_df[['track_id', 'objNuclei_Location_Center_X',
                                               'objNuclei_Location_Center_Y', 'ERKKTR_ratio',
                                               'Image_Metadata_T']]
        return sampled_values_df

    def calculate_neighbors(self, points: pd.DataFrame) -> torch.Tensor:
        """
        Calculates the adjacency matrix based on the spatial relationships between points (nuclei).

        Uses the Voronoi tessellation algorithm to determine the neighboring nuclei.

        Args:
        - points (pd.DataFrame): A DataFrame with position information for each track.

        Returns:
        - torch.Tensor: The adjacency matrix indicating the neighbors of each nucleus.
        """
        points = points.values
        vor = Voronoi(points[:, :3])

        unique_track_ids, inverse_indices = np.unique(points[:, 0], return_inverse=True)
        num_tracks = len(unique_track_ids)

        ridge_points = vor.ridge_points.flatten()
        ridge_neighbors = inverse_indices[ridge_points].reshape(-1, 2)

        adjacency_matrix = np.zeros((num_tracks, num_tracks), dtype=np.uint8)
        adjacency_matrix[ridge_neighbors[:, 0], ridge_neighbors[:, 1]] = 1
        adjacency_matrix[ridge_neighbors[:, 1], ridge_neighbors[:, 0]] = 1

        return torch.tensor(adjacency_matrix, device=DEVICE)

    def generate_video(self, number_of_frames = None):
        """
            Generates a simulated video of tracked nuclei over multiple frames, updating their positions and ERK values.

            Args:
            - df_first_frame (pd.DataFrame): The initial frame with position and ERK data.
            - number_of_frames (int): The total number of frames to simulate. Defaults to 259.

            Returns:
            - pd.DataFrame: The DataFrame containing the complete video simulation data for all frames.
            """
        if number_of_frames is None:
            number_of_frames = self.number_of_frames
        result_data_frame = self.df_first_frame.copy()
        current_frame = self.df_first_frame.copy()
        for T in tqdm(range(2, number_of_frames + 1), desc='Generating video'):
            adjacency_matrix = self.calculate_neighbors(current_frame)
            next_frame = self.generate_next_ERK(current_frame, adjacency_matrix, T)
            next_frame = self.generate_next_move(next_frame)
            result_data_frame = pd.concat([result_data_frame, next_frame])
            current_frame = next_frame

        result_data_frame = result_data_frame.reset_index(drop=True)

        return result_data_frame

    def get_augmented_data(
        self, 
        first_frame: pd.DataFrame, 
        subwindow_size: Tuple[int, int] = (200, 200),  
        rotations: Rotation = Rotation.DEG_0,          # <-- Poprawione
        extraction_method: Tuple[ExtractionMethod, int] = (ExtractionMethod.RANDOM, 3),  
        time_interval: int = None, 
        folder_path: str = '../../augmented_data/', 
        filename: str = 'augmented_data_'
    ):
        """
        
        """

        def get_origins():
            
            sub_height, sub_width = subwindow_size
            method, num_origins = extraction_method
            origins = []
            start_x, start_y = 0.0, 0.0

            assert start_x + sub_width < first_frame_width

            if method == ExtractionMethod.RANDOM:
                
                for _ in range(num_origins):
                    x = random.uniform(0.0, first_frame_width - sub_width)
                    y = random.uniform(0.0, first_frame_height - sub_height)
                    origins.append((x, y))


            if method == ExtractionMethod.GRID:

                for i in range(num_origins):

                    assert y + sub_height < first_frame_height, "Too many frames in this grid"
                    origins.append((x, y))
                    x += sub_width
                    if x > first_frame_width:
                        x = 0.0
                        y += sub_height
                    
            return origins

        def get_first_frames_from_origins(origins: List[Tuple[float, float]]) -> List[pd.DataFrame]:

            first_frames = []

            for origin in origins:
                x_min, y_min = origin
                x_max, y_max = x_min + sub_width, y_min + sub_height

                filtered_frame = first_frame[(first_frame['objNuclei_Location_Center_X'].between(x_min, x_max)) & (first_frame['objNuclei_Location_Center_Y'].between(y_min, y_max))]
                first_frames.append(filtered_frame)

            return first_frames

        def rotate_points_inplace(frames, angle_deg=rotations.value, x_col='objNuclei_Location_Center_X', y_col='objNuclei_Location_Center_Y'):

            for df in frames:
                angle_rad = np.radians(angle_deg)
                
                new_X = df[x_col] * np.cos(angle_rad) - df[y_col] * np.sin(angle_rad)
                new_Y = df[x_col] * np.sin(angle_rad) + df[y_col] * np.cos(angle_rad)
                
                df.loc[:, x_col] = new_X
                df.loc[:, y_col] = new_Y

        sub_width, sub_height = subwindow_size
        margin = 5.0
        min_X, max_X = first_frame['objNuclei_Location_Center_X'].min() - margin, first_frame['objNuclei_Location_Center_X'].max() + margin
        min_Y, max_Y = first_frame['objNuclei_Location_Center_Y'].min() - margin, first_frame['objNuclei_Location_Center_Y'].max() + margin

        first_frame_height = (max_Y - min_Y).round()
        first_frame_width = (max_X - min_X).round()

        assert sub_height <= first_frame_height and sub_width <= first_frame_width, (
            f"Subwindow should fit in the first frame. "
            f"First frame height: {first_frame_height}, First frame width: {first_frame_width}, "
            f"Subwindow height: {sub_height}, Subwindow width: {sub_width}"
        )

        origins = get_origins()
        first_frames = get_first_frames_from_origins(origins)
        rotate_points_inplace(first_frames)

        
        augumented_series = []
        for i, df in enumerate(first_frames):
            augumented_df = self.generate_video(time_interval)
            curr_filename = filename + str(i) + '.csv'
            full_path = os.path.join(folder_path, curr_filename)

            os.makedirs(folder_path, exist_ok=True)
            augumented_df.to_csv(full_path, index=False)
        




if __name__ == "__main__":
    df = utils.unpack_and_read('../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz')
    df_first_frame = df[(df['Image_Metadata_Site'] == 1) & (df['Exp_ID'] == 1) & (df['Image_Metadata_T'] == 1)][
        ['track_id', 'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 'ERKKTR_ratio', 'Image_Metadata_T']]
    generator = Generator(df_first_frame=df_first_frame)
    # video_data = generator.generate_video()
    generator.get_augmented_data(df_first_frame, time_interval = 3)
    # visualizer.visualize_simulation(video_data)

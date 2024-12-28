import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, List, Optional
from sklearn.neighbors import KernelDensity
from src.extractor.extractor import *
from src.visualizer.visualizer import *
from typing import Union


class Generator:
    def __init__(self) -> None:
        """Initializes the point generator."""
        pass

    def deploy_points(
        self, 
        height: int = 1024, 
        width: int = 1024, 
        points_num: int = 1128, 
        min_dist: int = 23
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Generates a list of points randomly distributed on a board of dimensions (width x height),
        with a minimum distance constraint between points.

        Args:
            height (int): The height of the board (default is 1024).
            width (int): The width of the board (default is 1024).
            points_num (int): The number of points to generate (default is 1128).
            min_dist (int): The minimum distance between points (default is 23).

        Returns:
            Optional[List[Tuple[int, int]]]: A list of points (x, y) or None in case of invalid parameters.
        """
        if height <= 0 or width <= 0 or points_num <= 0:
            raise ValueError("Board dimensions and the number of points must be positive.")

        points = []

        while len(points) < points_num:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # Check if the point is sufficiently far from all existing points
            if all(math.sqrt((x - px)**2 + (y - py)**2) >= min_dist for px, py in points):
                points.append((x, y))
        
        return points
    
    def update_points_positions(
        self,
        points: List[Tuple[int, int]], 
        max_shake: float = 0.2, 
        threshold: int = 23  # Minimum distance between points
    ) -> List[Tuple[int, int]]:
        """
        Shakes the points slightly by a random amount in the range of [-max_shake, max_shake],
        ensuring that the new positions maintain the minimum distance between them.

        Args:
            points (List[Tuple[int, int]]): Current positions of points.
            max_shake (int): Maximum displacement distance for shaking (default is 0.2).
            threshold (int): Minimum distance that must be maintained between points (default is 23).

        Returns:
            List[Tuple[int, int]]: New positions of points after shaking.
        """
        new_points = []

        for (px, py) in points:
            while True:
                # Slightly shake the point by adding small random values to x and y coordinates
                shake_x = random.uniform(-max_shake, max_shake)
                shake_y = random.uniform(-max_shake, max_shake)

                # New candidate position after shaking
                new_x = px + shake_x
                new_y = py + shake_y

                # Check if the new position maintains the minimum distance from all other points
                valid_position = True
                for (other_x, other_y) in new_points:
                    # Calculate Euclidean distance between the new position and the other points
                    distance = math.sqrt((new_x - other_x) ** 2 + (new_y - other_y) ** 2)
                    if distance < threshold:
                        valid_position = False
                        break  # If the new point is too close, break and retry

                # If the new position is valid, add it to the list and break the loop
                if valid_position:
                    new_points.append((new_x, new_y))
                    break

        return new_points

    def generate_video(self, 
        df_first_frame: pd.DataFrame,
        kde_model: object,
        number_of_frames:int = 10
        ) -> pd.DataFrame:
        """
        Generates a sequence of frames based on the initial frame and a probabilistic model.

        Parameters:
        -----------
        df_first_frame : pd.DataFrame
            Initial frame of the video, representing the starting state.
        number_of_frames : int
            Number of frames to generate in the sequence.
        kde_model : object
            A probabilistic model (e.g., KDE) used to generate the distribution for the next frames.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing all generated frames in sequence, indexed consecutively.
        """
        result_data_frame = df_first_frame.copy()
        current_frame = df_first_frame.copy()

        for T in range(2, number_of_frames + 1):
            next_frame = self.generate_next_ERK(current_frame, kde_model, T)
            result_data_frame = pd.concat([result_data_frame, next_frame])
            current_frame = next_frame

        result_data_frame = result_data_frame.reset_index(drop=True)
        return result_data_frame


    def generate_next_ERK(self, points: pd.DataFrame, 
    kde_model: object,
    frame_number:int=1
    ) -> pd.DataFrame:
        '''
        Generates the next frame of data by estimating the 'ERKKTR_ratio' for each point, 
        considering its neighbors and using a probabilistic model (e.g., KDE).

        Parameters:
        -----------
        points : pd.DataFrame or ndarray
            DataFrame or numpy array containing points with columns ['track_id', 
            'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 'ERKKTR_ratio'].
        T : int
            The current time step or frame number for the generated data.
        kde_model : object
            A probabilistic model (e.g., Kernel Density Estimation) used to sample the 'ERKKTR_ratio'
            based on the current value and its neighbors' values.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the next set of points with updated 'ERKKTR_ratio' values 
            and additional metadata ('track_id', 'objNuclei_Location_Center_X', 
            'objNuclei_Location_Center_Y', 'Image_Metadata_T').

        Notes:
        ------
        - The function estimates the 'ERKKTR_ratio' for each point by considering its neighbors 
          using Voronoi tessellation and sampling from the KDE model.
        - It updates the `ERKKTR_ratio` value based on the current point's value and the mean value 
          of its neighbors.
        '''
        points_data_frame = points.copy()
        points = points.values
        # print("points.head()")
        # print(points)
        vor = Voronoi(points[:, :3])
        neighbors = {track_id: set() for track_id in points[:, 0]}
        for p1, p2 in vor.ridge_points:
            neighbors[points[p1][0]].add(points[p2][0])
            neighbors[points[p2][0]].add(points[p1][0])

        sampled_values = []
        for cell in points[:, 0]:
            my_erk = points_data_frame[(points_data_frame['track_id'] == cell)]['ERKKTR_ratio'].values
            if len(my_erk) == 0:
                sampled_values.append(None)
                continue
            else:
                my_erk = my_erk[0]

            neighbor_values = []
            for neighbor in neighbors[cell]:
                array_values = (points_data_frame[(points_data_frame['track_id'] == neighbor)]['ERKKTR_ratio'].values)
                if len(array_values) > 0:
                    neighbor_values.append(array_values[0])

            neighbor_erk = my_erk + np.mean(neighbor_values)
            sampled_value = self.sample_erk_given_neighbor([neighbor_erk], kde_model=kde_model)
            sampled_values.append((cell, sampled_value[0]))

        df_sampled_values = pd.DataFrame(sampled_values, columns=['track_id', 'ERKKTR_ratio'])
        points_data_frame_filtered = points_data_frame.drop(columns=['ERKKTR_ratio'], errors='ignore')
        df_sampled_values = df_sampled_values.merge(points_data_frame_filtered, on='track_id')

        df_sampled_values['Image_Metadata_T'] = frame_number
        df_sampled_values['track_id'] = df_sampled_values['track_id'].astype(int)
        df_sampled_values = df_sampled_values[['track_id', 'objNuclei_Location_Center_X', 
                                                'objNuclei_Location_Center_Y', 'ERKKTR_ratio', 
                                                'Image_Metadata_T']]
        
        print("df_sampled_values")
        print(f"{df_sampled_values.columns}")
        return df_sampled_values


    def sample_erk_given_neighbor(self,
    neighbor_values: Union[list, np.ndarray],
    kde_model: object,
    n_samples:int = 1,
    my_erk_range:tuple=(0.1, 3),
    resolution:int=200):
        """
        Sample 'my_erk' values conditionally given 'neighbor_erk' values.
        
        Parameters:
        -----------
        neighbor_values : list or array
            A list or array of 'neighbor_erk' values from neighboring points.
        kde_model : object
            A fitted Kernel Density Estimation (KDE) model used to evaluate probabilities.
        n_samples : int, optional (default=1)
            The number of 'my_erk' samples to generate for each neighbor value.
        my_erk_range : tuple, optional (default=(0.1, 3))
            The range of values for 'my_erk' to consider during sampling.
        resolution : int, optional (default=200)
            The number of grid points used for evaluating the KDE model.

        Returns:
        --------
        np.ndarray
            An array of sampled 'my_erk' values for each neighbor value.

        """
        # Lista, która będzie przechowywać próbkowane wartości 'my_erk'
        sampled_values = []

        # Tworzenie siatki wartości 'my_erk' w zadanym zakresie
        my_erk_grid = np.linspace(my_erk_range[0], my_erk_range[1], resolution)

        # Iteracja po każdej wartości 'neighbor_erk' z sąsiadów
        for neighbor_value in neighbor_values:
            # Tworzenie siatki par [my_erk, neighbor_value]
            grid = np.column_stack((my_erk_grid, np.full(resolution, neighbor_value)))

            # Ocena prawdopodobieństw logarytmicznych za pomocą modelu KDE na siatce
            log_probs = kde_model.score_samples(grid)

            # Zamiana log-prawdopodobieństw na prawdopodobieństwa
            probs = np.exp(log_probs)

            # Normalizacja prawdopodobieństw, aby ich suma wynosiła 1
            probs /= probs.sum()

            # Próba wartości 'my_erk' z siatki na podstawie obliczonych prawdopodobieństw
            sampled_value = np.random.choice(my_erk_grid, size=n_samples, p=probs)
            
            # Dodanie pierwszej próbki do listy próbek
            sampled_values.append(sampled_value[0])

        # Zwrócenie próbkowanych wartości jako numpy array
        return np.array(sampled_values)


# Example usage
if __name__ == "__main__":
    generator = Generator()

    data = {
    'my_erk': np.random.exponential(scale=1.0, size=900),
    'neighbor_erk': np.random.exponential(scale=1.0, size=900)
    }
    df_ERK = pd.DataFrame(data)

    # Prepare data for KDE
    X = df_ERK[['my_erk', 'neighbor_erk']].values

    # Fit a 2D kernel density model
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)  # Adjust bandwidth as needed
    kde.fit(X)

    try:
        # Generate points
        points = generator.deploy_points(height=1024, width=1024, points_num=1128, min_dist=23)

        # Visualize the points
        visualize_points(points)
        visualize_points(generator.update_points_positions(points))
        visualize_voronoi_field(points, 1024, 23)

        print("czytam dane")
        data_df = pd.read_csv("/home/ajask/Desktop/zpp/ZPP/single-cell-tracks_exp1-6_noErbB2.csv")
        extractor = Extractor(data_df)
        print("przygotowuje pierwszą klatkę")
        first_frame = extractor.prepare_first_frame_for_simulation(columns = ['track_id', 'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 'ERKKTR_ratio', 'Image_Metadata_T'])
        print("przygotowuje video")
        video = generator.generate_video(first_frame, kde, 4)
        print("przygotowuje gif")
        save_simulation_to_gif(video, 'ERKKTR_ratio', 40, 'simulation.gif', 1)
    except ValueError as e:
        print(f"Error: {e}")

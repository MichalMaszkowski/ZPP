import numpy as np
import math
import pandas as pd
from typing import Tuple, List

class Extractor:
    def __init__(self,  data: pd.DataFrame) -> None:
        """
        Initializes the feature extractor.
        Args:
            data: data frame containing the data from experiments
        """
        self.data = data

    def extract_positions(
        self,
        exp_id : int,
        frame : int
    ) -> List[Tuple[int, int]]:
        """
        Extracts cells positions from experiments data.
        Args:
            exp_id : id of the experiment
            frame : number of the frame in the given experiment
        Returns:
            List[Tuple[int, int]]: a list of all the cells postions in the given experiment and frame
        """
        # Filter the data to get only rows corresponding to the specified experiment ID and frame
        filtered_data = self.data[(self.data['Exp_ID'] == exp_id) & (self.data['Image_Metadata_T'] == frame)]

        # Extract the cells positions into a list
        positions = list(
            zip(filtered_data['objNuclei_Location_Center_X'], filtered_data['objNuclei_Location_Center_Y']))

        return positions


    def extract_distances(
        self,
        exp_id: int,
        frame: int
    ) -> List[float]:
        """
        Extracts the distribution of distances between cells nuclei.
        Args:
            exp_id : id of the experiment
            frame : number of the frame in the given experiment
        Returns:
            List[float]: a list of all the distances
        """
        positions = self.extract_positions(exp_id, frame)
        distances = []

        # Calculate the distance between all pairs of positions
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]

                # Euclidean distance
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(distance)

        # If there are no distances (empty list or only one position), return an empty list
        if not distances:
            return []

        return distances

    def get_distances_distribution(
        self,
        exp_id: int,
        frame: int
    ) -> Tuple[float, float]:
        """
        Extracts the distribution of distances between cells nuclei.
        Args:
            exp_id : id of the experiment
            frame : number of the frame in the given experiment
        Returns:
            Tuple[float, float]: mean and variance of the distances.
        """
        distances = np.array(self.extract_distances(exp_id, frame))
        # If there are no distances (empty list or only one position), return 0, 0
        if not distances:
            return 0, 0
        return np.mean(distances), np.var(distances)

    def extract_cells_count(
        self,
        exp_id: int,
        frame: int
    ) -> int:
        """
            Extracts the count of cells in a frame.
            Args:
                exp_id : id of the experiment
                frame : number of the frame in the given experiment
            Returns:
                int: number of cells.
        """
        filtered_data = self.data[(self.data['Exp_ID'] == exp_id) & (self.data['Image_Metadata_T'] == frame)]
        return len(filtered_data)
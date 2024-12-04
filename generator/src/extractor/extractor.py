import numpy as np
import pandas as pd
import json
from typing import Tuple, List
from scipy.spatial import Voronoi
from scipy.spatial import distance_matrix
import networkx as nx

class Extractor:
    def __init__(self,  data: pd.DataFrame) -> None:
        """
        Initializes the feature extractor.
        Args:
            data: data frame containing the data from experiments
        """
        self.data = data

    def build_graph_voronoi(
        self,
        positions: List[Tuple[float, float]]
    ) -> nx.Graph:
        """
        Builds a graph based on the Voronoi diagram of a list of cells positions.

          Parameters:
              positions (List[Tuple[float, float]]): A list of 2D points (x, y) representing cell positions.
          Returns:
              nx.Graph: A NetworkX graph where nodes correspond to points and edges represent Voronoi neighbors.
          """
        # Convert the list of points to a NumPy array
        np_positions = np.array(positions)

        # Generate the Voronoi diagram
        vor = Voronoi(np_positions)
        # Create the graph
        graph = nx.Graph()

        # Add edges based on Voronoi neighbors
        for neighbors in vor.ridge_points:
            graph.add_edge(*neighbors)

        return graph

    def extract_positions(
        self,
        exp_id: int,
        frame: int
    ) -> List[Tuple[float, float]]:
        """
        Extracts cells positions from experiments data.
        Args:
            exp_id : id of the experiment
            frame : number of the frame in the given experiment
        Returns:
            List[Tuple[float, float]]: a list of all the cells positions in the given experiment and frame
        """
        # Filter the data to get only rows corresponding to the specified experiment ID and frame
        filtered_data = self.data[(self.data['Exp_ID'] == exp_id) & (self.data['Image_Metadata_T'] == frame)]

        # Extract the cells positions into a list
        positions = list(
            zip(filtered_data['objNuclei_Location_Center_X'], filtered_data['objNuclei_Location_Center_Y']))

        return positions


    def nearest_neighbours(
        self,
        positions: List[Tuple[float, float]]
    ) -> dict:
        """
        Extracts simple statistics of the list of distances to the cells nearest neighbours.
        Args:
            positions: a list of all the cells positions in one frame
        Returns:
            dict: A dictionary containing the following metrics:
            - 'min_distances': The minimum distances to the nearest neighbor for each point
            (the order of the min_distances array corresponds to the order of the points in the input positions array).
            - 'mean_min_distance': The mean of the minimum distances.
            - 'min_distance_overall': The smallest minimum distance across all points.
            - 'max_distance_overall': The largest minimum distance across all points.
        """
        distances = distance_matrix(positions, positions)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        min_distances = distances.min(axis=1)  # Distance to nearest neighbor

        return {
            'min_distances': min_distances,
            'mean_min_distance':  np.mean(min_distances),
            'min_distance_overall': np.min(min_distances),
            'max_distance_overall': np.max(min_distances)
        }

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

    def extract_ERKKTR_stats(
        self,
        exp_id: int,
        frame: int
    ) -> dict:
        """
        Extracts average and standard deviation of ratios of ERK and KTR markers in a given frame.
        
        Args:
            exp_id: ID of the experiment.
            frame: The frame number in the given experiment.
            
        Returns:
            dict: A dictionary with the following keys:
                - 'average_ERKKTR_ratio': The average of ERK/KTR ratios in the frame.
                - 'std_ERKKTR_ratio': The standard deviation of the ERK/KTR ratios in the frame.
        """
        # Filter the data to get only the rows corresponding to the specified experiment and frame
        filtered_data = self.data[(self.data['Exp_ID'] == exp_id) & (self.data['Image_Metadata_T'] == frame)]

        # Extract the ERKKTR_ratio values as a list
        erkktr_ratios = list(filtered_data['ERKKTR_ratio'])
        
        # Return the results as a dictionary
        return {
            'average_ERKKTR_ratio': np.mean(erkktr_ratios),
            'std_ERKKTR_ratio': np.std(erkktr_ratios)
        }
    

    def experiment_to_json(self, exp_id: int, filename: str) -> dict:
        """
        Extracts various data and saves it to a JSON file.
        
        Args:
            exp_id (int): ID of the experiment.
            filename (str): Path of the JSON file to save the data.
        
        Returns:
            dict: Dictionary containing the extracted data;
        """
        # Initialize a dictionary to store all extracted data
        data_dict = {}

        # Get the sorted unique frame numbers from the "Image_Metadata_T" column
        frame_numbers = sorted(self.data['Image_Metadata_T'].unique())
        # Check if the frame numbers are consecutive, starting from 0
        if frame_numbers != list(range(min(frame_numbers), max(frame_numbers) + 1)):
            raise ValueError(f"Gaps detected in the frame numbers. Expected a continuous sequence from 0 to {max(frame_numbers)}.")
        
        cell_counts = []
        erk_ratio_avgs = []
        erk_ratio_stds = []
        
        # Iterate over each frame number and count cells
        for frame in frame_numbers:
            cell_counts.append(self.extract_cells_count(exp_id, frame))
            stats = self.extract_ERKKTR_stats(exp_id, frame)
            erk_ratio_avgs.append(stats["average_ERKKTR_ratio"])
            erk_ratio_stds.append(stats["std_ERKKTR_ratio"])
        
        data_dict["cell_count"] = {
            "avg": np.mean(cell_counts),  # Average cell count
            "std": np.std(cell_counts)   # Standard deviation of cell count
        }
        data_dict["ERKKTR_ratio"] = {
            # Averages stats
            "avg": np.mean(erk_ratio_avgs), # Average ratio over all frames
            "std_of_averages": np.std(erk_ratio_avgs), # Standard deviation of averages
            # Standard deviation stats
            "avg_std": np.mean(erk_ratio_stds) # Average standard deviation 
        }

        # Save the data to a JSON file
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=4)

        # Return the dictionary
        return data_dict


# Example usage
if __name__ == "__main__":
    data_df = pd.read_csv("path/to/data.csv")
    extractor = Extractor(data_df)

    print(f"In experiment no 1, in the first frame there are {extractor.extract_cells_count(exp_id=1, frame=1)} cells")
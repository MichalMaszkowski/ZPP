import numpy as np
import math
import pandas as pd
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

    def extract_ERKKTR_ratios(
        self,
        exp_id: int,
        frame: int
    ) -> List[float]:
        """
            Extracts ratios of ERK and KTR markers in a given frame.
            Args:
                exp_id : id of the experiment
                frame : number of the frame in the given experiment
            Returns:
                List[float]: the list of ratios
        """
        filtered_data = self.data[(self.data['Exp_ID'] == exp_id) & (self.data['Image_Metadata_T'] == frame)]
        return list(filtered_data['ERKKTR_ratio'])

    def prepare_first_frame_for_simulation(self, site_filter: int = 1, exp_id: int = 1, time_step: int = 1, columns: list = None) -> pd.DataFrame:
        """
        Prepares the first frame of data for simulation with optional filters and column selection.

        :param site_filter: The site number to filter by (default is 1).
        :param exp_id: The experiment ID to filter by (default is 1).
        :param time_step: The time step (frame number) to filter by (default is 1).
        :param columns: List of column names to include in the returned DataFrame. If None, all columns are included.
        
        :return: pd.DataFrame - DataFrame containing the first frame of the simulation with selected columns.
        """
        # Filtering the dataset to get the relevant data
        df_WT = self.data[self.data['Image_Metadata_Site'] <= 4]
        df_WT_Field = df_WT[(df_WT['Image_Metadata_Site'] == site_filter) & (df_WT['Exp_ID'] == exp_id)]

        # Select the first frame (time step) and relevant columns
        df_first_frame = df_WT_Field[df_WT_Field['Image_Metadata_T'] == time_step]

        if columns:
            df_first_frame = df_first_frame[columns]

        print(f"Kolumny pierwszej ramki = {df_first_frame.columns}")
        return df_first_frame

    def get_neighbour_changes(df, exp_id, field_of_view, t_prefix):
        # Filter the DataFrame once
        df_Field = df[(df['Image_Metadata_Site'] == field_of_view) & (df['Exp_ID'] == exp_id) ].copy()

        # Create a lookup dictionary for ERKKTR_ratio by (T, track_id)
        erk_lookup = df_Field.set_index(['Image_Metadata_T', 'track_id'])['ERKKTR_ratio'].to_dict()

        # Group the DataFrame by Image_Metadata_T for faster access
        grouped_T = df_Field.groupby('Image_Metadata_T')

        # Initialize a list to collect ERK data
        ERKs = []

        # Precompute Voronoi neighbors for each time point
        neighbors_dict = {}
        for i in range(1, t_prefix):
            group = grouped_T.get_group(i) if i in grouped_T.groups else pd.DataFrame()
            points = group[['objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y']].values
            if len(points) < 2:
                neighbors_dict[i] = {}
                continue
            vor = Voronoi(points)
            
            # Initialize neighbors for each cell (1-based indexing)
            neighbors = {idx + 1: set() for idx in range(len(points))}
            for p1, p2 in vor.ridge_points:
                neighbors[p1 + 1].add(p2 + 1)
                neighbors[p2 + 1].add(p1 + 1)
            neighbors_dict[i] = neighbors

        # Iterate through each time point and compute ERK relationships
        for i in range(1, t_prefix):
            group = grouped_T.get_group(i) if i in grouped_T.groups else pd.DataFrame()
            if group.empty:
                continue
            
            neighbors = neighbors_dict.get(i, {})
            
            for cell in group['track_id']:
                my_erk = erk_lookup.get((i, cell))
                if my_erk is None:
                    continue
                
                neighbor_erk = []
                
                # Get my ERK at t-1
                my_erk_t_minus_1 = erk_lookup.get((i - 1, cell))
                if my_erk_t_minus_1 is not None:
                    neighbor_erk.append(my_erk_t_minus_1)
                
                # Get neighbors' ERK at t-1
                for neighbor in neighbors.get(cell, []):
                    neighbor_val = erk_lookup.get((i - 1, neighbor))
                    if neighbor_val is not None:
                        neighbor_erk.append(neighbor_val)
                
                if neighbor_erk:
                    neighbor_erk_mean = np.mean(neighbor_erk)
                    ERKs.append([my_erk, neighbor_erk_mean, i])

        # Create the final DataFrame
        df_ERK = pd.DataFrame(ERKs, columns=['my_erk', 'neighbor_erk', 'time']).dropna()

        # Reset index if needed
        df_ERK.reset_index(drop=True, inplace=True)
        return df_ERK



# Example usage
if __name__ == "__main__":
    data_df = pd.read_csv("path/to/data.csv")
    extractor = Extractor(data_df)

    print(f"In experiment no 1, in the first frame there are {extractor.extract_cells_count(exp_id=1, frame=1)} cells")



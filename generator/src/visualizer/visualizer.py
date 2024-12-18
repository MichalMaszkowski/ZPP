import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, List, Optional
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
import networkx as nx

def visualize_points( 
        points: List[Tuple[int, int]], 
        height: int = 1024, 
        width: int = 1024
    ) -> None:
        """
        Visualizes the distributed points on the board.

        Args:
            points (List[Tuple[int, int]]): A list of points to visualize.
            height (int): The height of the board (default is 1024).
            width (int): The width of the board (default is 1024).
        """
        if not points:
            raise ValueError("The list of points is empty. Cannot visualize.")

        x_coords, y_coords = zip(*points)  # Unpack points into x and y coordinates

        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, c='blue', s=10, alpha=0.6)
        plt.title("Points Visualization on the Board")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-20, width + 20)
        plt.ylim(-20, height + 20)
        plt.grid(True)
        plt.show()

def visualize_voronoi_field(cell_positions, field_size, min_distance):
    """
    Generates a visualization of a Voronoi field with neighbor relationships.

    Parameters:
        cell_positions (np.ndarray): An array of cell positions (x, y coordinates).
        field_size (float): Size of the field (both width and height).
        min_distance (float): Minimum allowed distance between neighbors.
        validate_min_distance (callable): A function to validate the minimum distance constraint.

    Returns:
        dict: A dictionary with calculated statistics about the field.
    """
    # Create the Voronoi diagram
    vor = Voronoi(cell_positions)

    # Create a graph based on Voronoi neighbors
    graph = nx.Graph()
    for point_idx, neighbors in enumerate(vor.ridge_points):
        graph.add_edge(*neighbors)

    # Validate the field of view
    # is_valid = validate_min_distance(cell_positions, graph, min_distance)
    # assert is_valid, "Some neighbors violate the minimum distance constraint!"

    # Calculate distances
    distances = distance_matrix(cell_positions, cell_positions)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances
    min_distances = distances.min(axis=1)  # Distance to nearest neighbor
    mean_min_distance = np.mean(min_distances)
    min_distance_overall = np.min(min_distances)
    max_distance_overall = np.max(min_distances)

    # Calculate the average number of neighbors per cell from the Voronoi graph
    num_neighbors = [len(list(graph.neighbors(node))) for node in graph.nodes]
    avg_num_neighbors = np.mean(num_neighbors)

    # Plot the field of view and graph
    plt.figure(figsize=(10, 10))
    voronoi_plot_2d(vor, show_vertices=False, line_colors='gray', line_width=0.5, line_alpha=0.2, point_size=2)
    nx.draw(graph, pos={i: cell_positions[i] for i in range(len(cell_positions))}, 
            node_size=10, edge_color="blue", alpha=0.2, with_labels=False)
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    plt.title(
        f"Field of View with Cells and Voronoi Neighbors\n"
        f"Mean NN Distance: {mean_min_distance:.2f}, Min: {min_distance_overall:.2f}, Max: {max_distance_overall:.2f}\n"
        f"Avg. Number of Neighbors: {avg_num_neighbors:.2f}"
    )
    plt.show()

    # Return calculated statistics
    return {
        "mean_min_distance": mean_min_distance,
        "min_distance_overall": min_distance_overall,
        "max_distance_overall": max_distance_overall,
        "avg_num_neighbors": avg_num_neighbors
    }
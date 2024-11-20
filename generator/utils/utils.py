import numpy as np
from scipy.stats import truncnorm
import networkx as nx

def validate_min_distance(cell_positions, graph, min_distance):
    """
    Validates if no neighboring cells violate the minimum distance constraint.

    Parameters:
        cell_positions: np.ndarray of cell positions (N x 2)
        graph: networkx.Graph representing the Voronoi neighbors
        min_distance: Minimum allowed distance between neighboring cells
    Returns:
        bool: True if all neighbors satisfy the minimum distance, False otherwise
    """
    violations = []

    for edge in graph.edges:
        cell1, cell2 = edge
        distance = np.linalg.norm(cell_positions[cell1] - cell_positions[cell2])

        if distance < min_distance:
            violations.append((cell1, cell2, distance))

    if violations:
        print("Violations found:")
        for v in violations:
            print(f"Cells {v[0]} and {v[1]} are {v[2]:.2f} pixels apart (violates min_distance = {min_distance})")

        return False

    print("All neighboring cells satisfy the minimum distance constraint.")
    return True

# Helper function to generate truncated normal samples
def generate_truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a, b, loc=mean, scale=std).rvs(size)
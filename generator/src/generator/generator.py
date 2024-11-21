import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, List, Optional
import src.extractor.extractor as extractor
import src.visualizer.visualizer as visualizer



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

# Example usage
if __name__ == "__main__":
    generator = Generator()

    try:
        # Generate points
        points = generator.deploy_points(height=1024, width=1024, points_num=1128, min_dist=23)

        # Visualize the points
        visualizer.visualize_points(points)
        visualizer.visualize_points(generator.update_points_positions(points))
        visualizer.visualize_voronoi_field(points, 1024, 23)
    except ValueError as e:
        print(f"Error: {e}")

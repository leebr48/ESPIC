"""Defines Uniform1DGrid, which sets up a one-dimensional spatial grid, and Uniform2DGrid, which sets up a two-dimensional spatial grid."""

import numpy as np


class Uniform1DGrid:
    """
    Creates a 1D spatial grid with uniform spacing.

    Inputs:
        num_points(int): Number of grid points.
        x_min(float): Least extent of the grid.
        x_max(float): Greatest extent of the grid.

    Derived Attributes:
        grid(np.ndarray): The coordinates of the spatial grid nodes.
        size(int): Number of elements in grid.
        shape(tuple): Dimensions of each axis of grid.
    """

    def __init__(
        self, num_points: int = 100, x_min: float = -1, x_max: float = 1
    ) -> None:
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.grid = np.linspace(self.x_min, self.x_max, self.num_points)
        self.size = self.grid.size
        self.shape = self.grid.shape


class Uniform2DGrid:
    """
    Creates a 2D spatial grid with uniform spacing in each dimension.

    Inputs:
        num_points(int): Number of grid points for each dimension.
        x_min(float): Least extent of the first grid.
        x_max(float): Greatest extent of the first grid.
        y_min(float): Least extent of the second grid.
        y_max(float): Greatest extent of the second grid.

    Derived Attributes:
        grid(np.ndarray): The coordinates of the spatial grid nodes.
        size(int): Number of elements in grid.
        shape(tuple): Dimensions of each axis of grid.
    """

    def __init__(
        self,
        num_points: int = 100,
        x_min: float = -1,
        x_max: float = 1,
        y_min: float = -1,
        y_max: float = 1,
    ) -> None:
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_grid = np.linspace(self.x_min, self.x_max, self.num_points)
        self.y_grid = np.linspace(self.y_min, self.y_max, self.num_points)
        self.grid = np.meshgrid(self.x_grid, self.y_grid)
        self.size = (
            self.x_grid.size * self.y_grid.size
        )  # FIXME why not take from grid directly?
        self.shape = (
            self.x_grid.shape,
            self.y_grid.shape,
        )  # FIXME why not take from grid directly? Even if not, why use a tuple of tuples?

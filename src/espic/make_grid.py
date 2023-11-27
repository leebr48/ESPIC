"""Defines Uniform1DGrid, which sets up a one-dimensional spatial grid."""

import numpy as np


class Uniform1DGrid:
    def __init__(self, num_points=100, x_min=-1, x_max=1):
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.grid = np.linspace(self.x_min, self.x_max, self.num_points)
        self.size = self.grid.size
        self.shape = self.grid.shape


class Uniform2DGrid:
    def __init__(self, num_points=100, x_min=-1, x_max=1, y_min=-1, y_max=1):
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x_grid = np.linspace(self.x_min, self.x_max, self.num_points)
        self.y_grid = np.linspace(self.y_min, self.y_max, self.num_points)

        self.grid = np.meshgrid(self.x_grid, self.y_grid)

        self.size = self.x_grid.size * self.y_grid.size
        self.shape = (self.x_grid.shape, self.y_grid.shape)

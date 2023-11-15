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

"""Defines Uniform1DGrid, which sets up a one-dimensional spatial grid."""

import numpy as np


class Uniform1DGrid:
    def __init__(self, num_points=100, x0=-1, xm=1):
        self.num_points = num_points
        self.x0 = x0
        self.xm = xm

        self.grid = np.linspace(self.x0, self.xm, self.num_points)

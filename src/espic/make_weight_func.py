"""Defines ChargeWeightFunc, which allows charges to be weighted appropriately for deposition on the spatial grid."""

import numpy as np


class ChargeWeightFunc:
    def __init__(self, evalutation_point, grid_point, spatial_grid_delta):
        self.evalutation_point = evalutation_point
        self.grid_point = grid_point
        self.spatial_grid_delta = spatial_grid_delta

    def zeroth_order(self):
        if (
            np.abs(self.grid_point - self.evalutation_point)
            < self.spatial_grid_delta / 2
        ):
            return 1 / self.spatial_grid_delta
        return 0

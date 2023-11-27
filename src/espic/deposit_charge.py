"""Defines ChargeDeposition class, which places charges at float positions on the spatial grid."""

import numpy as np

from espic.make_grid import Uniform1DGrid
from espic.make_weight_func import ChargeWeightFunc


class ChargeDeposition:
    def __init__(self, shape_func="zeroth_order", grid=Uniform1DGrid()):
        self.shape_func = shape_func
        self.grid = grid

    def return_coords(self, grid):
        if len(self.grid.shape) == 1:
            coords = self.grid.grid.reshape((len(self.grid.grid), 1))
        else:
            c = ()
            for i in range(len(self.grid.grid)):
                c = c + (self.grid.grid[len(self.grid.grid) - (i + 1)].ravel(),)
            coords = np.column_stack(c)
        return coords

    def deposit(self, q_arr, pos_arr):
        if len(pos_arr.shape) == 1:
            pos_arr = pos_arr.reshape((len(pos_arr), 1))
        else:
            pos_arr = pos_arr.reshape((len(pos_arr), len(pos_arr[0])))
        rho = np.zeros(self.grid.size)

        coords = self.return_coords(self.grid)

        dist = np.linalg.norm(coords[:, None] - pos_arr[None, :], axis=2)
        for i in range(len(pos_arr)):
            disti = dist[:, i]
            rho += ChargeWeightFunc(disti, self.grid.delta).zeroth_order() * q_arr[i]

        rho = rho.reshape(self.grid.shape)

        return rho

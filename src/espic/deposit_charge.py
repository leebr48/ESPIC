"""Defines ChargeDeposition class, which places charges at float positions on the spatial grid."""

import numpy as np
from make_grid import Uniform1DGrid
from make_weight_funcs import ChargeWeightFunc


class ChargeDeposition:
    def __init__(self, shape_func="zeroth_order", grid=Uniform1DGrid()):
        self.shape_func = ChargeWeightFunc.shape_func
        self.grid = grid
        self.delta = (
            grid[1] - grid[0]
        )  # Assumes uniform grid. FIXME fix later for arbitrary grid

    def deposit(self, qarr, xarr):
        rho = np.zeros(len(self.grid))

        for i in range(len(rho)): #FIXME should be vectorized with numpy if possible, possibly in ChargeWeightFunc
            for j in range(len(xarr)):
                rho[i] += self.shape_func(xarr[j], self.grid[i], self.delta) * qarr[j]

        return rho

    def zeroth(self, Xi, xj, delta):
        if np.abs(xj - Xi) < delta / 2:
            return 1 / delta
        else:
            return 0

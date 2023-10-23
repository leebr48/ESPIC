"""Defines ChargeDeposition class, which places charges at float positions on the spatial grid."""

import numpy as np
from make_grid import Uniform1DGrid


class ChargeDeposition:
    def __init__(self, shape="zeroth", grid=Uniform1DGrid()):
        self.shape = shape
        self.grid = grid
        self.delta = (
            grid[1] - grid[0]
        )  # Assumes uniform grid. FIXME fix later for arbitrary grid

    def deposit(self, qarr, xarr):
        rho = np.zeros(len(self.grid))

        for i in range(len(rho)):
            for j in range(len(xarr)):
                rho[i] += self.shape_func(xarr[j], self.grid[i], self.delta) * qarr[j]

        return rho

    def choose_shape(self):
        if self.shape == "zeroth":
            self.shape_func = self.zeroth

    def zeroth(self, Xi, xj, delta):
        if np.abs(xj - Xi) < delta / 2:
            return 1 / delta
        else:
            return 0

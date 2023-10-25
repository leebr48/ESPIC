"""Defines ChargeDeposition class, which places charges at float positions on the spatial grid."""

import numpy as np

from .make_grid import Uniform1DGrid
from .make_weight_func import ChargeWeightFunc


class ChargeDeposition:
    def __init__(self, shape_func="zeroth_order", grid_array=Uniform1DGrid()):
        self.shape_func = shape_func
        self.grid = grid_array.grid
        self.delta = (
            self.grid[1] - self.grid[0]
        )  # Assumes uniform grid. FIXME fix later for arbitrary grid

    def deposit(self, qarr, xarr):
        rho = np.zeros(len(self.grid))

        # FIXME loop should be vectorized with numpy if possible, possibly in ChargeWeightFunc
        # FIXME this will hopefully make the function call less disgusting
        for i in range(len(rho)):
            for j in range(len(xarr)):
                rho[i] += (
                    getattr(
                        ChargeWeightFunc(xarr[j], self.grid[i], self.delta),
                        self.shape_func,
                    )()
                    * qarr[j]
                )

        return rho

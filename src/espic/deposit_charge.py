"""Defines ChargeDeposition class, which places charges at float positions on the spatial grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from espic.make_grid import Uniform1DGrid, Uniform2DGrid
from espic.make_weight_func import ChargeWeightFunc

FArray = NDArray[np.float64]


class ChargeDeposition:
    def __init__(
        self,
        shape_func: str = "zeroth_order",
        grid: Uniform1DGrid | Uniform2DGrid = Uniform1DGrid(),
    ) -> None:
        self.shape_func = shape_func
        self.grid = grid

    def return_coords(self, grid: Uniform1DGrid | Uniform2DGrid) -> FArray:
        if len(self.grid.shape) == 1:
            coords = np.array(self.grid.grid).reshape((len(self.grid.grid), 1))
        else:
            c: tuple[FArray, ...] = ()
            for i in range(len(self.grid.grid)):
                c = c + (self.grid.grid[len(self.grid.grid) - (i + 1)].ravel(),)
            coords = np.column_stack(c)
        return coords

    def deposit(self, q_arr: FArray, pos_arr: FArray) -> FArray:
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

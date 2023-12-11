"""
Defines ChargeDeposition class, which places
charges at float positions on the spatial grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from espic.make_grid import Uniform1DGrid, Uniform2DGrid
from espic.make_weight_func import ChargeWeightFunc

FArray = NDArray[np.float64]


class ChargeDeposition:
    """
    Computes the charge density on grid points using a particular shape function.
    Basically splits the charge of a particle between the nearest grid points using
    the rule defined by the shape function.

    Parameters
    ----------
    shape_func
        String indicating which shape function to use. Shape functions are defined in
        make_weight_func.py
    grid
        ``Uniform1DGrid`` or ``Uniform2DGrid`` defining the grid points where
        the charge density will be calculated.
    """

    def __init__(
        self,
        shape_func: str = "zeroth_order",
        grid: Uniform1DGrid | Uniform2DGrid = Uniform1DGrid,  # type: ignore[assignment]
    ) -> None:
        self.shape_func = shape_func
        self.grid = grid

        if type(self.grid) == type:  # type: ignore[comparison-overlap]
            self.grid = self.grid()

    def return_coords(self, grid: Uniform1DGrid | Uniform2DGrid) -> FArray:
        """
        Return the coordinate pairs from the grid.
        For a 1D grid, just returns a column vector with each x point.
        For a 2D grid, it returns all the possible (x,y) coordinates.

        Parameters
        ----------
        grid
            The grid used to calculate the charge density.

        Returns
        -------
            The different coordinate pairs.

        """
        if len(grid.shape) == 1:
            coords = np.array(grid.grid).reshape((len(grid.grid), 1))
        else:
            c: tuple[FArray, ...] = ()
            for i in range(len(grid.grid)):
                c = (*c, grid.grid[len(grid.grid) - (i + 1)].ravel())
            coords = np.column_stack(c)
        return coords

    def deposit(self, q_arr: FArray, pos_arr: FArray) -> FArray:
        """
        Compute the charge density on the given grid.

        Parameters
        ----------
        q_arr
            The charges of each particle in the simulation.
        pos_arr
            The positions of each particle in the simulation.

        Returns
        -------
            The charge density for each point in the grid.

        """
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

        return rho.reshape(self.grid.shape)

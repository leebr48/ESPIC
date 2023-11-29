"""Defines InterpolatedField, which allows for the conversion of the potential on the spatial grid to the electric field at an arbitrary point in space."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from espic.make_grid import Uniform1DGrid

FArray = NDArray[np.float64]


class InterpolatedField:
    """
    Provides the ability to convert the on-grid electric potential into an
    electric field at an arbitrary point in space.

    Inputs:
        grids(list(Uniform1DGrid)): List of the grids on which the potential is defined.
        phi_on_grid(numpy.ndarray): Array of floats with phi_on_grid.ndim == len(grids).
                                    Note that this array must be indexed Pythonically.
                                    In particular, if it is generated with numpy.meshgrid,
                                    the indexing='ij' option MUST be used.
    """

    def __init__(self, grids: list[Uniform1DGrid], phi_on_grid: FArray) -> None:
        self.grids = [g.grid for g in grids]
        self.phi_on_grid = phi_on_grid

        # We need to ensure that E_on_grid is a list of arrays
        if len(self.grids) == 1:
            self.E_on_grid = [-1 * np.gradient(self.phi_on_grid, *self.grids)]
        else:
            self.E_on_grid = [
                -1 * ar for ar in np.gradient(self.phi_on_grid, *self.grids)
            ]

        # We can interpolate the electric field using the PChip algorithm because it does not overshoot,
        # which is quite important when working with electric fields and potentials
        self.interpolated_E = [
            RegularGridInterpolator(self.grids, ar, method="pchip")
            for ar in self.E_on_grid
        ]

    def evaluate(self, coords: ArrayLike) -> FArray:
        """
        Evalutate the electric field at an arbritrary point in space.

        Inputs:
            coords(array_like): Coordinates at which to evaluate the electric field.
                                Can be either a single evaluation point or an array
                                of evaluation points.
                                Must have no more axes than phi_on_grid. (That is, a
                                2D field cannot be derived from a 1D potential, etc.)
        Outputs:
            np.ndarray with values for the interpolated field. The dimensionality
            will match that of coords.
        """
        return np.asarray([interp(coords) for interp in self.interpolated_E]).T

"""
Defines ``InterpolatedField``, which allows for the conversion of the potential
on the spatial grid to the electric field at an arbitrary point in space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import ArrayLike, NDArray

    from espic.make_grid import Uniform1DGrid

    FArray = NDArray[np.float64]


class InterpolatedField:
    """
    Provides the ability to convert the on-grid electric potential into an
    electric field at an arbitrary point in space.

    Parameters
    ----------
    grids
        List of the grids on which the potential is defined.
    phi_on_grid
        Array of floats with ``phi_on_grid.ndim == len(grids)``.
        Note that this array must be indexed Pythonically.
        In particular, if it is generated with ``numpy.meshgrid``,
        the ``indexing='ij'`` option **must** be used.
    omega_p
        Plasma frequency in inverse seconds.
    c
        Speed of light in meters per second.
    normalize
        If False, perform calculations in "raw" units. If True,
        normalize equations using the natural units specified
        by ``omega_p`` and ``c``.
    """

    def __init__(
        self,
        grids: list[Uniform1DGrid],
        phi_on_grid: FArray,
        omega_p: float = 1,
        c: float = 1,
        normalize: bool = False,
    ):
        self.grids = [g.grid for g in grids]
        self.phi_on_grid = phi_on_grid
        self.omega_p = omega_p
        self.c = c
        self.normalize = normalize

    @property
    def e_on_grid(self) -> list[FArray]:
        """
        Electric field on the spatial grid, derived from finite difference
        gradients of the potential on the spatial grid. Implemented as
        a list of array objects. The index of the list corresponds
        to the spatial dimension, and each array gives the
        electric field in the direction corresponding to the list index.
        (For instance, in the 2D case, this object would contain
        :math:`[E_{x}, E_{y}]`).
        """
        # We need to ensure that e_on_grid is a list of arrays
        if len(self.grids) == 1:
            if self.normalize:
                return [
                    -self.c / self.omega_p * np.gradient(self.phi_on_grid, *self.grids),
                ]
            return [-1 * np.gradient(self.phi_on_grid, *self.grids)]

        if self.normalize:
            return [
                -self.c / self.omega_p * ar
                for ar in np.gradient(self.phi_on_grid, *self.grids)
            ]
        return [-1 * ar for ar in np.gradient(self.phi_on_grid, *self.grids)]

    @property
    def interpolated_e(self) -> list[Callable[[ArrayLike], FArray]]:
        """
        Interpolator that allows for the electric field to be computed at an
        arbitrary point in space. The interpolation is based on ``e_on_grid``.
        Implemented as a list of interpolation functions. The index of the
        list corresponds to the spatial dimension, and each interpolation
        function gives the electric field in the direction corresponding to
        the list index. (For instance, in the 2D case, this object would contain
        :math:`[E_{x}, E_{y}]`).
        """
        # We can interpolate the electric field using the PCHIP algorithm because it
        # does not overshoot, which is quite important when working with electric
        # fields and potentials.
        return [
            RegularGridInterpolator(self.grids, ar, method="pchip")
            for ar in self.e_on_grid
        ]

    def evaluate(self, coords: ArrayLike) -> FArray:
        """
        Evalutate the electric field at an arbritrary point in space.

        Parameters
        ----------
        coords
            Coordinates at which to evaluate the electric field.
            Can be either a single evaluation point or an array
            of evaluation points.

        Returns
        -------
            Array with values for the interpolated field. The
            dimensionality will match that of ``coords``.
        """
        return np.asarray([interp(coords) for interp in self.interpolated_e]).T

    def update_potential(self, new_phi_on_grid: FArray) -> None:
        """
        Update the electric potential.

        Parameters
        ----------
        new_phi_on_grid
            The new potential to be used.
        """
        self.phi_on_grid = new_phi_on_grid

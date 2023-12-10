"""
Implements solvers in 1D and 2D for Poisson's equation for the electrostatic
potential given the charge density.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import epsilon_0
from scipy.linalg import solve, solve_banded

from espic.make_grid import Uniform1DGrid, Uniform2DGrid

FArray = NDArray[np.float64]


class MaxwellSolver1D:
    """
    Solves Poisson's equation in 1D.

    Parameters
    ----------
    boundary_conditions
        Array indicating boundary conditions.
    grid
        ``Uniform1DGrid`` defining the grid points where the potential
        will be calculated.
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
        boundary_conditions: FArray = np.zeros(2),
        grid: Uniform1DGrid = Uniform1DGrid(),
        omega_p: float = 1,
        c: float = 1,
        normalize: bool = False,
    ) -> None:
        self.grid = grid.grid
        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros(len(self.grid))
        self.omega_p = omega_p
        self.c = c
        self.normalize = normalize

    # Centered differences. FIXME should we make it arbitrary?
    def solve(self, rho: FArray) -> FArray:
        """
        Solves the 1D Poisson's equation for a given charge distribution.

        Parameters
        ----------
        rho
            The charge density defined on the grid points.

        Returns
        -------
            The electrostatic potential phi on the grid points.

        """
        delta = self.grid[1] - self.grid[0]
        dim = len(self.grid) - 2
        phi = np.zeros(len(self.grid))
        phi[0] = self.boundary_conditions[0]
        phi[-1] = self.boundary_conditions[1]

        bands = np.empty((3, dim))
        bands[0, 1:] = np.ones(dim - 1)
        bands[1, :] = -2 * np.ones(dim)
        bands[2, :-1] = np.ones(dim - 1)

        rho = rho[1:-1]
        bc = np.zeros(dim)
        bc[0] = self.boundary_conditions[0]
        bc[-1] = self.boundary_conditions[1]
        rhs = -(delta**2) * rho / (epsilon_0) + bc

        # For some reason, not setting check_finite=False results in NaN errors?
        # Might be due to how small the values are, which is a separate issue.
        phi[1:-1] = solve_banded((1, 1), bands, rhs, check_finite=False)

        if self.normalize:
            return self.c / self.omega_p * phi
        return phi


# Code taken from https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2009%20-%20Elliptic%20Equations/903_Poisson%20Equation-Boundary.html
# FIXME: for now, this assumes equal spacing in x and y.
class MaxwellSolver2D:
    """
    Solves Poisson's equation in 2D.

    Parameters
    ----------
    boundary_conditions
        Dictionary of array containing boundary conditions.
    grid
        ``Uniform2DGrid`` defining the grid points where the potential
        will be calculated.
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
        boundary_conditions: dict[str, FArray] | None = None,
        grid: Uniform2DGrid = Uniform2DGrid(),
        omega_p: float = 1,
        c: float = 1,
        normalize: bool = False,
    ) -> None:
        self.grid = grid.grid
        self.x_grid = grid.x_grid
        self.y_grid = grid.y_grid
        self.omega_p = omega_p
        self.c = c
        self.normalize = normalize

        if boundary_conditions is None:
            n = len(self.x_grid)
            boundary_conditions = {
                "bottom": np.zeros(n),
                "top": np.zeros(n),
                "left": np.zeros(n),
                "right": np.zeros(n),
            }

        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros((len(self.grid), len(self.grid)))

    @cached_property
    def a(self) -> FArray:
        """
        Returns the :math:`a` matrix in :math:`a * phi = b` that will be
        inverted to solve for :math:`a`. # FIXME solve for a or phi?
        It contains the coffeicients that arise from the finite-differencing
        scheme.

        Returns
        -------
            The :math:`a` matrix in :math:`a * phi = b`.

        """
        # It's better to set a in the initialization, since it doesn't change over time.
        return self.set_a(len(self.x_grid))

    def set_a(self, n: int) -> FArray:
        """
        The loops used to initialize the :math:`a` matrix in :math:`a * phi = b`.

        Parameters
        ----------
        n
            DESCRIPTION. # FIXME

        Returns
        -------
            The :math:`a` matrix in :math:`a * phi = b`.

        """
        n2 = (n - 2) * (n - 2)
        a = np.zeros((n2, n2))
        ## Diagonal
        for i in range(n - 2):
            for j in range(n - 2):
                a[i + (n - 2) * j, i + (n - 2) * j] = -4

        # LOWER DIAGONAL
        for i in range(1, n - 2):
            for j in range(n - 2):
                a[i + (n - 2) * j, i + (n - 2) * j - 1] = 1
        # UPPPER DIAGONAL
        for i in range(n - 3):
            for j in range(n - 2):
                a[i + (n - 2) * j, i + (n - 2) * j + 1] = 1

        # LOWER IDENTITY MATRIX
        for i in range(n - 2):
            for j in range(1, n - 2):
                a[i + (n - 2) * j, i + (n - 2) * (j - 1)] = 1

        # UPPER IDENTITY MATRIX
        for i in range(n - 2):
            for j in range(n - 3):
                a[i + (n - 2) * j, i + (n - 2) * (j + 1)] = 1

        return a

    def set_rhs(self, n: int, h: float, rho: FArray, bc: dict[str, FArray]) -> FArray:
        """
        Sets the :math:`b` vector in :math:`a * phi = b`.
        It contains information about the charge density
        and the boundary conditions.

        Parameters
        ----------
        n
            The number of grid points.
        h
            The spacing between grid points.
        rho
            The charge density on grid points.
        bc
            A dictionary containing the boundary conditions.

        Returns
        -------
            The :math:`b` vector in :math:`a * phi = b`.

        """
        n2 = (n - 2) * (n - 2)
        rho = rho[1:-1, 1:-1]
        rho_v = rho.ravel()

        r = np.zeros(n2)

        r = -(h**2) * rho_v / epsilon_0
        bc = self.boundary_conditions

        # Boundary
        b_bottom_top = np.zeros(n2)
        for i in range(n - 2):
            b_bottom_top[i] = bc["bottom"][i]  # Bottom Boundary
            b_bottom_top[i + (n - 2) * (n - 3)] = bc["top"][i]  # Top Boundary

        b_left_right = np.zeros(n2)
        for j in range(n - 2):
            b_left_right[(n - 2) * j] = bc["left"][j]  # Left Boundary
            b_left_right[n - 3 + (n - 2) * j] = bc["right"][j]  # Right Boundary

        b = b_left_right + b_bottom_top

        return r - b

    def solve(self, rho: FArray) -> FArray:
        """
        Solves the 1D Poisson's equation for a given charge distribution.

        Parameters
        ----------
        rho
            The charge density on grid points.

        Returns
        -------
            The electrostatic potential evaluated on grid points.

        """
        gridsize = len(self.x_grid)
        h = self.x_grid[1] - self.x_grid[0]
        rhs = self.set_rhs(gridsize, h, rho, self.boundary_conditions)

        # This is still a banded matrix, but now the locations of
        # the bands depends on n. Will make more efficient later. # FIXME true?
        phi_v = solve(self.a, rhs)
        phi = np.zeros((gridsize, gridsize))

        # Apply bc. Have to flip top and bottom bc because phi matrix goes down to up
        phi[:, -1] = self.boundary_conditions["right"]
        phi[:, 0] = self.boundary_conditions["left"]
        phi[-1, :] = self.boundary_conditions["top"]
        phi[0, :] = self.boundary_conditions["bottom"]

        phi[1 : gridsize - 1, 1 : gridsize - 1] = phi_v.reshape(
            (gridsize - 2, gridsize - 2),
        )

        if self.normalize:
            return self.c / self.omega_p * phi
        return phi

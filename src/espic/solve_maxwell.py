"""Defines MaxwellSolver1D, which calculates the electric field on the spatial grid."""
from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.constants import epsilon_0
from scipy.linalg import solve, solve_banded

from espic.make_grid import Uniform1DGrid, Uniform2DGrid

FArray = NDArray[np.float64]


class MaxwellSolver1D:
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
        # rhs = -4 * np.pi * delta**2 * rho + bc
        rhs = -(delta**2) * rho / epsilon_0 + bc

        phi[1:-1] = solve_banded((1, 1), bands, rhs)

        if self.normalize:
            return self.c / self.omega_p * phi
        else:
            return phi


# Code taken from https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2009%20-%20Elliptic%20Equations/903_Poisson%20Equation-Boundary.html
# FIXME: for now, this assumes equal spacing in x and y.
class MaxwellSolver2D:
    def __init__(
        self,
        boundary_conditions: dict[str, FArray] | None = None,
        grid: Uniform2DGrid = Uniform2DGrid(),
    ) -> None:
        self.grid = grid.grid
        self.x_grid = grid.x_grid
        self.y_grid = grid.y_grid
        if boundary_conditions is None:
            N = len(self.x_grid)
            boundary_conditions = {
                "bottom": np.zeros(N),
                "top": np.zeros(N),
                "left": np.zeros(N),
                "right": np.zeros(N),
            }

        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros((len(self.grid), len(self.grid)))

    @cached_property
    def A(self) -> FArray:
        # It's better to set A in the initialization, since it doesn't change over time.
        return self.set_A(len(self.x_grid))

    def set_A(self, N: int) -> FArray:
        N2 = (N - 2) * (N - 2)
        A = np.zeros((N2, N2))
        ## Diagonal
        for i in range(N - 2):
            for j in range(N - 2):
                A[i + (N - 2) * j, i + (N - 2) * j] = -4

        # LOWER DIAGONAL
        for i in range(1, N - 2):
            for j in range(N - 2):
                A[i + (N - 2) * j, i + (N - 2) * j - 1] = 1
        # UPPPER DIAGONAL
        for i in range(N - 3):
            for j in range(N - 2):
                A[i + (N - 2) * j, i + (N - 2) * j + 1] = 1

        # LOWER IDENTITY MATRIX
        for i in range(N - 2):
            for j in range(1, N - 2):
                A[i + (N - 2) * j, i + (N - 2) * (j - 1)] = 1

        # UPPER IDENTITY MATRIX
        for i in range(N - 2):
            for j in range(N - 3):
                A[i + (N - 2) * j, i + (N - 2) * (j + 1)] = 1

        return A

    def set_rhs(self, N: int, h: float, rho: FArray, bc: dict[str, FArray]) -> FArray:
        N2 = (N - 2) * (N - 2)
        #        rho = np.ones((N-1,N-1))
        rho = rho[1:-1, 1:-1]
        rho_v = rho.ravel()

        r = np.zeros(N2)

        r = -4 * np.pi * h**2 * rho_v
        bc = self.boundary_conditions

        # Boundary
        b_bottom_top = np.zeros(N2)
        for i in range(N - 2):
            b_bottom_top[i] = bc["bottom"][i]  # Bottom Boundary
            b_bottom_top[i + (N - 2) * (N - 3)] = bc["top"][i]  # Top Boundary

        b_left_right = np.zeros(N2)
        for j in range(N - 2):
            b_left_right[(N - 2) * j] = bc["left"][j]  # Left Boundary
            b_left_right[N - 3 + (N - 2) * j] = bc["right"][j]  # Right Boundary

        b = b_left_right + b_bottom_top

        rhs = r - b
        return rhs

    def solve(self, rho: FArray) -> FArray:
        gridsize = len(self.x_grid)
        h = self.x_grid[1] - self.x_grid[0]
        rhs = self.set_rhs(gridsize, h, rho, self.boundary_conditions)

        # This is still a banded matrix, but now the locations of the bands depends on N. Will make more efficient later.
        phi_v = solve(self.A, rhs)
        phi = np.zeros((gridsize, gridsize))

        # Apply bc. Have to flip top and bottom bc because phi matrix goes down to up
        phi[:, -1] = self.boundary_conditions["right"]
        phi[:, 0] = self.boundary_conditions["left"]
        phi[-1, :] = self.boundary_conditions["top"]
        phi[0, :] = self.boundary_conditions["bottom"]

        phi[1 : gridsize - 1, 1 : gridsize - 1] = phi_v.reshape(
            (gridsize - 2, gridsize - 2),
        )
        return phi

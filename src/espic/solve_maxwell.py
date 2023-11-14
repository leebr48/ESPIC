"""Defines MaxwellSolver1D, which calculates the electric field on the spatial grid."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve

from espic.make_grid import Uniform1DGrid, Uniform2DGrid


class MaxwellSolver1D:
    def __init__(self, grid=Uniform1DGrid(), boundary_conditions=np.zeros(2)):
        self.grid = grid.grid
        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros(len(self.grid))

    # Centered differences. FIXME should we make it arbitrary?
    def solve(self, rho):
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
        rhs = -4 * np.pi * delta**2 * rho + bc

        phi[1:-1] = solve_banded((1, 1), bands, rhs)

        return phi


# Code taken from https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2009%20-%20Elliptic%20Equations/903_Poisson%20Equation-Boundary.html
# FIXME: for now, this assumes equal spacing in x and y.
class MaxwellSolver2D:
    def boundary_zero(grid):
        x = np.linspace(0, 1, len(grid))

        grid[0, :] = np.interp(x, [0, 1], [0, 1])
        grid[:, -1] = np.interp(x, [0, 1], [1, 0])
        grid[-1, :] = np.interp(x, [0, 1], [-1, 0])
        grid[:, 0] = np.interp(x, [0, 1], [0, -1])

    def __init__(self, grid=Uniform2DGrid(), boundary_conditions=None):
        self.grid = grid.grid
        self.xgrid = grid.xgrid
        self.ygrid = grid.ygrid

        if boundary_conditions is None:
            N = len(self.xgrid)
            boundary_conditions = {
                "bottom": np.zeros(N),
                "top": np.zeros(N),
                "left": np.zeros(N),
                "right": np.zeros(N),
            }

        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros((len(self.grid), len(self.grid)))

    def set_A(self, N):
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

    def set_rhs(self, N, h, rho, bc):
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

    def solve(self, rho):
        gridsize = len(self.xgrid)
        A = self.set_A(gridsize)

        h = self.xgrid[1] - self.xgrid[0]
        rhs = self.set_rhs(gridsize, h, rho, self.boundary_conditions)

        phi_v = spsolve(A, rhs)
        phi = np.zeros((gridsize, gridsize))
        phi[0, :] = self.boundary_conditions["top"]
        phi[:, -1] = self.boundary_conditions["right"]
        phi[-1, :] = self.boundary_conditions["bottom"]
        phi[:, 0] = self.boundary_conditions["left"]

        phi[1 : gridsize - 1, 1 : gridsize - 1] = phi_v.reshape(
            (gridsize - 2, gridsize - 2),
        )
        return phi

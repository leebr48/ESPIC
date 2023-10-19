import numpy as np
from make_grid import Uniform1DGrid
from scipy.linalg import solve_banded


class MaxwellSolver1D:
    def __init__(self, grid=Uniform1DGrid(), boundary_conditions=np.zeros(2)):
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        self.phi = np.zeros(len(self.grid))

    # Centered differences. Should we make it arbitrary?
    def solve(self, rho):
        delta = self.grid[1] - self.grid[0]
        dim = len(self.grid)
        bands = np.empty((3, dim))
        bands[0, 1:] = np.ones(dim - 1)
        bands[1, :] = -2 * np.ones(dim)
        bands[2, :-1] = np.ones(dim - 1)

        bc = np.zeros(dim)
        bc[0] = self.boundary_conditions[0]
        bc[-1] = self.boundary_conditions[1]
        rhs = -4 * np.pi * delta**2 * rho + bc

        return solve_banded((1, 1), bands, rhs)

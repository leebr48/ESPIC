"""Defines MaxwellSolver1D, which calculates the electric field on the spatial grid."""

import numpy as np
from scipy.linalg import solve_banded
from .make_grid import Uniform1DGrid


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
    

"""Defines InterpolateField, which allows for the conversion of the potential on the spatial grid to the electric field on the spatial grid."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class InterpolateField:
    def __init__(self, grids, phi_on_grid):
        # grids is a list of arrays (x or x,y), phi_on_grid is an array (x or x,y)
        # phi_on_grid indexed in the np.meshgrid way *with indexing='ij'*. The default will not work!!
        self.grids = grids
        self.phi_on_grid = phi_on_grid
        
        # We need to ensure that E_on_grid is a list of arrays
        if len(self.grids) == 1:
            self.E_on_grid = [-1 * np.gradient(self.phi_on_grid, *self.grids)]
        else:
            self.E_on_grid = [-1 * ar for ar in np.gradient(self.phi_on_grid, *self.grids)]
        
        # We can interpolate the electric field using the PChip algorithm because it does not overshoot,
        # which is quite important when working with electric fields and potentials
        self.interpolated_E = [RegularGridInterpolator(grids, ar, method='pchip') for ar in self.E_on_grid]
    
    def ev(self, coords):
        return np.asarray([interp(coords) for interp in self.interpolated_E]).T

"""Defines InterpolateField, which allows for the conversion of the potential on the spatial grid to the electric field on the spatial grid."""

import numpy as np


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
    
    def find_nearest_spatial_node(self, coords):
        # Coords should be 1D numpy array
        assert len(self.grids) == coords.size
        ids = [(np.abs(grid - coord)).argmin() for (coord, grid) in zip(coords, self.grids)]
        return tuple(ids)

    def ev(self, coords):
        coords = np.asarray(coords)
        ids = self.find_nearest_spatial_node(coords)
        return [ar[ids] for ar in self.E_on_grid]

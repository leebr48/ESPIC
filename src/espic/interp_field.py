"""Defines InterpolateField, which allows for the conversion of the potential on the spatial grid to the electric field on the spatial grid."""

import numpy as np


class InterpolateField:
    def __init__(self, grids, phi_on_grid, indexing='xy'):
        # grids is a list of arrays (x or x,y), phi_on_grid is an array (x or x,y)
        # phi_on_grid indexed in the standard np.meshgrid way, and this is what 'indexing' refers to
        self.grids = grids
        self.phi_on_grid = phi_on_grid
        self.indexing = indexing
        
        # We need to ensure that E_on_grid is a list of arrays
        if len(self.grids) == 1:
            self.E_on_grid = [-1 * np.gradient(self.phi_on_grid, *self.grids)]
        else:
            # FIXME reverse or not??? Write tests!!
            if self.indexing == 'xy':
                # We must reverse the order of the np.gradient output to match our desired form
                self.E_on_grid = list(reversed([-1 * ar for ar in np.gradient(self.phi_on_grid, *self.grids)]))
            elif self.indexing == 'ij':
                self.E_on_grid = [-1 * ar for ar in np.gradient(self.phi_on_grid, *self.grids)]
            else:
                raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    
    def find_nearest_spatial_node(self, coords):
        # Coords should be 1D numpy array
        assert len(self.grids) == coords.size
        ids = [(np.abs(grid - coord)).argmin() for (coord, grid) in zip(coords, self.grids)]
        return tuple(ids)

    def ev(self, coords):
        coords = np.asarray(coords)
        ids = self.find_nearest_spatial_node(coords)
        return [ar[ids] for ar in self.E_on_grid]

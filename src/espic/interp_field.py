# FIXME add comment

import numpy as np
from make_grid import Uniform1DGrid

class InterpolateField:

    def __init__(self, E_on_grid, particle_pos):
        self.E_on_grid = E_on_grid
        self.particle_pos = particle_pos



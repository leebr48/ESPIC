"""Implements the Particles class, which holds data for a swarm of particles."""

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Particles:
    # Note that the first index of any of these arrays specify the particle label.
    charges: np.ndarray  # FIXME Must be 1D array of ints!
    masses: np.ndarray  # FIXME Must be 1D array of positive floats!
    positions: np.ndarray  # FIXME 2D, 1+ elements in second dimension
    velocities: np.ndarray  # FIXME 2D, 1+ elements in second dimension

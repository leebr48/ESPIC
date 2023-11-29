"""Implements the Particles class, which holds data for a swarm of particles."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FArray = NDArray[np.float64]
IArray = NDArray[np.int32]


@dataclass(eq=False)
class Particles:
    """
    Store the data for a swarm of particles in a single object.
    Every attribute is either a 1D or 2D NumPy array.
    The first index specifies the particle number/label.
    The second index, if it exists, specifies vector information
    (see below for details).

    Inputs:
        charges(numpy.ndarray): 1D array of integers specifying the
                                charge of each particle
        masses(numpy.ndarray): 1D array of floats specifying the
                               mass of each particle
        positions(numpy.ndarray): 2D array of floats, with the second
                                  index specifying the position (in
                                  arbitrary-dimensional cartesian
                                  coordinates) of a given particle
        velocities(numpy.ndarray): 2D array of floats, with the second
                                   index specifying the velocity (in
                                   arbitrary-dimensional cartesian
                                   coordinates) of a given particle
    """

    charges: IArray
    masses: FArray
    positions: FArray
    velocities: FArray

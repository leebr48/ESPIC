"""Implements the ``Particles`` class, which holds data for a swarm of particles."""


import numpy as np
from numpy.typing import NDArray

FArray = NDArray[np.float64]
IArray = NDArray[np.int32]


class Particles:
    """
    Store the data for a swarm of particles in a single object.
    Every attribute is either a 1D or 2D ``numpy`` array.
    The first index specifies the particle number/label.
    The second index, if it exists, specifies vector information
    (see below for details).

    Parameters
    ----------
    charges
        1D array specifying the charge of each particle.
    masses
        1D array specifying the mass of each particle.
    positions
        2D array with the second index specifying the position (in
        arbitrary-dimensional cartesian coordinates) of a given particle.
    velocities
        2D array with the second index specifying the velocity (in
        arbitrary-dimensional cartesian coordinates) of a given particle.
    """

    def __init__(
        self,
        charges: IArray,
        masses: FArray,
        positions: FArray,
        velocities: FArray,
    ):
        self.charges = charges
        self.masses = masses
        self.positions = positions
        self.velocities = velocities

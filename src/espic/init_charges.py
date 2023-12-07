"""
Provides the ``Initialize`` class, which allows for the
positions and velocities of particles to be set up easily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import maxwell

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FArray = NDArray[np.float64]


class Initialize:
    """
    Provides convenient methods for setting up the initial
    positions and velocities of particles.

    Parameters
    ----------
    num_particles
        Number of particles in the simulation.
    num_dim
        Dimensionality of the simulation (1D, 2D, and 3D are standard).
    """

    def __init__(self, num_particles: int, num_dim: int):
        self.size = (num_particles, num_dim)

    def normal(self, mean: float = 0, stdev: float = 1.0) -> FArray:
        """
        Produce a normal distribution in the proper format for the
        ``positions`` or ``velocities`` attributes of the ``Particles``
        class.

        Parameters
        ----------
        mean
            Center of the distribution.
        stdev
            Standard deviation of the distribution.

        Returns
        -------
            Samples from a normal distribution.
        """
        return np.random.default_rng().normal(loc=mean, scale=stdev, size=self.size)

    def uniform(self, lower_bound: float, upper_bound: float) -> FArray:
        """
        Produce a uniform distribution in the proper format for the
        ``positions`` or ``velocities`` attributes of the ``Particles``
        class.

        Parameters
        ----------
        lower_bound
            Minimum value of the distribution (inclusive).
        upper_bound
            Maximum value of the distribution (exclusive).

        Returns
        -------
            Samples from a uniform distribution.
        """
        return np.random.default_rng().uniform(
            low=lower_bound,
            high=upper_bound,
            size=self.size,
        )

    def maxwellian(self, spread: float, start: float = 0) -> FArray:
        """
        Produce a maxwellian distribution in the proper format for the
        ``positions`` or ``velocities`` attributes of the ``Particles``
        class.

        Parameters
        ----------
        spread
            Width of the distribution. If the distribution is used to model
            particle velocities as in statistical mechanics, ``spread`` ==
            sqrt((Boltzmann constant) * (temperature) / (mass)).
        start
            Value at which the distribution returns zero. In statistical
            mechanics, this only occurs at a velocity of zero (unless the
            unphysical limit of infinite velocity is considered).

        Returns
        -------
            Samples from a Maxwellian distribution.
        """
        return np.asarray(maxwell.rvs(loc=start, scale=spread, size=self.size))

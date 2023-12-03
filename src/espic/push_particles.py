"""
Implements the ``ParticlePusher`` class, which evolves
particle positions and velocities forward one time step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from espic.def_particles import Particles
    from espic.interp_field import InterpolatedField

    FArray = NDArray[np.float64]


class ParticlePusher:
    """
    Evolves particles one time step at a time.

    Parameters
    ----------
    particles
        ``Particles`` object containing all particles to be moved.
    electric_field
        ``InterpolatedField`` object containing the field that will
        accelerate the particles. The ``grids`` on which this field
        is defined will be used to enforce reflecting boundary
        conditions.
    dt
        Time step for Euler integration.
    omega_p
        Plasma frequency in inverse seconds.
    c
        Speed of light in meters per second.
    normalize
        If False, perform calculations in "raw" units. If True,
        normalize equations using the natural units specified
        by ``omega_p`` and ``c``.
    """

    def __init__(
        self,
        particles: Particles,
        electric_field: InterpolatedField,
        dt: float = 1e-2,
        omega_p: float = 1,
        c: float = 1,
        normalize: bool = False,
    ):
        self.particles = particles
        self.E = electric_field
        self.dt = dt
        self.normalize = normalize

        self.omega_p = omega_p
        self.c = c

    def evolve(self, dt: float | None = None) -> None:
        """
        Evolve the particle positions and velocities forward one time step.
        These attributes are modified in-place (that is, in the ``particles``
        attribute of this class).

        Parameters
        ----------
        dt
            If None, use the ``dt`` assigned at class instantiation.
            Otherwise, use the specified value.
        """
        if dt is None:
            dt = self.dt
        dx = self.particles.velocities * dt
        dv = (
            self.particles.charges[:, np.newaxis]
            / self.particles.masses[:, np.newaxis]
            * self.E.evaluate(self.particles.positions)
            * dt
        )
        # Normalization introduces 1/(omega_p * c)
        # But to go from E_n to E, need to multiply by (c/omega_p)**2
        # (tracking normalizations from phi)
        if self.normalize:
            dv *= 1 / (self.omega_p * self.c)
        self.particles.positions = self.particles.positions + dx
        self.particles.velocities = self.particles.velocities + dv
        self.enforce_boundaries()

    def update_field(self, new_electric_field: InterpolatedField) -> None:
        """
        Update the electric field that accelerates the particles.

        Parameters
        ----------
        new_electric_field
            The new field to be used.
        """
        self.E = new_electric_field

    def update_potential(self, new_phi_on_grid: FArray) -> None:
        """
        Update the electric potential used to derive the electric
        field that accelerates the particles. This is essentially
        the same as ``update_field``, but you do not need to
        generate a new ``InterpolatedField`` object on your
        own.

        Parameters
        ----------
        new_phi_on_grid
            The new potential to be used.
        """
        self.E.update_potential(new_phi_on_grid)

    def enforce_boundaries(self) -> None:
        """
        Enforce reflecting boundary conditions for all particles in
        ``particles``. Whenever a particle crosses outside the
        boundaries specified by one of the ``grids`` of ``electric_field``,
        it is instead placed on the appropriate boundary of the grid and
        the component of the velocity associated with that grid is
        reversed. This function will typically not be called directly,
        but it is available in case the need arises.
        """
        mins = [np.min(ar) for ar in self.E.grids]
        maxes = [np.max(ar) for ar in self.E.grids]
        for i, min_val in enumerate(mins):
            inds = (np.argwhere(self.particles.positions[:, i] <= min_val).flatten(), i)
            self.particles.positions[inds] = min_val
            self.particles.velocities[inds] *= -1
        for i, max_val in enumerate(maxes):
            inds = (np.argwhere(self.particles.positions[:, i] >= max_val).flatten(), i)
            self.particles.positions[inds] = max_val
            self.particles.velocities[inds] *= -1

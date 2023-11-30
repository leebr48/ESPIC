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
        accelerate the particles.
    dt
        Time step for Euler integration.
    """

    def __init__(
        self,
        particles: Particles,
        electric_field: InterpolatedField,
        dt: float = 1e-2,
    ):
        self.particles = particles
        self.E = electric_field
        self.dt = dt

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
        self.particles.positions = self.particles.positions + dx
        self.particles.velocities = self.particles.velocities + dv

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

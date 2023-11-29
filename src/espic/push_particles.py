"""Implements ParticlePusher class, which evolves particle positions and velocities forward one time step."""

import numpy as np

from espic.def_particles import Particles
from espic.interp_field import InterpolatedField


class ParticlePusher:
    """
    Evolves particles one timestep at a time.

    Inputs:
        particles(Particles): Instance of the Particles class containing all particles
                              to be moved.
        electric_field(InterpolatedField): Instance of the InterpolatedField class
                                           containing the field that will move the
                                           particles.
        dt(float): Time step for Euler integration.

    Methods
    -------
        evolve: Evolve the particle positions and velocities forward one time step.
    """

    def __init__(
        self, particles: Particles, electric_field: InterpolatedField, dt: float = 1e-2
    ):
        self.particles = particles
        self.E = electric_field
        self.dt = dt

    def evolve(self, dt: float | None = None) -> None:
        """
        Evolve the particle positions and velocities forward one time step.
        These attributes are modified in-place.

        Inputs:
            dt(float or None): If None, use the dt assigned at class instantiation.
                               Otherwise, use the specified float.
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

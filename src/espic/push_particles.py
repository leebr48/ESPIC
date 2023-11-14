"""Implements ParticlePusher class, which evolves particle positions and velocities forward one time step."""

import numpy as np

class ParticlePusher:
    def __init__(self, particles, electric_field, dt=1e-2):
        # particles is a Particles object
        # electric_field is InterpolatedField object
        # dt is float
        self.particles = particles
        self.E = electric_field
        self.dt = dt

    def evolve(self, dt=None):
        if dt is None:
            dt = self.dt
        dx = self.particles.velocities * dt
        dv = self.particles.charges[:, np.newaxis] / self.particles.masses[:, np.newaxis] * self.E.ev(self.particles.positions) * dt
        self.particles.positions = self.particles.positions + dx
        self.particles.velocities = self.particles.velocities + dv

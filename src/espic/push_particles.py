"""Implements ParticlePusher class, which evolves particle positions and velocities forward one time step."""

import numpy as np

class ParticlePusher:
    def __init__(self, particles, electric_field, dt=1e-2):
        # particles is list of Particle objects
        # electric_field is InterpolatedField object
        # dt is float
        self.particles = particles
        self.E = electric_field
        self.dt = dt
        # FIXME maybe you should make everything into numpy arrays so cls.positions will return all positions, etc?
        # Would help with vectorization!

    def evolve(self, dt=None):
        if dt is None: #FIXME test this!
            dt = self.dt
        for particle in self.particles:
            dx = particle.velocity * dt
            dv = particle.charge / particle.mass * self.E.ev(particle.position) * dt # FIXME seems to be changing dimensions of output
            particle.position = particle.position + dx
            particle.velocity = particle.velocity + dv
        # FIXME write tests!

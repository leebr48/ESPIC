# FIXME add comment

import numpy as np

class ParticlePush:
    # FIXME how handle electric field?
    def __init__(self, particles, electric_field, dt=1e-2):
        self.particles = particles
        self.E = electric_field
        self.dt = dt

    def evolve(self, dt=None):
        if dt is None: #FIXME test this!
            dt = self.dt
        for particle in self.particles:
            dx = particle.velocity * dt
            dv = particle.charge / particle.mass * self.E(particle.position) * dt
            particle.position = particle.position + dx
            particle.velocity = particle.velocity + dv
        # FIXME write tests!

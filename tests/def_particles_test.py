"""Test particle class"""

import numpy as np
from espic.def_particles import Particles


def test_particles_1D():
    qs = np.asarray([1, 2])
    ms = np.asarray([5.5, 6.5])
    positions = np.asarray([[1.1], [2.2]])
    velocities = np.asarray([[-2.5], [-3.5]])
    particles = Particles(qs, ms, positions, velocities)

    assert np.array_equal(particles.charges, qs)
    assert np.array_equal(particles.masses, ms)
    assert np.array_equal(particles.positions, positions)
    assert np.array_equal(particles.velocities, velocities)

def test_particle_2D():
    qs = np.asarray([1, 2])
    ms = np.asarray([5.5, 6.5])
    positions = np.asarray([[1.1, -2.2], [-3.3, 4.4]])
    velocities = np.asarray([[-2.5, 3.3], [4.7, -5.5]])
    particles = Particles(qs, ms, positions, velocities)
    
    assert np.array_equal(particles.charges, qs)
    assert np.array_equal(particles.masses, ms)
    assert np.array_equal(particles.positions, positions)
    assert np.array_equal(particles.velocities, velocities)

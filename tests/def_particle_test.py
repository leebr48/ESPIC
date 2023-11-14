"""Test particle class"""

import numpy as np
from espic.def_particle import Particle


def test_particle_1D():
    q = 1
    m = 5.5
    pos = np.asarray([1.1])
    vel = np.asarray([-2.5])
    particle = Particle(q, m, pos, vel)

    assert particle.charge == q
    assert particle.mass == m
    assert particle.position == pos
    assert particle.velocity == vel

def test_particle_2D():
    q = -1
    m = 7.2
    pos = np.asarray([1.1, -2.2])
    vel = np.asarray([-2.5, 3.3])
    particle = Particle(q, m, pos, vel)

    assert particle.charge == q
    assert particle.mass == m
    assert np.array_equal(particle.position, pos)
    assert np.array_equal(particle.velocity, vel)

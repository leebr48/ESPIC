"""Test particle pusher class"""

import numpy as np
from espic.def_particle import Particle
from espic.interp_field import InterpolatedField
from espic.push_particles import ParticlePusher


def test_push_1D_null_field():
    q = 2
    m = 5.5
    pos = np.asarray([0.5])
    vel = np.asarray([-2])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = np.ones(x_vec.size)
    dt = 1e-2
    particles = [Particle(q, m, pos, vel)]
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()
    
    assert np.allclose(particle_pusher.particles[0].position, pos + vel * dt)

def test_push_1D_linear_potential():
    q = 2
    m = 5
    pos = np.asarray([0.5])
    vel = np.asarray([2])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = 3 * x_vec
    dt = 1e-2
    particles = [Particle(q, m, pos, vel)]
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()
    
    assert np.allclose(particle_pusher.particles[0].position, pos + vel * dt)

    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles[0].position, 0.53988)
# FIXME should you create a class that can make a swarm of particles??
def test_push_1D_linear_potential_multiparticle():
    q1 = 2
    q2 = -1
    m1 = 5
    m2 = 2
    pos1 = np.asarray([0.5])
    pos2 = np.asarray([0.25])
    vel1 = np.asarray([2])
    vel2 = np.asarray([-1.5])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = 3 * x_vec
    dt = 1e-2
    particles = [Particle(q1, m1, pos1, vel1), Particle(q2, m2, pos2, vel2)]
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()
    
    assert np.allclose(particle_pusher.particles[0].position, pos1 + vel1 * dt)
    assert np.allclose(particle_pusher.particles[1].position, pos2 + vel2 * dt)

    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles[0].position, 0.53988)
    assert np.allclose(particle_pusher.particles[1].position, 0.22015)

def test_push_2D_parabolic_potential_multiparticle():
    q1 = 2
    q2 = -1
    m1 = 5
    m2 = 2
    pos1 = np.asarray([0.5, -0.25])
    pos2 = np.asarray([0.25, 1])
    vel1 = np.asarray([2, -1])
    vel2 = np.asarray([-1.5, 0.25])
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz = xx**2 + yy**2
    dt = 1e-2
    particles = [Particle(q1, m1, pos1, vel1), Particle(q2, m2, pos2, vel2)]
    interpolated_field = InterpolatedField([x_vec, y_vec], zz)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()
    
    assert np.allclose(particle_pusher.particles[0].position, pos1 + vel1 * dt)
    assert np.allclose(particle_pusher.particles[1].position, pos2 + vel2 * dt)

    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles[0].position, np.asarray([0.5399584, -0.2699792]), atol=2e-6, rtol=0)
    assert np.allclose(particle_pusher.particles[1].position, np.asarray([0.2200235, 1.00510025]), atol=2e-6, rtol=0)

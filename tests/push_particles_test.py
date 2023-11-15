"""Test particle pusher class"""

import numpy as np

from espic.def_particles import Particles
from espic.interp_field import InterpolatedField
from espic.push_particles import ParticlePusher


def test_push_1D_null_field():
    q = np.asarray([2])
    m = np.asarray([5.5])
    pos = np.asarray([0.5])
    vel = np.asarray([-2])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = np.ones(x_vec.size)
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt)


def test_push_1D_linear_potential():
    q = np.asarray([2])
    m = np.asarray([5])
    pos = np.asarray([0.5])
    vel = np.asarray([2])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = 3 * x_vec
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt)

    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, 0.53988)


def test_push_1D_linear_potential_multiparticle():
    qs = np.asarray([2, -1])
    ms = np.asarray([5, 2])
    positions = np.asarray([[0.5], [0.25]])
    velocities = np.asarray([[2], [-1.5]])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = 3 * x_vec
    dt = 1e-2
    particles = Particles(qs, ms, positions, velocities)
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, positions + velocities * dt)

    particle_pusher.evolve()

    assert np.allclose(
        particle_pusher.particles.positions,
        np.asarray([0.53988, 0.22015])[:, np.newaxis],
    )


def test_push_2D_parabolic_potential_multiparticle():
    qs = np.asarray([2, -1])
    ms = np.asarray([5, 2])
    positions = np.asarray([[0.5, -0.25], [0.25, 1]])
    velocities = np.asarray([[2, -1], [-1.5, 0.25]])
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz = xx**2 + yy**2
    dt = 1e-2
    particles = Particles(qs, ms, positions, velocities)
    interpolated_field = InterpolatedField([x_vec, y_vec], zz)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, positions + velocities * dt)

    particle_pusher.evolve()

    target_positions = np.asarray([[0.5399584, -0.2699792], [0.2200235, 1.00510025]])

    assert np.allclose(
        particle_pusher.particles.positions, target_positions, atol=2e-6, rtol=0
    )


def test_dt_change():
    q = np.asarray([2])
    m = np.asarray([5.5])
    pos = np.asarray([0.5])
    vel = np.asarray([-2])
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = np.ones(x_vec.size)
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    dt_new = 1e-1
    particle_pusher.evolve(dt=dt_new)

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt_new)

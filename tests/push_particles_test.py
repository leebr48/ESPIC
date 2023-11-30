"""Test particle pusher class"""

import numpy as np

from espic.def_particles import Particles
from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid
from espic.push_particles import ParticlePusher


def test_push_1D_null_field():
    # Simplest possible test - one particle moves in straight lines
    q = np.asarray([2])
    m = np.asarray([5.5])
    pos = np.asarray([0.5])
    vel = np.asarray([-2])
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = np.ones(x_grid.size)
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt)


def test_push_1D_linear_potential():
    # Particle moves in 1D under influence of a constant electric field
    q = np.asarray([2])
    m = np.asarray([5])
    pos = np.asarray([0.5])
    vel = np.asarray([2])
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = 3 * x_grid.grid
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt)

    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, 0.53988)


def test_push_1D_linear_potential_multiparticle():
    # Particles move in 1D under influence of a constant electric field
    qs = np.asarray([2, -1])
    ms = np.asarray([5, 2])
    positions = np.asarray([[0.5], [0.25]])
    velocities = np.asarray([[2], [-1.5]])
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = 3 * x_grid.grid
    dt = 1e-2
    particles = Particles(qs, ms, positions, velocities)
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, positions + velocities * dt)

    particle_pusher.evolve()

    assert np.allclose(
        particle_pusher.particles.positions,
        np.asarray([0.53988, 0.22015])[:, np.newaxis],
    )


def test_push_2D_parabolic_potential_multiparticle():
    # Particles move in 2D under influence of parabolic potential - check numerical accuracy
    qs = np.asarray([2, -1])
    ms = np.asarray([5, 2])
    positions = np.asarray([[0.5, -0.25], [0.25, 1]])
    velocities = np.asarray([[2, -1], [-1.5, 0.25]])
    x_grid = Uniform1DGrid(num_points=1000, x_min=-2, x_max=2)
    y_grid = x_grid
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz = xx**2 + yy**2
    dt = 1e-2
    particles = Particles(qs, ms, positions, velocities)
    interpolated_field = InterpolatedField([x_grid, y_grid], zz)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    particle_pusher.evolve()

    assert np.allclose(particle_pusher.particles.positions, positions + velocities * dt)

    particle_pusher.evolve()

    target_positions = np.asarray([[0.5399584, -0.2699792], [0.2200235, 1.00510025]])

    assert np.allclose(
        particle_pusher.particles.positions,
        target_positions,
        atol=2e-6,
        rtol=0,
    )


def test_dt_change():
    # Check that time step override is working properly
    q = np.asarray([2])
    m = np.asarray([5.5])
    pos = np.asarray([0.5])
    vel = np.asarray([-2])
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = np.ones(x_grid.size)
    dt = 1e-2
    particles = Particles(q, m, pos, vel)
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    particle_pusher = ParticlePusher(particles, interpolated_field, dt=dt)
    dt_new = 1e-1
    particle_pusher.evolve(dt=dt_new)

    assert np.allclose(particle_pusher.particles.positions, pos + vel * dt_new)

# Initial stab at 1D PIC code
# FIXME formalize all this once it's ready, then put 1D and 2D
# examples in an 'examples' directory in root

import numpy as np

from espic.def_particles import Particles
from espic.deposit_charge import ChargeDeposition
from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid
from espic.push_particles import ParticlePusher
from espic.solve_maxwell import MaxwellSolver1D

num_particles = 100
charges = np.ones(num_particles)
masses = np.ones(num_particles)
init_vel = np.random.normal(size=num_particles)
init_pos = np.random.uniform(low=0.25, high=0.75, size=num_particles)

particles = Particles(charges, masses, init_pos, init_vel)
grid = Uniform1DGrid(x_min=0, x_max=1)


dt = 0.01
t = 0
tm = 0.01

# Initialization
cd = ChargeDeposition(grid=grid)
rho = cd.deposit(particles.charges, particles.positions)
ms = MaxwellSolver1D(grid=grid)
phi = ms.solve(rho)
efield = InterpolatedField(grids=[grid], phi_on_grid=phi)

while t < tm:
    particle_pusher = ParticlePusher(particles, efield, dt)
    particle_pusher.evolve()

    rho = cd.deposit(
        particle_pusher.particles.charges,
        particle_pusher.particles.positions,
    )
    phi = ms.solve(rho)
    efield = InterpolatedField(grids=[grid], phi_on_grid=phi)

    t += dt

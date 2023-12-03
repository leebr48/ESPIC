# Initial stab at 1D PIC code

import numpy as np

# %% Initialize Physical Parameters
import scipy.constants as sc

from espic.def_particles import Particles
from espic.deposit_charge import ChargeDeposition
from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid
from espic.push_particles import ParticlePusher
from espic.solve_maxwell import MaxwellSolver1D

num_particles = 100
ne = 100
q = sc.e
m = sc.m_e
omega_p = np.sqrt(ne * q**2 / (m * sc.epsilon_0))
prefac = 1 / (omega_p * sc.c) * q / m

a = 0.01
vth = a * sc.c
vthp = a

# %% Initialize distribution of particles

# init_speed = maxwell.rvs(size=num_particles, scale=vthp)
init_speed = np.zeros(num_particles)
init_vel = np.zeros(len(init_speed))
for i in range(len(init_speed)):
    if i % 2 == 0:
        init_vel[i] = -init_speed[i]
    else:
        init_vel[i] = init_speed[i]

masses = m * np.ones(num_particles)
x_min = -10
x_max = 10
dt = 1 / (omega_p)
t = 0
t_max = 400 / omega_p
init_amp = 0.2

init_pos = np.random.uniform(low=x_min / 2, high=x_max / 2, size=num_particles)
charges = q * init_amp * np.sin(init_pos / 2)

init_vel = init_vel.reshape((len(init_vel), 1))
init_pos = init_pos.reshape((len(init_pos), 1))

particles = Particles(charges, masses, init_pos, init_vel)
grid = Uniform1DGrid(x_min=x_min, x_max=x_max, num_points=1000)
boundary_conditions = 0 * np.ones(2)

# %% Run simulation
cd = ChargeDeposition(grid=grid)
rho = cd.deposit(particles.charges, particles.positions)
init_rho = cd.deposit(particles.charges, particles.positions)
ms = MaxwellSolver1D(
    boundary_conditions=boundary_conditions,
    grid=grid,
    omega_p=omega_p,
    c=sc.c,
    normalize=True,
)
phi = ms.solve(rho)
init_phi = ms.solve(rho)
efield = InterpolatedField(
    grids=[grid],
    phi_on_grid=phi,
    omega_p=omega_p,
    c=sc.c,
    normalize=True,
)
particle_pusher = ParticlePusher(particles, efield, dt, omega_p, sc.c, normalize=True)

phi_v_time = ()
rho_v_time = ()

while t < t_max:
    particle_pusher.evolve()

    rho = cd.deposit(
        particle_pusher.particles.charges,
        particle_pusher.particles.positions,
    )
    phi = ms.solve(rho)

    phi_v_time += (phi,)
    rho_v_time += (rho,)

    particle_pusher.update_potential(phi)

    t += dt

# %% Compute fft
phi_v_time_arr = np.array(phi_v_time)
phi_fft = np.abs(np.fft.fft2(phi_v_time_arr))


# %% Make animation

import matplotlib.pyplot as plt
from matplotlib import animation

writer = animation.FFMpegWriter(fps=5)
# writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

phi_v_time_arr = np.array(phi_v_time)
fig = plt.figure()


def animate(i):
    plt.plot(grid.grid, phi_v_time[i])


ani = animation.FuncAnimation(fig, animate, frames=len(phi_v_time_arr), repeat=True)

writervideo = animation.FFMpegWriter(fps=5)
ani.save("phi_v_time.mp4", writer=writervideo)
plt.show()


# %%

# charges = 1e-4* np.ones(num_particles)
masses = np.ones(num_particles)
init_vel = np.random.normal(size=num_particles)
# init_vel = np.zeros(num_particles)
init_pos = np.random.uniform(low=-1, high=1, size=num_particles)
charges = 1e-4 * np.sin(10 * init_pos)

init_vel = init_vel.reshape((len(init_vel), 1))
init_pos = init_pos.reshape((len(init_pos), 1))

particles = Particles(charges, masses, init_pos, init_vel)
grid = Uniform1DGrid(x_min=-10, x_max=10, num_points=1000)
boundary_conditions = 0 * np.ones(2)

dt = 0.01
t = 0
tm = 1

# Initialization
cd = ChargeDeposition(grid=grid)
rho = cd.deposit(particles.charges, particles.positions)
init_rho = cd.deposit(particles.charges, particles.positions)
ms = MaxwellSolver1D(boundary_conditions=boundary_conditions, grid=grid)
phi = ms.solve(rho)
init_phi = ms.solve(rho)
efield = InterpolatedField(grids=[grid], phi_on_grid=phi)

rho_t = ()
vel_t = ()
phi_t = ()
efield_t = []
while t < tm:
    particle_pusher = ParticlePusher(particles, efield, dt)
    particle_pusher.evolve()

    rho = cd.deposit(
        particle_pusher.particles.charges,
        particle_pusher.particles.positions,
    )
    phi = ms.solve(rho)
    efield = InterpolatedField(grids=[grid], phi_on_grid=phi)

    rho_t += (rho,)
    phi_t += (phi,)
    efield_t.append(
        efield.evaluate(grid.grid) * np.hanning(tm / dt).reshape((int(tm / dt), 1)),
    )
    t += dt

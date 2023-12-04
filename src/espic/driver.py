# Initial stab at 1D PIC code

import matplotlib.pyplot as plt
import numpy as np

# %% Initialize Physical Parameters
import scipy.constants as sc

from espic.def_particles import Particles
from espic.deposit_charge import ChargeDeposition
from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid
from espic.push_particles import ParticlePusher
from espic.solve_maxwell import MaxwellSolver1D

num_particles = 500
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
dt = 1 / (100 * omega_p)
t = 0
n_max = 20
t_max = n_max / omega_p
init_amp = 0.2
grid = Uniform1DGrid(x_min=x_min, x_max=x_max, num_points=1000)

# %%

k = 2 / (2 * x_max)
p_temp = np.abs(np.sin(2 * np.pi * grid.grid * k))
pm = p_temp / np.sum(p_temp)
init_pos = np.random.choice(grid.grid, size=num_particles, p=pm)
init_pos = np.sort(init_pos)

signs = np.sin(2 * np.pi * init_pos * k) / np.abs(np.sin(2 * np.pi * init_pos * k))
charges = -q * np.ones(len(init_pos))
charges *= signs


init_vel = init_vel.reshape((len(init_vel), 1))
init_pos = init_pos.reshape((len(init_pos), 1))

particles = Particles(charges, masses, init_pos, init_vel)
boundary_conditions = 0 * np.ones(2)

cd = ChargeDeposition(grid=grid)
rho = cd.deposit(particles.charges, particles.positions)
init_rho = cd.deposit(particles.charges, particles.positions)

# %% Run simulation

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
integrated_phi = np.zeros(int(t_max / dt) + 2)
count = 0


def integrate_phi(x, phi):
    return np.trapz(phi, x)


while t < t_max:
    particle_pusher.evolve()

    rho = cd.deposit(
        particle_pusher.particles.charges,
        particle_pusher.particles.positions,
    )
    phi = ms.solve(rho)

    phi_v_time += (phi,)
    integrated_phi[count] = integrate_phi(grid.grid, phi)
    count += 1

    particle_pusher.update_potential(phi)

    t += dt

plt.plot(np.arange(len(integrated_phi)) * dt * omega_p, integrated_phi)


# %% Diagnostics

phi_v_time_arr = np.array(phi_v_time)
phi_v_time_fft = np.abs(np.fft.fft2(phi_v_time_arr))
freq = np.fft.fftfreq(phi_v_time_fft.shape[0], dt)
k_arr = np.fft.fftfreq(phi_v_time_fft.shape[1], grid.delta)

# Plot space fft of initial phi to verify we have the right k
plt.figure(1)
specific_k = k_arr[1]
plt.plot(k_arr, phi_v_time_fft[0, :])
plt.xlim([0, 0.5])
plt.axvline(k, color="r")

# Plot time fft at right k
plt.figure(2)
plt.plot(freq, phi_v_time_fft[:, 2])
plt.xlim([0, 500])
plt.axvline(omega_p / (2 * np.pi), color="r")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")


# %% Compute fft

# phi_fft = np.abs(np.fft.fft2(phi_v_time_arr))

phi_fft = np.abs(np.fft.fft(integrated_phi))
freq = np.fft.fftfreq(len(phi_fft), dt)
k = np.fft.fftfreq(len(grid.grid), grid.delta)

plt.plot(freq, phi_fft)
plt.axvline(omega_p / (2 * np.pi), color="r")
plt.xlim([0, 500])


phi_v_time_arr = np.array(phi_v_time)
phi_v_time_fft = np.abs(np.fft.fft2(phi_v_time_arr))

phi_single = phi_v_time_arr[:, 250]
phi_single_fft = np.abs(np.fft.fft(phi_single))
plt.plot(freq[:-1], phi_single_fft)
plt.axvline(omega_p / (2 * np.pi), color="r")
plt.xlim([0, 500])

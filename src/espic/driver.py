# Initial stab at 1D PIC code
# FIXME formalize all this once it's ready, then put 1D and 2D
# examples in an 'examples' directory in root

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

FArray = NDArray[np.float64]

from espic.init_charges import Initialize
from espic.make_grid import Uniform1DGrid
from espic.run_espic import RunESPIC

# %% Initialize RunESPIC
# Some parameters have defaults, but for the tutorial we'll specify all of them here

num_particles: int = 500
num_grid_points: int = 1000
dim: int = 1

# Cold Plasma


# Plasma Parameters
ne: float = 100
q: float = sc.e
m: float = sc.m_e
omega_p: float = np.sqrt(ne * q**2 / (m * sc.epsilon_0))
physical_parameters = {"q": q, "m": m, "c": sc.c, "ne": ne, "vth": 0}

# Grid and Temporal Parameters
x_min = -10
x_max = 10
dt = 1 / (100 * omega_p)
n_max = 20
t_max = n_max / omega_p
# t_max = 2*dt
grid = Uniform1DGrid(x_min=x_min, x_max=x_max, num_points=num_grid_points)


# For plasma waves, we want a wave-like perturbation in physical space
# Also assuming cold plasma
k = 2 / (2 * x_max)
init_pos = Initialize(num_particles, dim).sinusoidal(k, grid.grid)
init_vel: FArray = np.zeros(num_particles)
signs = (
    np.sin(2 * np.pi * init_pos * k) / np.abs(np.sin(2 * np.pi * init_pos * k))
).reshape(num_particles)
init_vel = init_vel.reshape((len(init_vel), 1))
init_pos = init_pos.reshape((len(init_pos), 1))

# Boundary Conditions
boundaries = {"left": -10, "right": 10}
boundary_conditions = np.zeros(2)

run_espic = RunESPIC(
    init_pos,
    init_vel,
    boundary_conditions,
    boundaries,
    physical_parameters,
    signs,
    num_particles,
    num_grid_points,
    dim,
    dt,
    t_max,
    k,
    normalize=True,
)

run_espic.run()

# %% Computes FFTs in space and time in order to study the frequency spectrum

phi_v_time_arr = np.array(run_espic.phi_v_time)
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

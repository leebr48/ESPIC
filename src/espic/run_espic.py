import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

from espic.def_particles import Particles
from espic.deposit_charge import ChargeDeposition
from espic.init_charges import Initialize
from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid, Uniform2DGrid
from espic.push_particles import ParticlePusher
from espic.solve_maxwell import MaxwellSolver1D, MaxwellSolver2D

FArray = NDArray[np.float64]


class RunESPIC:
    def __init__(
        self,
        init_pos: FArray,
        init_vel: FArray,
        boundary_conditions: FArray | dict[str, FArray],
        boundaries: dict[str, float],
        physical_parameters: dict[str, float],
        signs: FArray,
        num_particles: int = 100,
        num_grid: int = 1000,
        dim: int = 1,
        dt: float = 0.1,
        t_max: float = 10,
        k: float = 1,
        normalize: bool = False,
    ) -> None:
        self.num_particles = num_particles
        self.num_grid = num_grid
        self.dim = dim
        self.normalize = normalize
        self.dt = dt
        self.t_max = t_max
        self.k = k

        init_state = Initialize(self.num_particles, self.dim)
        if boundaries is None:
            if dim == 1:
                self.boundaries = {"left": -1, "right": 1}
            else:
                self.boundaries = {
                    "bottom": -1,
                    "top": 1,
                    "left": -1,
                    "right": 1,
                }
        else:
            self.boundaries = boundaries

        if boundary_conditions is None:
            if dim == 1:
                self.boundary_conditions = np.zeros(2)
            else:
                self.boundary_conditions = {
                    "bottom": np.zeros(self.num_grid),
                    "top": np.zeros(self.num_grid),
                    "left": np.zeros(self.num_grid),
                    "right": np.zeros(self, num_grid),
                }
        else:
            self.boundary_conditions = boundary_conditions

        if dim == 1:
            self.grid = Uniform1DGrid(
                self.num_grid, self.boundaries["left"], self.boundaries["right"]
            )
        if dim == 2:
            self.grid = Uniform2DGrid(
                self.num_grid,
                self.boundaries["left"],
                self.bundaries["right"],
                self.boundaries["bottom"],
                self.boundaries["top"],
            )

        if init_pos is None:
            # self.init_pos = init_state.uniform(boundaries["left"], boundaries["right"])
            self.init_pos = init_state.sinusoidal(self.k, self.grid.grid)
        else:
            self.init_pos = init_pos
        if init_vel is None:
            self.init_vel = init_state.uniform(boundaries["left"], boundaries["right"])
        else:
            self.init_vel = init_vel

        if physical_parameters is None:
            self.physical_parameters = {
                "q": sc.e,
                "m": sc.m_e,
                "c": sc.c,
                "ne": 100,
                "vth": 0.01 * sc.c,
            }
        else:
            self.physical_parameters = physical_parameters

        if self.normalize:
            self.physical_parameters["vth"] /= sc.c
        self.omega_p = self.compute_plasma_frequency()

        if signs is None:
            self.signs = np.ones(self.num_particles)
        else:
            self.signs = signs

        self.masses = self.physical_parameters["m"] * np.ones(self.num_particles)
        self.charges = (
            self.physical_parameters["q"] * np.ones(self.num_particles) * self.signs
        )

        # self.initialize_solvers()

    def initialize_solvers(
        self,
    ) -> None:
        particles = Particles(self.charges, self.masses, self.init_pos, self.init_vel)

        charge_deposition = ChargeDeposition(grid=self.grid)
        rho = charge_deposition.deposit(particles.charges, particles.positions)

        if self.dim == 1:
            maxwell_solver = MaxwellSolver1D(
                boundary_conditions=self.boundary_conditions,
                grid=self.grid,
                omega_p=self.omega_p,
                c=self.physical_parameters["c"],
                normalize=self.normalize,
            )
        else:
            maxwell_solver = MaxwellSolver2D(
                boundary_conditions=self.boundary_conditions,
                grid=self.grid,
                omega_p=self.omega_p,
                c=self.physical_parameters["c"],
                normalize=self.normalize,
            )
        phi = maxwell_solver.solve(rho)
        efield = InterpolatedField(
            grids=[self.grid],
            phi_on_grid=phi,
            omega_p=self.omega_p,
            c=self.physical_parameters["c"],
            normalize=self.normalize,
        )
        particle_pusher = ParticlePusher(
            particles,
            efield,
            self.dt,
            self.omega_p,
            self.physical_parameters["c"],
            normalize=self.normalize,
        )

        return (charge_deposition, maxwell_solver, particle_pusher)

    def run(self) -> None:
        t = 0
        self.rho_v_time = ()
        self.phi_v_time = ()
        self.integrated_phi = ()

        self.cd, self.ms, self.pp = self.initialize_solvers()

        while t < self.t_max:
            self.pp.evolve()

            rho = self.cd.deposit(
                self.pp.particles.charges,
                self.pp.particles.positions,
            )

            phi = self.ms.solve(rho)

            self.phi_v_time += (phi,)
            self.rho_v_time += (rho,)
            self.integrated_phi += (self.integrate_phi(phi),)

            self.pp.update_potential(phi)

            t += self.dt

    def compute_plasma_frequency(self) -> float:
        ne = self.physical_parameters["ne"]
        q = self.physical_parameters["q"]
        m = self.physical_parameters["m"]

        return np.sqrt(ne * q**2 / (m * sc.epsilon_0))

    def integrate_phi(self, phi) -> float:
        return np.trapz(self.grid.grid, phi)
"""Defines the driver class for ESPIC used to run simulations."""

from __future__ import annotations

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
    """
    Driver class for ESPIC.

    Parameters
    ----------
    init_state
        Dictionary containing the initial state of the particles. Keys are
        ``initial position``, and ``initial velocity``.
    boundary_conditions
        Array or tuple containing the boundary conditions.
    boundaries
        Dictionary containing the limits of the simulation domain. Keys are
        ``top``, ``bottom``, ``left``, and ``right``.
    physical_parameters
        Dictionary containing important physical parameters: the particle charge,
        the particle mass, the speed of light, the background charge density,
        and the plasma frequency.
    perturbation:
        Dictionary containing information about the initial perturbation
        to the system.
        Keys are ``signs`` and ``k`` (the wavenumber)
    time_param:
        Dictionary containing information about the time parameters.
        Keys are ``dt`` (the timestep) and ``t_max`` (the maximum sim time)
    normalize
        If ``False``, perform calculations in "raw" units. If ``True``,
        normalize equations using the natural units specified
        by ``omega_p`` and ``c``.
    """

    def __init__(  # noqa: PLR0913
        self,
        init_state: dict[str, FArray],
        boundary_conditions: FArray | dict[str, FArray],
        boundaries: dict[str, float],
        physical_parameters: dict[str, float],
        perturbation: dict[str, FArray | float],
        resolution: dict[str, int] = {  # noqa: B006
            "number of particles": 100,
            "number of grid points": 1000,
            "dimension": 1,
        },
        time_param: dict[str, float] = {"dt": 0.1, "t_max": 1.0},  # noqa: B006
        normalize: bool = False,
    ) -> None:
        self.num_particles: int = resolution["number of particles"]
        self.num_grid: int = resolution["number of grid points"]
        self.dim: int = resolution["dimension"]
        self.normalize: bool = normalize
        self.dt: float = time_param["dt"]
        self.t_max: float = time_param["t_max"]

        self.init_state = init_state
        self.boundary_conditions = boundary_conditions
        self.boundaries = boundaries
        self.physical_parameters = physical_parameters
        self.perturbation = perturbation

        self.initialize_boundaries()
        self.initialize_grid()
        self.initialize_perturbation()
        self.initialize_state()
        self.initialize_parameters()

        self.masses = self.physical_parameters["m"] * np.ones(self.num_particles)
        self.charges = (
            self.physical_parameters["q"]
            * np.ones(self.num_particles)
            * self.perturbation["signs"]
        )

    def initialize_perturbation(self) -> None:
        """
        Initialize parameters for the initial (sinusoidal) perturbation
        to the system.
        """
        if self.perturbation is None:
            self.perturbation = {"signs": np.ones(self.num_particles), "k": 0.0}

    def initialize_parameters(self) -> None:
        """Initialize the physical parameters of this system."""
        if self.physical_parameters is None:
            self.physical_parameters = {
                "q": sc.e,
                "m": sc.m_e,
                "c": sc.c,
                "ne": 100,
                "vth": 0.01 * sc.c,
            }

        if self.normalize:
            self.physical_parameters["vth"] /= sc.c
        self.omega_p = self.compute_plasma_frequency()

    def initialize_state(
        self,
    ) -> None:
        """
        Initialize the initial state of the particles, i.e. their positions
        and velocities.

        """
        initialize = Initialize(self.num_particles, self.dim)
        if self.init_state is None:
            self.init_state = {
                "initial position": initialize.sinusoidal(
                    self.perturbation["k"],
                    self.grid.grid,
                ),
                "initial velocity": initialize.uniform(
                    self.boundaries["left"],
                    self.boundaries["right"],
                ),
            }

    def initialize_grid(self) -> None:
        """Initialize the computational grid."""
        self.grid: Uniform1DGrid | Uniform2DGrid
        if self.dim == 1:
            self.grid = Uniform1DGrid(
                self.num_grid,
                self.boundaries["left"],
                self.boundaries["right"],
            )
        else:
            self.grid = Uniform2DGrid(
                self.num_grid,
                self.boundaries["left"],
                self.boundaries["right"],
                self.boundaries["bottom"],
                self.boundaries["top"],
            )

    def initialize_boundaries(
        self,
    ) -> None:
        """Initialize the boundaries and boundary conditions of the simulation."""
        if self.boundaries is None:
            if self.dim == 1:
                self.boundaries = {"left": -1, "right": 1}
            else:
                self.boundaries = {
                    "bottom": -1,
                    "top": 1,
                    "left": -1,
                    "right": 1,
                }

        if self.boundary_conditions is None:
            if self.dim == 1:
                self.boundary_conditions = np.zeros(2)
            else:
                self.boundary_conditions = {
                    "bottom": np.zeros(self.num_grid),
                    "top": np.zeros(self.num_grid),
                    "left": np.zeros(self.num_grid),
                    "right": np.zeros(self.num_grid),
                }

    def initialize_solvers(
        self,
    ) -> tuple[ChargeDeposition, MaxwellSolver1D | MaxwellSolver2D, ParticlePusher]:
        """
        Initialize the ``Particles``, ``ChargeDeposition``, ``MaxwellSolver1D``
        (or 2D), and ``PariclePusher`` objects used to self-consistently evolve
        the particle states.
        """
        particles = Particles(
            self.charges,
            self.masses,
            self.init_state["initial position"],
            self.init_state["initial velocity"],
        )

        charge_deposition = ChargeDeposition(grid=self.grid)
        rho = charge_deposition.deposit(particles.charges, particles.positions)
        maxwell_solver: MaxwellSolver1D | MaxwellSolver2D

        if self.dim == 1:
            maxwell_solver = MaxwellSolver1D(
                boundary_conditions=self.boundary_conditions,  # type: ignore[arg-type]
                grid=self.grid,  # type: ignore[arg-type]
                omega_p=self.omega_p,
                c=self.physical_parameters["c"],
                normalize=self.normalize,
            )
        else:
            maxwell_solver = MaxwellSolver2D(
                boundary_conditions=self.boundary_conditions,  # type: ignore[arg-type]
                grid=self.grid,  # type: ignore[arg-type]
                omega_p=self.omega_p,
                c=self.physical_parameters["c"],
                normalize=self.normalize,
            )
        phi = maxwell_solver.solve(rho)

        efield = InterpolatedField(
            grids=[self.grid],  # type: ignore[list-item]
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
        """
        Run the simuation. Steps are
        (1) evolve particle positions and velocities using current electric field,
        (2) compute new charge density, and
        (3) compute new electrostatic potential.
        """
        t = 0.0
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

            self.phi_v_time += (phi,)  # type: ignore[assignment]
            self.rho_v_time += (rho,)  # type: ignore[assignment]
            self.integrated_phi += (self.integrate_phi(phi),)  # type: ignore[assignment]

            self.pp.update_potential(phi)

            t += self.dt

        self.phi_v_time = np.array(self.phi_v_time)  # type: ignore[assignment]
        self.rho_v_time = np.array(self.rho_v_time)  # type: ignore[assignment]

    def compute_plasma_frequency(self) -> float:
        """
        Compute plasma frequency :math:`= \\sqrt{\\frac{n  e^{2}}{m \\epsilon_{0}}}`.

        Returns
        -------
            The plasma frequency.
        """
        ne = self.physical_parameters["ne"]
        q = self.physical_parameters["q"]
        m = self.physical_parameters["m"]

        return float(np.sqrt(ne * q**2 / (m * sc.epsilon_0)))

    def integrate_phi(self, phi: FArray) -> float:
        """
        Integrate the electrostatic potential (currently in 1D).
        Uses a trapezoidal scheme.

        Parameters
        ----------
        phi
            The electrostatic potential.

        Returns
        -------
            The integral of ``phi`` along x.

        """
        return float(np.trapz(self.grid.grid, phi))

    def compute_fft(self, quantity: FArray) -> tuple[FArray, FArray, FArray]:
        """
        Compute the FFT of a given quantity over space and time.

        Parameters
        ----------
        quantity : FArray
            Quantity to be FFT'd.

        Returns
        -------
        FArray
            The FFT of the input quantity.

        """
        fft = np.abs(np.fft.fft2(quantity))
        freq = np.fft.fftfreq(fft.shape[0], self.dt)
        k_arr = np.fft.fftfreq(fft.shape[1], self.grid.delta)

        return freq, k_arr, fft

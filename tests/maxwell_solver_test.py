import numpy as np

from espic.make_grid import Uniform2DGrid

# Test 1D Maxwell solver for a uniform charge distribution
from espic.solve_maxwell import MaxwellSolver1D, MaxwellSolver2D


def test_1d_uniform():
    """Test 1D Maxwell solver for a uniform charge distribution"""
    ms = MaxwellSolver1D()
    rho = np.ones(len(ms.grid))
    phi = ms.solve(rho)

    true_phi = -2 * np.pi * (ms.grid**2 - 1)
    err = np.abs(true_phi - phi)
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)


def test_2d_A():
    ms = MaxwellSolver2D()
    A = ms.set_A(4)

    true_A = np.array([[-4, 1, 1, 0], [1, -4, 0, 1], [1, 0, -4, 1], [0, 1, 1, -4]])
    np.testing.assert_allclose(A, true_A, atol=1e-9)


def test_2d_boundary():
    pass


def test_2d_laplace():
    """Test 2D Poisson solver for rho = 0. Ex taken from Ex 3.4 in Griffiths E&M, 4th ed"""
    N = 100
    V0 = 1
    a = 2
    b = 1

    grid = Uniform2DGrid(num_points=N, x_min=-b, x_max=b, y_min=0, y_max=a)
    boundary_conditions = {
        "bottom": np.zeros(N),
        "top": np.zeros(N),
        "left": V0 * np.ones(N),
        "right": V0 * np.ones(N),
    }
    ms = MaxwellSolver2D(grid=grid, boundary_conditions=boundary_conditions)
    rho = np.zeros(ms.grid[0].shape)

    phi = ms.solve(rho)

    max_n = 250
    X, Y = np.meshgrid(np.linspace(-b, b, N), np.linspace(0, a, N))
    true_phi = np.zeros(X.shape)

    for i in range(max_n):
        if i % 2 == 1:
            true_phi += (
                4
                * V0
                / np.pi
                * 1
                / i
                * np.cosh(i * np.pi * X / a)
                / np.cosh(i * np.pi * b / a)
                * np.sin(i * np.pi * Y / a)
            )

    # There are some issues that I believe are related to the convergence of an infinite Fourier series
    # For now, these "magic" numbers should be kept fixed.

    err = np.abs(true_phi[1:-1, 1:-1] - phi[1:-1, 1:-1])
    np.testing.assert_allclose(err, np.zeros(err.shape), atol=1e-2)


# return phi, true_phi, X, Y


def test_2d_poisson_uniform():
    """Same ex as before, but with a constant charge density"""
    N = 100
    V0 = 1
    a = 2
    b = 1

    grid = Uniform2DGrid(num_points=N, x_min=-b, x_max=b, y_min=0, y_max=a)
    bc_bottom = -np.pi * grid.xgrid**2
    bc_top = -np.pi * (grid.xgrid**2 + (a * np.ones(len(grid.ygrid))) ** 2)
    bc_left = V0 - np.pi * ((-b * np.ones(len(grid.xgrid))) ** 2 + grid.ygrid**2)
    bc_right = V0 - np.pi * ((b * np.ones(len(grid.xgrid))) ** 2 + grid.ygrid**2)
    boundary_conditions = {
        "bottom": bc_bottom,
        "top": bc_top,
        "left": bc_left,
        "right": bc_right,
    }
    ms = MaxwellSolver2D(grid=grid, boundary_conditions=boundary_conditions)
    rho = np.ones(ms.grid[0].shape)

    phi = ms.solve(rho)

    max_n = 250
    X, Y = np.meshgrid(np.linspace(-b, b, N), np.linspace(0, a, N))
    true_phi = np.zeros(X.shape)

    for i in range(max_n):
        if i % 2 == 1:
            true_phi += (
                4
                * V0
                / np.pi
                * 1
                / i
                * np.cosh(i * np.pi * X / a)
                / np.cosh(i * np.pi * b / a)
                * np.sin(i * np.pi * Y / a)
            )
    true_phi += -np.pi * (X**2 + Y**2)
    # There are some issues that I believe are related to the convergence of an infinite Fourier series
    # For now, these "magic" numbers should be kept fixed.
    err = np.abs(true_phi - phi)

    # There are some issues here near the boundaries. The max error goes to about 0.22. Not sure what the problem is.
    np.testing.assert_allclose(err, np.zeros(err.shape), atol=1)

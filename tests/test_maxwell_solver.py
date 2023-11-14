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

    grid = Uniform2DGrid(num_points=N, x0=-b, xm=b, y0=0, ym=a)
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
    # print(np.max(err))
    np.testing.assert_allclose(err, np.zeros(err.shape), atol=1e-2)


# return phi, true_phi, X, Y


def test_2d_poisson():
    """Same ex as before, but with a constant charge density"""
    N = 100
    V0 = 1
    a = 2
    b = 1

    grid = Uniform2DGrid(num_points=N, x0=-b, xm=b, y0=0, ym=a)
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

    # err = np.abs(true_phi[1:-1,1:-1] - phi[1:-1,1:-1])
    # print(np.max(err))
    # np.testing.assert_allclose(err, np.zeros(err.shape), atol=1e-2)

    return phi, true_phi, X, Y


def calc_true_phi(X, Y):
    max_n = 250
    V0 = 1
    b = 1
    a = 2
    true_phi = 0
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
    return true_phi


def test_2d_uniform():
    """Test 2D Maxwell solver for a uniform charge distribution"""
    ms = MaxwellSolver2D()
    rho = np.ones(ms.grid[0].shape)
    phi = ms.solve(rho)

    return phi


# def ex_poission_2d():
#     N = 100
#     N2 = (N - 1) * (N - 1)
#     A = np.zeros((N2, N2))
#     ## Diagonal
#     for i in range(N - 1):
#         for j in range(N - 1):
#             A[i + (N - 1) * j, i + (N - 1) * j] = -4

#     # LOWER DIAGONAL
#     for i in range(1, N - 1):
#         for j in range(N - 1):
#             A[i + (N - 1) * j, i + (N - 1) * j - 1] = 1
#     # UPPPER DIAGONAL
#     for i in range(N - 2):
#         for j in range(N - 1):
#             A[i + (N - 1) * j, i + (N - 1) * j + 1] = 1

#     # LOWER IDENTITY MATRIX
#     for i in range(N - 1):
#         for j in range(1, N - 1):
#             A[i + (N - 1) * j, i + (N - 1) * (j - 1)] = 1

#     # UPPER IDENTITY MATRIX
#     for i in range(N - 1):
#         for j in range(N - 2):
#             A[i + (N - 1) * j, i + (N - 1) * (j + 1)] = 1

#     # Ainv=np.linalg.inv(A)

#     rho = np.ones((N - 1, N - 1))
#     rho_v = rho.ravel()

#     r = np.zeros(N2)

#     x = np.linspace(-1, 1, 100)
#     y = np.linspace(-1, 1, 100)
#     h = x[1] - x[0]

#     r = -4 * np.pi * h**2 * rho_v
#     bc = {
#         "bottom": np.zeros(N),
#         "top": np.zeros(N),
#         "left": np.zeros(N),
#         "right": np.zeros(N),
#     }
#     # # vector r
#     # for i in range (0,N-1):
#     #     for j in range (0,N-1):
#     #         r[i+(N-1)*j]=100*h*h*(x[i+1]*x[i+1]+y[j+1]*y[j+1])
#     # Boundary
#     b_bottom_top = np.zeros(N2)
#     for i in range(N - 1):
#         b_bottom_top[i] = bc["bottom"][i + 1]  # Bottom Boundary
#         b_bottom_top[i + (N - 1) * (N - 2)] = bc["top"][i + 1]  # Top Boundary

#     b_left_right = np.zeros(N2)
#     for j in range(N - 1):
#         b_left_right[(N - 1) * j] = bc["left"][j + 1]  # Left Boundary
#         b_left_right[N - 2 + (N - 1) * j] = bc["right"][j + 1]  # Right Boundary

#     b = b_left_right + b_bottom_top

#     rhs = r + b_left_right + b_bottom_top
#     phi_v = spsolve(A, rhs)
#     phi = np.zeros((N, N))
#     phi[0, :] = bc["top"]
#     phi[:, -1] = bc["right"]
#     phi[-1, :] = bc["bottom"]
#     phi[:, 0] = bc["left"]

#     phi[1:N, 1:N] = phi_v.reshape((N - 1, N - 1))
#     return phi


# # phi = ex_poission_2d()


# def test_A(N):
#     N2 = (N - 2) * (N - 2)
#     A = np.zeros((N2, N2))
#     ## Diagonal
#     for i in range(N - 2):
#         for j in range(N - 2):
#             A[i + (N - 2) * j, i + (N - 2) * j] = -4

#     # LOWER DIAGONAL
#     for i in range(1, N - 2):
#         for j in range(N - 2):
#             A[i + (N - 2) * j, i + (N - 2) * j - 1] = 1
#     # UPPPER DIAGONAL
#     for i in range(N - 3):
#         for j in range(N - 2):
#             A[i + (N - 2) * j, i + (N - 2) * j + 1] = 1

#     # LOWER IDENTITY MATRIX
#     for i in range(N - 2):
#         for j in range(1, N - 2):
#             A[i + (N - 2) * j, i + (N - 2) * (j - 1)] = 1

#     # UPPER IDENTITY MATRIX
#     for i in range(N - 2):
#         for j in range(N - 3):
#             A[i + (N - 2) * j, i + (N - 2) * (j + 1)] = 1

#     plt.imshow(A, interpolation="none")

#     return A


# # A = test_A(4)

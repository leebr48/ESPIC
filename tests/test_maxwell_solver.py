import numpy as np
from solve_maxwell import MaxwellSolver1D

#Test 1D Maxwell solver for a uniform charge distribution
def test_1d_uniform():
    ms = MaxwellSolver1D()
    rho = np.ones(len(ms.grid))
    phi = ms.solve(rho)

    true_phi = -2*np.pi*(ms.grid**2 - 1)
    err = np.abs(true_phi - phi)
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)
"""Test charge initialization"""

import numpy as np

from espic.init_charges import Initialize


def test_uniform_distribution():
    # Simple statistical test of uniform distribution
    num_particles = 10000
    num_dim = 10000
    lower = 0
    upper = 1
    dist = Initialize(num_particles, num_dim).uniform(lower, upper)

    assert np.all(dist >= lower)
    assert np.all(dist < upper)
    assert dist.shape == (10000, 10000)

    num_particles = int(1e8)
    num_dim = 1
    dist = Initialize(num_particles, num_dim).uniform(lower, upper)

    assert np.allclose(np.average(dist), np.asarray([(lower + upper) / 2]), atol=1e-3, rtol=0)


def test_normal_distribution():
    # Simple statistical test of normal distribution
    num_particles = int(1e8)
    num_dim = 1
    mean = 0.2
    stdev = 1.3
    dist = Initialize(num_particles, num_dim).normal(mean, stdev)

    assert np.allclose(mean, np.mean(dist), atol=1e-3, rtol=0)
    assert np.allclose(stdev, np.std(dist, ddof=1), atol=1e-3, rtol=0)

    num_particles = 7
    num_dim = 12
    dist = Initialize(num_particles, num_dim).normal(mean, stdev)

    assert dist.shape == (7, 12)


def test_maxwellian_distrubtion():
    # Simple statistical test of maxwellian distribution
    num_particles = int(1e8)
    num_dim = 1
    spread = 1.2
    start = 0
    dist = Initialize(num_particles, num_dim).maxwellian(spread, start=start)

    assert np.allclose(2 * spread * np.sqrt(2 / np.pi), np.mean(dist), atol=1e-3, rtol=0)
    assert np.allclose(spread * np.sqrt((3 * np.pi - 8) / np.pi), np.std(dist, ddof=1),
        atol=1e-3, rtol=0)

    num_particles = 9
    num_dim = 10
    dist = Initialize(num_particles, num_dim).maxwellian(spread, start=start)

    assert dist.shape == (9, 10)

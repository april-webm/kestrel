# tests/test_cir_process.py
"""Tests for CIR process implementation."""

import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.cir import CIRProcess
from kestrel.utils.kestrel_result import KestrelResult


@pytest.fixture
def sample_cir_data():
    """Generates CIR process data for testing."""
    np.random.seed(44)

    # CIR parameters
    kappa = 0.5
    theta = 0.05
    sigma = 0.1
    dt = 1 / 252
    n_steps = 500

    x0 = 0.04
    data = [x0]

    for _ in range(n_steps - 1):
        x_curr = data[-1]
        dW = np.random.normal(0, np.sqrt(dt))
        dx = kappa * (theta - x_curr) * dt + sigma * np.sqrt(max(0, x_curr)) * dW
        x_next = max(1e-6, x_curr + dx)  # Ensure positivity
        data.append(x_next)

    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
    return pd.Series(data, index=dates)


@pytest.fixture
def simple_positive_data():
    """Simple positive data for basic testing."""
    np.random.seed(45)
    data = np.random.uniform(0.01, 0.1, 100)
    return pd.Series(data)


# Initialisation tests
def test_cir_process_init_default():
    """Test default initialisation of CIRProcess."""
    cir = CIRProcess()
    assert cir.kappa is None
    assert cir.theta is None
    assert cir.sigma is None
    assert not cir.is_fitted


def test_cir_process_init_with_params():
    """Test initialisation of CIRProcess with parameters."""
    cir = CIRProcess(kappa=0.1, theta=0.05, sigma=0.02)
    assert cir.kappa == 0.1
    assert cir.theta == 0.05
    assert cir.sigma == 0.02
    assert not cir.is_fitted


# Fit method tests
def test_cir_process_fit_mle(sample_cir_data):
    """Test MLE fitting of CIRProcess."""
    cir = CIRProcess()
    cir.fit(sample_cir_data, method='mle')

    assert cir.is_fitted
    assert hasattr(cir, 'kappa_')
    assert hasattr(cir, 'theta_')
    assert hasattr(cir, 'sigma_')
    assert cir.kappa_ > 0
    assert cir.theta_ > 0
    assert cir.sigma_ > 0


def test_cir_process_fit_lsq(sample_cir_data):
    """Test LSQ fitting of CIRProcess."""
    cir = CIRProcess()
    cir.fit(sample_cir_data, method='lsq')

    assert cir.is_fitted
    assert hasattr(cir, 'kappa_')
    assert hasattr(cir, 'theta_')
    assert hasattr(cir, 'sigma_')
    assert cir.kappa_ > 0
    assert cir.theta_ > 0
    assert cir.sigma_ > 0


def test_cir_process_fit_with_explicit_dt(simple_positive_data):
    """Test fitting with explicit dt."""
    cir = CIRProcess()
    cir.fit(simple_positive_data, dt=1 / 252, method='lsq')

    assert cir.is_fitted
    assert cir._dt_ == 1 / 252


# Sample method tests
def test_cir_process_sample_after_fit(sample_cir_data):
    """Test sampling after fitting."""
    cir = CIRProcess()
    cir.fit(sample_cir_data, method='lsq')

    n_paths = 5
    horizon = 10
    sim_paths = cir.sample(n_paths=n_paths, horizon=horizon)

    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)
    assert (sim_paths.paths.values >= 0).all()  # CIR paths non-negative


def test_cir_process_sample_with_initial_params():
    """Test sampling with parameters provided at initialisation."""
    kappa, theta, sigma = 0.5, 0.05, 0.1
    cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma)

    n_paths = 5
    horizon = 10
    sim_paths = cir.sample(n_paths=n_paths, horizon=horizon, dt=1 / 252)

    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)
    assert (sim_paths.paths.values >= 0).all()


def test_cir_process_sample_unfitted_no_params_raises_error():
    """Test sampling without fitting or initial params raises error."""
    cir = CIRProcess()
    with pytest.raises(RuntimeError):
        cir.sample(n_paths=1, horizon=1)


# Error handling tests
def test_cir_process_fit_invalid_data_type():
    """Test fitting with invalid data type."""
    cir = CIRProcess()
    with pytest.raises(ValueError, match="Input data must be a pandas Series."):
        cir.fit([1, 2, 3])


def test_cir_process_fit_non_positive_data():
    """Test fitting with non-positive data raises error."""
    cir = CIRProcess()
    data = pd.Series([-0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="CIR process requires strictly positive data."):
        cir.fit(data)


def test_cir_process_fit_unknown_method():
    """Test fitting with unknown method."""
    cir = CIRProcess()
    data = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="Unknown estimation method"):
        cir.fit(data, method='invalid')


# Feller condition test
def test_cir_feller_condition():
    """Test Feller condition checking."""
    # Satisfies Feller: 2*0.5*0.05 = 0.05 > 0.01^2 = 0.0001
    cir_good = CIRProcess(kappa=0.5, theta=0.05, sigma=0.01)
    assert cir_good.feller_condition_satisfied()

    # Does not satisfy Feller: 2*0.1*0.01 = 0.002 < 0.1^2 = 0.01
    cir_bad = CIRProcess(kappa=0.1, theta=0.01, sigma=0.1)
    assert not cir_bad.feller_condition_satisfied()

# tests/test_merton_process.py
"""Tests for Merton Jump Diffusion process implementation."""

import pytest
import pandas as pd
import numpy as np
from kestrel.jump_diffusion.merton import MertonProcess
from kestrel.utils.kestrel_result import KestrelResult


@pytest.fixture
def sample_merton_data():
    """Generates Merton jump-diffusion data for testing."""
    np.random.seed(45)

    # Parameters
    mu = 0.05
    sigma = 0.2
    lambda_ = 1.0  # 1 jump per year on average
    jump_mu = -0.02
    jump_sigma = 0.05
    dt = 1 / 252
    n_steps = 500

    returns = []
    for _ in range(n_steps):
        # Continuous component
        continuous = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal()

        # Jump component
        n_jumps = np.random.poisson(lambda_ * dt)
        if n_jumps > 0:
            jump = np.sum(np.random.normal(jump_mu, jump_sigma, n_jumps))
        else:
            jump = 0

        returns.append(continuous + jump)

    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
    return pd.Series(returns, index=dates)


@pytest.fixture
def simple_returns_data():
    """Simple returns data for basic testing."""
    np.random.seed(46)
    # Generate returns with some fat tails (mixture)
    n = 200
    normal_returns = np.random.normal(0, 0.01, int(n * 0.9))
    jump_returns = np.random.normal(-0.03, 0.02, int(n * 0.1))
    returns = np.concatenate([normal_returns, jump_returns])
    np.random.shuffle(returns)
    return pd.Series(returns)


# Initialisation tests
def test_merton_process_init_default():
    """Test default initialisation of MertonProcess."""
    merton = MertonProcess()
    assert merton.mu is None
    assert merton.sigma is None
    assert merton.lambda_ is None
    assert merton.jump_mu is None
    assert merton.jump_sigma is None
    assert not merton.is_fitted


def test_merton_process_init_with_params():
    """Test initialisation of MertonProcess with parameters."""
    merton = MertonProcess(mu=0.05, sigma=0.2, lambda_=1.0, jump_mu=-0.02, jump_sigma=0.05)
    assert merton.mu == 0.05
    assert merton.sigma == 0.2
    assert merton.lambda_ == 1.0
    assert merton.jump_mu == -0.02
    assert merton.jump_sigma == 0.05
    assert not merton.is_fitted


# Fit method tests
def test_merton_process_fit_mle(sample_merton_data):
    """Test MLE fitting of MertonProcess."""
    merton = MertonProcess()
    merton.fit(sample_merton_data, method='mle')

    assert merton.is_fitted
    assert hasattr(merton, 'mu_')
    assert hasattr(merton, 'sigma_')
    assert hasattr(merton, 'lambda_')
    assert hasattr(merton, 'jump_mu_')
    assert hasattr(merton, 'jump_sigma_')
    assert merton.sigma_ > 0
    assert merton.lambda_ > 0
    assert merton.jump_sigma_ > 0


def test_merton_process_fit_with_explicit_dt(simple_returns_data):
    """Test fitting with explicit dt."""
    merton = MertonProcess()
    merton.fit(simple_returns_data, dt=1 / 252, method='mle')

    assert merton.is_fitted
    assert merton._dt_ == 1 / 252


# Sample method tests
def test_merton_process_sample_after_fit(sample_merton_data):
    """Test sampling after fitting."""
    merton = MertonProcess()
    merton.fit(sample_merton_data, method='mle')

    n_paths = 5
    horizon = 10
    sim_paths = merton.sample(n_paths=n_paths, horizon=horizon)

    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)


def test_merton_process_sample_with_initial_params():
    """Test sampling with parameters provided at initialisation."""
    merton = MertonProcess(mu=0.05, sigma=0.2, lambda_=1.0, jump_mu=-0.02, jump_sigma=0.05)

    n_paths = 5
    horizon = 10
    sim_paths = merton.sample(n_paths=n_paths, horizon=horizon, dt=1 / 252)

    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)


def test_merton_process_sample_unfitted_no_params_raises_error():
    """Test sampling without fitting or initial params raises error."""
    merton = MertonProcess()
    with pytest.raises(RuntimeError):
        merton.sample(n_paths=1, horizon=1)


# Error handling tests
def test_merton_process_fit_invalid_data_type():
    """Test fitting with invalid data type."""
    merton = MertonProcess()
    with pytest.raises(ValueError, match="Input data must be a pandas Series."):
        merton.fit([1, 2, 3])


def test_merton_process_fit_unknown_method():
    """Test fitting with unknown method."""
    merton = MertonProcess()
    data = pd.Series(np.random.normal(0, 0.01, 20))
    with pytest.raises(ValueError, match="Unknown estimation method"):
        merton.fit(data, method='invalid')


def test_merton_process_fit_insufficient_data():
    """Test fitting with insufficient data."""
    merton = MertonProcess()
    data = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="MLE estimation requires at least 10 data points."):
        merton.fit(data, method='mle')


# Analytics tests
def test_merton_expected_return():
    """Test expected return calculation."""
    merton = MertonProcess(mu=0.1, sigma=0.2, lambda_=2.0, jump_mu=-0.01, jump_sigma=0.05)

    # E[r] = mu - 0.5*sigma^2 + lambda*jump_mu
    # = 0.1 - 0.5*0.04 + 2*(-0.01) = 0.1 - 0.02 - 0.02 = 0.06
    expected = merton.expected_return()
    assert expected == pytest.approx(0.06, rel=1e-6)


def test_merton_total_variance():
    """Test total variance calculation."""
    merton = MertonProcess(mu=0.1, sigma=0.2, lambda_=2.0, jump_mu=-0.01, jump_sigma=0.05)

    # Var[r] = sigma^2 + lambda*(jump_mu^2 + jump_sigma^2)
    # = 0.04 + 2*(0.0001 + 0.0025) = 0.04 + 2*0.0026 = 0.04 + 0.0052 = 0.0452
    variance = merton.total_variance()
    assert variance == pytest.approx(0.0452, rel=1e-6)

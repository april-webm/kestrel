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
def test_merton_process_fit_em(sample_merton_data):
    """Test EM fitting of MertonProcess."""
    merton = MertonProcess()
    result = merton.fit(sample_merton_data, method='em')

    assert merton.is_fitted
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert result.params['lambda_'] is not None
    assert result.params['jump_mu'] is not None
    assert result.params['jump_sigma'] is not None
    assert result.params['sigma'] > 0
    assert result.params['lambda_'] > 0
    assert result.params['jump_sigma'] > 0
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_merton_data)

    # Check that the parameters are also stored on the model instance for sampling
    assert merton.mu == result.params['mu']
    assert merton.sigma == result.params['sigma']
    assert merton.lambda_ == result.params['lambda_']
    assert merton.jump_mu == result.params['jump_mu']
    assert merton.jump_sigma == result.params['jump_sigma']


def test_merton_process_fit_with_explicit_dt(simple_returns_data):
    """Test fitting with explicit dt."""
    merton = MertonProcess()
    result = merton.fit(simple_returns_data, dt=1 / 252, method='em')

    assert merton.is_fitted
    assert merton._dt_ == 1 / 252 # _dt_ is stored on the instance


# Sample method tests
def test_merton_process_sample_after_fit(sample_merton_data):
    """Test sampling after fitting."""
    merton = MertonProcess()
    merton.fit(sample_merton_data, method='em') # Fit method updates merton parameters

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
    with pytest.raises(ValueError, match="EM estimation requires at least 10 data points."):
        merton.fit(data, method='em')


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
    # = 0.04 + 2*(0.0001 + 0.0025) = 0.04 + 0.0052 = 0.0452
    variance = merton.total_variance()
    assert variance == pytest.approx(0.0452, rel=1e-6)

def test_residuals_properties(sample_merton_data):
    """Test residuals and associated properties."""
    merton = MertonProcess()
    result = merton.fit(sample_merton_data)

    assert result.residuals is not None
    assert isinstance(result.residuals, pd.Series)
    # For Merton, residuals are log-density contributions. We expect them to be negative.
    assert (result.residuals < 0).all()

    # Ljung-Box and normality tests on log-density residuals might not be directly interpretable
    # in the same way as for standardized Gaussian residuals, but we ensure they run without error.
    lb_test_results = result.ljung_box()
    assert lb_test_results is not None
    assert isinstance(lb_test_results, pd.DataFrame)
    assert 'lb_stat' in lb_test_results.columns

    normality_test_results = result.normality_test()
    assert normality_test_results is not None
    assert 'statistic' in normality_test_results
    assert 'p-value' in normality_test_results
    result.qq_plot()

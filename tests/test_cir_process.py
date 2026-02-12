# tests/test_cir_process.py
"""Tests for CIR process implementation."""

import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.cir import CIRProcess
from kestrel.utils.kestrel_result import KestrelResult
import warnings


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
    result = cir.fit(sample_cir_data, method='mle')

    assert cir.is_fitted
    assert result.params['kappa'] is not None
    assert result.params['theta'] is not None
    assert result.params['sigma'] is not None
    assert result.params['kappa'] > 0
    assert result.params['theta'] > 0
    assert result.params['sigma'] > 0
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_cir_data) - 1

    # Check that the parameters are also stored on the model instance for sampling
    assert cir.kappa == result.params['kappa']
    assert cir.theta == result.params['theta']
    assert cir.sigma == result.params['sigma']


def test_cir_process_fit_lsq(sample_cir_data):
    """Test LSQ fitting of CIRProcess."""
    cir = CIRProcess()
    result = cir.fit(sample_cir_data, method='lsq')

    assert cir.is_fitted
    assert result.params['kappa'] is not None
    assert result.params['theta'] is not None
    assert result.params['sigma'] is not None
    assert result.params['kappa'] > 0
    assert result.params['theta'] > 0
    assert result.params['sigma'] > 0
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_cir_data) - 1

    # Check that the parameters are also stored on the model instance for sampling
    assert cir.kappa == result.params['kappa']
    assert cir.theta == result.params['theta']
    assert cir.sigma == result.params['sigma']


def test_cir_process_fit_with_explicit_dt(simple_positive_data):
    """Test fitting with explicit dt."""
    cir = CIRProcess()
    result = cir.fit(simple_positive_data, dt=1 / 252, method='lsq')

    assert cir.is_fitted
    assert cir._dt_ == 1 / 252 # _dt_ is stored on the instance


# Sample method tests
def test_cir_process_sample_after_fit(sample_cir_data):
    """Test sampling after fitting."""
    cir = CIRProcess()
    cir.fit(sample_cir_data, method='lsq') # Fit method updates cir.kappa, cir.theta, cir.sigma

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
    cir = CIRProcess() # Use an unfitted instance
    # Satisfies Feller: 2*0.5*0.05 = 0.05 >= 0.01^2 = 0.0001
    assert cir.feller_condition_satisfied(kappa=0.5, theta=0.05, sigma=0.01) is True

    # Violates Feller: 2*0.1*0.01 = 0.002 < 0.25 = 0.5^2
    assert cir.feller_condition_satisfied(kappa=0.1, theta=0.01, sigma=0.5) is False

def test_cir_exact_sampling_moments():
    """Test that exact CIR sampling produces non-negative values and plausible moments."""
    np.random.seed(500)
    kappa, theta, sigma, dt = 0.5, 0.05, 0.1, 1/252
    x0 = 0.04
    horizon = 252 # Simulate for 1 year
    n_paths = 50000

    cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma)
    sim_result = cir.sample(n_paths=n_paths, horizon=horizon, dt=dt)

    terminal_values = sim_result.paths.iloc[-1, :]
    
    # Check non-negativity
    assert (terminal_values >= 0).all()

    T = horizon * dt
    # Theoretical Expected value: E[X_T] = theta + (X_0 - theta) * e^{-kappa*T}
    expected_mean = theta + (x0 - theta) * np.exp(-kappa * T)
    
    # Theoretical Variance: Var[X_T] = X_0 * sigma^2/kappa * (e^{-kappa*T} - e^{-2*kappa*T}) 
    #                                 + theta * sigma^2/(2*kappa) * (1 - e^{-kappa*T})^2
    expected_variance = (x0 * sigma**2 / kappa * (np.exp(-kappa * T) - np.exp(-2 * kappa * T)) +
                         theta * sigma**2 / (2 * kappa) * (1 - np.exp(-kappa * T))**2)
    
    assert np.isclose(terminal_values.mean(), expected_mean, rtol=0.1, atol=0.01)
    assert np.isclose(terminal_values.var(), expected_variance, rtol=0.2, atol=0.01)


def test_cir_exact_sampling_fallback_on_feller_violation():
    """Test that sampling falls back to Euler-Maruyama if Feller condition is violated."""
    np.random.seed(501)
    # Parameters that violate Feller: 2*kappa*theta = 0.002 < sigma^2 = 0.25
    kappa, theta, sigma, dt = 0.1, 0.01, 0.5, 1/252
    x0 = 0.04
    horizon = 10
    n_paths = 5

    cir = CIRProcess(kappa=kappa, theta=theta, sigma=sigma)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim_result = cir.sample(n_paths=n_paths, horizon=horizon, dt=dt)
        assert any("falling back to Euler-Maruyama" in str(warn.message).lower() for warn in w)
    
    assert isinstance(sim_result, KestrelResult)
    assert sim_result.paths.shape == (horizon + 1, n_paths)
    assert (sim_result.paths.values >= 0).all() # Should still be non-negative due to reflection


def test_residuals_properties(sample_cir_data):
    """Test residuals and associated properties."""
    cir = CIRProcess()
    result = cir.fit(sample_cir_data)

    assert result.residuals is not None
    assert isinstance(result.residuals, pd.Series)
    # Residuals for CIR are based on Gaussian approx, so mean should be close to 0
    assert np.isclose(result.residuals.mean(), 0, atol=0.1)
    # Std dev close to 1 for standardized residuals
    assert np.isclose(result.residuals.std(), 1, atol=0.1)

    lb_test_results = result.ljung_box()
    assert lb_test_results is not None
    assert isinstance(lb_test_results, pd.DataFrame)
    assert 'lb_stat' in lb_test_results.columns

    normality_test_results = result.normality_test()
    assert normality_test_results is not None
    assert 'statistic' in normality_test_results
    assert 'p-value' in normality_test_results
    result.qq_plot()

# tests/test_brownian_process.py
"""Tests for Brownian motion and GBM implementations."""

import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.brownian import BrownianMotion, GeometricBrownianMotion
from kestrel.utils.kestrel_result import KestrelResult
from scipy.stats import norm


@pytest.fixture
def sample_brownian_data():
    """Generates Brownian motion data for testing."""
    np.random.seed(50)

    mu = 0.05
    sigma = 0.2
    dt = 1 / 252
    n_steps = 500

    x0 = 0.0
    data = [x0]
    for _ in range(n_steps - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        x_next = data[-1] + mu * dt + sigma * dW
        data.append(x_next)

    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
    return pd.Series(data, index=dates)


@pytest.fixture
def sample_gbm_data():
    """Generates GBM price data for testing."""
    np.random.seed(51)

    mu = 0.1
    sigma = 0.3
    dt = 1 / 252
    n_steps = 500

    s0 = 100.0
    prices = [s0]
    for _ in range(n_steps - 1):
        Z = np.random.normal()
        s_next = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(s_next)

    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
    return pd.Series(prices, index=dates)


# BrownianMotion tests
class TestBrownianMotion:
    """Tests for BrownianMotion class."""

    def test_init_default(self):
        """Test default initialisation."""
        bm = BrownianMotion()
        assert bm.mu is None
        assert bm.sigma is None
        assert not bm.is_fitted

    def test_init_with_params(self):
        """Test initialisation with parameters."""
        bm = BrownianMotion(mu=0.05, sigma=0.2)
        assert bm.mu == 0.05
        assert bm.sigma == 0.2
        assert not bm.is_fitted

    def test_fit_mle(self, sample_brownian_data):
        """Test MLE fitting."""
        bm = BrownianMotion()
        result = bm.fit(sample_brownian_data, method='mle')

        assert bm.is_fitted
        assert result.params['mu'] is not None
        assert result.params['sigma'] > 0
        assert result.log_likelihood is not None
        assert result.aic is not None
        assert result.bic is not None
        assert result.residuals is not None
        assert len(result.residuals) == len(sample_brownian_data) - 1

        # Check that the parameters are also stored on the model instance for sampling
        assert bm.mu == result.params['mu']
        assert bm.sigma == result.params['sigma']

    def test_fit_moments(self, sample_brownian_data):
        """Test moments fitting."""
        bm = BrownianMotion()
        result = bm.fit(sample_brownian_data, method='moments')

        assert bm.is_fitted
        assert result.params['mu'] is not None
        assert result.params['sigma'] > 0
        assert result.log_likelihood is not None
        assert result.aic is not None
        assert result.bic is not None
        assert result.residuals is not None
        assert len(result.residuals) == len(sample_brownian_data) - 1

        # Check that the parameters are also stored on the model instance for sampling
        assert bm.mu == result.params['mu']
        assert bm.sigma == result.params['sigma']

    def test_fit_gmm(self, sample_brownian_data):
        """Test GMM fitting."""
        true_mu, true_sigma, dt = 0.05, 0.2, 1/252
        bm = BrownianMotion()
        result = bm.fit(sample_brownian_data, dt=dt, method='gmm')

        assert bm.is_fitted
        assert result.params['mu'] is not None
        assert result.params['sigma'] > 0
        assert result.log_likelihood is not None
        assert result.aic is not None
        assert result.bic is not None
        assert result.residuals is not None
        assert len(result.residuals) == len(sample_brownian_data) - 1

        # Check if estimated parameters are reasonably close to true parameters
        assert np.isclose(result.params['mu'], true_mu, rtol=0.1, atol=0.01)
        assert np.isclose(result.params['sigma'], true_sigma, rtol=0.1, atol=0.01)

        # Check that the parameters are also stored on the model instance for sampling
        assert bm.mu == result.params['mu']
        assert bm.sigma == result.params['sigma']


    def test_sample_after_fit(self, sample_brownian_data):
        """Test sampling after fitting."""
        bm = BrownianMotion()
        bm.fit(sample_brownian_data, method='mle') # Fit method updates bm.mu, bm.sigma

        n_paths = 5
        horizon = 10
        sim_paths = bm.sample(n_paths=n_paths, horizon=horizon)

        assert isinstance(sim_paths, KestrelResult)
        assert sim_paths.paths.shape == (horizon + 1, n_paths)

    def test_sample_with_initial_params(self):
        """Test sampling with initialised parameters."""
        bm = BrownianMotion(mu=0.05, sigma=0.2)

        n_paths = 5
        horizon = 10
        sim_paths = bm.sample(n_paths=n_paths, horizon=horizon, dt=1 / 252)

        assert isinstance(sim_paths, KestrelResult)
        assert sim_paths.paths.shape == (horizon + 1, n_paths)

    def test_sample_unfitted_no_params_raises_error(self):
        """Test sampling without fitting or params raises error."""
        bm = BrownianMotion()
        with pytest.raises(RuntimeError):
            bm.sample(n_paths=1, horizon=1)

    def test_fit_invalid_data_type(self):
        """Test fitting with invalid data type."""
        bm = BrownianMotion()
        with pytest.raises(ValueError, match="Input data must be a pandas Series."):
            bm.fit([1, 2, 3])

    def test_fit_unknown_method(self):
        """Test fitting with unknown method."""
        bm = BrownianMotion()
        data = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown estimation method"):
            bm.fit(data, method='invalid')

    def test_residuals_properties(self, sample_brownian_data):
        """Test residuals and associated properties."""
        bm = BrownianMotion()
        result = bm.fit(sample_brownian_data)

        assert result.residuals is not None
        assert isinstance(result.residuals, pd.Series)
        assert np.isclose(result.residuals.mean(), 0, atol=0.1) # Standardized residuals should have mean close to 0
        assert np.isclose(result.residuals.std(), 1, atol=0.1)  # Standardized residuals should have std dev close to 1

        lb_test_results = result.ljung_box()
        assert lb_test_results is not None
        assert isinstance(lb_test_results, pd.DataFrame)
        assert 'lb_stat' in lb_test_results.columns

        normality_test_results = result.normality_test()
        assert normality_test_results is not None
        assert 'statistic' in normality_test_results
        assert 'p-value' in normality_test_results
        # No easy assertion for Q-Q plot; just ensure it doesn't raise an error.
        result.qq_plot()

# GeometricBrownianMotion tests
class TestGeometricBrownianMotion:
    """Tests for GeometricBrownianMotion class."""

    def test_init_default(self):
        """Test default initialisation."""
        gbm = GeometricBrownianMotion()
        assert gbm.mu is None
        assert gbm.sigma is None
        assert not gbm.is_fitted

    def test_init_with_params(self):
        """Test initialisation with parameters."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.3)
        assert gbm.mu == 0.1
        assert gbm.sigma == 0.3
        assert not gbm.is_fitted

    def test_fit_mle(self, sample_gbm_data):
        """Test MLE fitting."""
        gbm = GeometricBrownianMotion()
        result = gbm.fit(sample_gbm_data, method='mle')

        assert gbm.is_fitted
        assert result.params['mu'] is not None
        assert result.params['sigma'] > 0
        assert result.log_likelihood is not None
        assert result.aic is not None
        assert result.bic is not None
        assert result.residuals is not None
        assert len(result.residuals) == len(sample_gbm_data) - 1

        # Check that the parameters are also stored on the model instance for sampling
        assert gbm.mu == result.params['mu']
        assert gbm.sigma == result.params['sigma']

    def test_sample_after_fit(self, sample_gbm_data):
        """Test sampling after fitting."""
        gbm = GeometricBrownianMotion()
        gbm.fit(sample_gbm_data, method='mle') # Fit method updates gbm.mu, gbm.sigma

        n_paths = 5
        horizon = 10
        sim_paths = gbm.sample(n_paths=n_paths, horizon=horizon)

        assert isinstance(sim_paths, KestrelResult)
        assert sim_paths.paths.shape == (horizon + 1, n_paths)
        assert (sim_paths.paths.values > 0).all()  # GBM prices strictly positive

    def test_sample_with_initial_params(self):
        """Test sampling with initialised parameters."""
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.3)

        n_paths = 5
        horizon = 10
        sim_paths = gbm.sample(n_paths=n_paths, horizon=horizon, dt=1 / 252)

        assert isinstance(sim_paths, KestrelResult)
        assert sim_paths.paths.shape == (horizon + 1, n_paths)
        assert (sim_paths.paths.values > 0).all()

    def test_sample_unfitted_no_params_raises_error(self):
        """Test sampling without fitting or params raises error."""
        gbm = GeometricBrownianMotion()
        with pytest.raises(RuntimeError):
            gbm.sample(n_paths=1, horizon=1)

    def test_fit_non_positive_data_raises_error(self):
        """Test fitting with non-positive data raises error."""
        gbm = GeometricBrownianMotion()
        data = pd.Series([100, -50, 80])
        with pytest.raises(ValueError, match="GBM requires strictly positive price data."):
            gbm.fit(data)

    def test_expected_price(self, sample_gbm_data):
        """Test expected price calculation."""
        gbm = GeometricBrownianMotion()
        result = gbm.fit(sample_gbm_data, method='mle')

        s0 = 100.0
        t = 1.0
        expected = gbm.expected_price(t, s0)

        # E[S_t] = S_0 * exp(mu * t)
        assert expected == pytest.approx(result.params['mu'] * np.exp(result.params['mu'] * t), rel=1e-6)

    def test_variance_price(self, sample_gbm_data):
        """Test variance calculation."""
        gbm = GeometricBrownianMotion()
        result = gbm.fit(sample_gbm_data, method='mle')

        s0 = 100.0
        t = 1.0
        variance = gbm.variance_price(t, s0)

        # Var[S_t] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
        expected_var = (s0 ** 2) * np.exp(2 * result.params['mu'] * t) * (np.exp(result.params['sigma'] ** 2 * t) - 1)
        assert variance == pytest.approx(expected_var, rel=1e-6)

    def test_residuals_properties(self, sample_gbm_data):
        """Test residuals and associated properties."""
        gbm = GeometricBrownianMotion()
        result = gbm.fit(sample_gbm_data)

        assert result.residuals is not None
        assert isinstance(result.residuals, pd.Series)
        assert np.isclose(result.residuals.mean(), 0, atol=0.1)
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
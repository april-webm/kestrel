# tests/test_brownian_process.py
"""Tests for Brownian motion and GBM implementations."""

import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.brownian import BrownianMotion, GeometricBrownianMotion
from kestrel.utils.kestrel_result import KestrelResult


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
        bm.fit(sample_brownian_data, method='mle')

        assert bm.is_fitted
        assert hasattr(bm, 'mu_')
        assert hasattr(bm, 'sigma_')
        assert bm.sigma_ > 0

    def test_fit_moments(self, sample_brownian_data):
        """Test moments fitting."""
        bm = BrownianMotion()
        bm.fit(sample_brownian_data, method='moments')

        assert bm.is_fitted
        assert hasattr(bm, 'mu_')
        assert hasattr(bm, 'sigma_')
        assert bm.sigma_ > 0

    def test_sample_after_fit(self, sample_brownian_data):
        """Test sampling after fitting."""
        bm = BrownianMotion()
        bm.fit(sample_brownian_data, method='mle')

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
        gbm.fit(sample_gbm_data, method='mle')

        assert gbm.is_fitted
        assert hasattr(gbm, 'mu_')
        assert hasattr(gbm, 'sigma_')
        assert gbm.sigma_ > 0

    def test_sample_after_fit(self, sample_gbm_data):
        """Test sampling after fitting."""
        gbm = GeometricBrownianMotion()
        gbm.fit(sample_gbm_data, method='mle')

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
        gbm.fit(sample_gbm_data, method='mle')

        s0 = 100.0
        t = 1.0
        expected = gbm.expected_price(t, s0)

        # E[S_t] = S_0 * exp(mu * t)
        assert expected == pytest.approx(s0 * np.exp(gbm.mu_ * t), rel=1e-6)

    def test_variance_price(self, sample_gbm_data):
        """Test variance calculation."""
        gbm = GeometricBrownianMotion()
        gbm.fit(sample_gbm_data, method='mle')

        s0 = 100.0
        t = 1.0
        variance = gbm.variance_price(t, s0)

        # Var[S_t] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
        expected_var = (s0 ** 2) * np.exp(2 * gbm.mu_ * t) * (np.exp(gbm.sigma_ ** 2 * t) - 1)
        assert variance == pytest.approx(expected_var, rel=1e-6)

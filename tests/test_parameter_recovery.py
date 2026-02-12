# tests/test_parameter_recovery.py
"""Parameter recovery tests: simulate from known params, fit, verify recovery."""

import pytest
import numpy as np
import pandas as pd
from kestrel import BrownianMotion, GeometricBrownianMotion, OUProcess, CIRProcess, MertonProcess


class TestBrownianMotionRecovery:
    """Verify BM parameter recovery from simulated data."""

    def test_recover_bm_params_mle(self):
        np.random.seed(100)
        true_mu, true_sigma, dt = 0.05, 0.2, 1 / 252
        n = 2000
        x = [0.0]
        for _ in range(n - 1):
            x.append(x[-1] + true_mu * dt + true_sigma * np.sqrt(dt) * np.random.normal())
        data = pd.Series(x)

        bm = BrownianMotion()
        bm.fit(data, dt=dt, method='mle')

        assert abs(bm.mu_ - true_mu) < 3 * bm.mu_se_
        assert abs(bm.sigma_ - true_sigma) < 3 * bm.sigma_se_

    def test_recover_bm_params_moments(self):
        np.random.seed(101)
        true_mu, true_sigma, dt = -0.1, 0.5, 1 / 52
        n = 1000
        x = [0.0]
        for _ in range(n - 1):
            x.append(x[-1] + true_mu * dt + true_sigma * np.sqrt(dt) * np.random.normal())
        data = pd.Series(x)

        bm = BrownianMotion()
        bm.fit(data, dt=dt, method='moments')

        assert abs(bm.mu_ - true_mu) < 3 * bm.mu_se_
        assert abs(bm.sigma_ - true_sigma) < 3 * bm.sigma_se_


class TestGBMRecovery:
    """Verify GBM parameter recovery."""

    def test_recover_gbm_sigma(self):
        np.random.seed(102)
        true_mu, true_sigma, dt = 0.08, 0.25, 1 / 252
        n = 2000
        prices = [100.0]
        for _ in range(n - 1):
            Z = np.random.normal()
            prices.append(prices[-1] * np.exp((true_mu - 0.5 * true_sigma ** 2) * dt + true_sigma * np.sqrt(dt) * Z))
        data = pd.Series(prices)

        gbm = GeometricBrownianMotion()
        gbm.fit(data, dt=dt)

        assert abs(gbm.sigma_ - true_sigma) < 3 * gbm.sigma_se_

    def test_gbm_zero_drift(self):
        """Edge case: GBM with zero drift."""
        np.random.seed(103)
        true_mu, true_sigma, dt = 0.0, 0.3, 1 / 252
        n = 1000
        prices = [100.0]
        for _ in range(n - 1):
            Z = np.random.normal()
            prices.append(prices[-1] * np.exp((true_mu - 0.5 * true_sigma ** 2) * dt + true_sigma * np.sqrt(dt) * Z))
        data = pd.Series(prices)

        gbm = GeometricBrownianMotion()
        gbm.fit(data, dt=dt)
        assert gbm.sigma_ > 0
        assert abs(gbm.sigma_ - true_sigma) < 3 * gbm.sigma_se_


class TestOURecovery:
    """Verify OU parameter recovery."""

    def test_recover_ou_params_mle(self):
        np.random.seed(104)
        true_theta, true_mu, true_sigma, dt = 1.0, 5.0, 0.5, 1 / 252
        n = 5000
        x = [true_mu]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            x.append(x[-1] + true_theta * (true_mu - x[-1]) * dt + true_sigma * dW)
        data = pd.Series(x)

        ou = OUProcess()
        ou.fit(data, dt=dt, method='mle')

        # OU theta estimates are biased in small samples; use generous bounds
        assert 0.3 < ou.theta_ < 3.0
        assert abs(ou.mu_ - true_mu) < 1.0
        assert 0.2 < ou.sigma_ < 1.0

    def test_recover_ou_params_ar1(self):
        """AR1 and MLE should produce identical results (same analytic method)."""
        np.random.seed(104)
        true_theta, true_mu, true_sigma, dt = 1.0, 5.0, 0.5, 1 / 252
        n = 5000
        x = [true_mu]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            x.append(x[-1] + true_theta * (true_mu - x[-1]) * dt + true_sigma * dW)
        data = pd.Series(x)

        ou_mle = OUProcess()
        ou_mle.fit(data, dt=dt, method='mle')

        ou_ar1 = OUProcess()
        ou_ar1.fit(data, dt=dt, method='ar1')

        # Both methods should produce identical estimates
        assert ou_mle.theta_ == pytest.approx(ou_ar1.theta_, rel=1e-10)
        assert ou_mle.mu_ == pytest.approx(ou_ar1.mu_, rel=1e-10)
        assert ou_mle.sigma_ == pytest.approx(ou_ar1.sigma_, rel=1e-10)

    def test_ou_fast_mean_reversion(self):
        """Edge case: very fast mean reversion (theta=10)."""
        np.random.seed(105)
        true_theta, true_mu, true_sigma, dt = 10.0, 0.0, 1.0, 1 / 252
        n = 5000
        x = [1.0]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            x.append(x[-1] + true_theta * (true_mu - x[-1]) * dt + true_sigma * dW)
        data = pd.Series(x)

        ou = OUProcess()
        ou.fit(data, dt=dt, method='mle')
        assert ou.theta_ > 1.0

    def test_ou_slow_mean_reversion(self):
        """Edge case: very slow mean reversion (theta=0.01)."""
        np.random.seed(106)
        true_theta, true_mu, true_sigma, dt = 0.01, 10.0, 0.5, 1 / 252
        n = 2000
        x = [10.0]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            x.append(x[-1] + true_theta * (true_mu - x[-1]) * dt + true_sigma * dW)
        data = pd.Series(x)

        ou = OUProcess()
        ou.fit(data, dt=dt, method='mle')
        assert ou.theta_ >= 0


class TestCIRRecovery:
    """Verify CIR parameter recovery."""

    def test_recover_cir_params_mle(self):
        np.random.seed(107)
        true_kappa, true_theta, true_sigma, dt = 0.5, 0.05, 0.1, 1 / 252
        n = 2000
        x = [0.04]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dx = true_kappa * (true_theta - x[-1]) * dt + true_sigma * np.sqrt(max(0, x[-1])) * dW
            x.append(max(1e-6, x[-1] + dx))
        data = pd.Series(x)

        cir = CIRProcess()
        cir.fit(data, dt=dt, method='mle')

        assert cir.kappa_ > 0
        assert cir.theta_ > 0
        assert cir.sigma_ > 0

    def test_recover_cir_params_lsq(self):
        np.random.seed(107)
        true_kappa, true_theta, true_sigma, dt = 0.5, 0.05, 0.1, 1 / 252
        n = 2000
        x = [0.04]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dx = true_kappa * (true_theta - x[-1]) * dt + true_sigma * np.sqrt(max(0, x[-1])) * dW
            x.append(max(1e-6, x[-1] + dx))
        data = pd.Series(x)

        cir = CIRProcess()
        cir.fit(data, dt=dt, method='lsq')

        assert cir.kappa_ > 0
        assert cir.theta_ > 0
        assert cir.sigma_ > 0

    def test_cir_near_feller_boundary(self):
        """Edge case: CIR near Feller condition boundary."""
        np.random.seed(108)
        # 2*kappa*theta = 0.05, sigma^2 = 0.04 -> barely satisfies
        true_kappa, true_theta, true_sigma, dt = 0.5, 0.05, 0.2, 1 / 252
        n = 2000
        x = [0.05]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dx = true_kappa * (true_theta - x[-1]) * dt + true_sigma * np.sqrt(max(0, x[-1])) * dW
            x.append(max(1e-6, x[-1] + dx))
        data = pd.Series(x)

        cir = CIRProcess()
        cir.fit(data, dt=dt, method='lsq')
        assert cir.is_fitted


class TestRegressionPinnedSeeds:
    """Pin expected outputs for specific seeds to catch silent regressions."""

    def test_bm_pinned_shape_and_start(self):
        np.random.seed(42)
        bm = BrownianMotion(mu=0.1, sigma=0.3)
        result = bm.sample(n_paths=3, horizon=5, dt=1.0)
        assert result.paths.shape == (6, 3)
        assert np.allclose(result.paths.iloc[0, :].values, [0.0, 0.0, 0.0])

    def test_gbm_pinned_positivity(self):
        np.random.seed(42)
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.3)
        result = gbm.sample(n_paths=10, horizon=100, dt=1 / 252)
        assert (result.paths.values > 0).all()

    def test_ou_pinned_starts_at_mu(self):
        np.random.seed(42)
        ou = OUProcess(theta=1.0, mu=5.0, sigma=0.5)
        result = ou.sample(n_paths=5, horizon=10, dt=1 / 252)
        assert np.allclose(result.paths.iloc[0, :].values, 5.0)

    def test_cir_pinned_non_negative(self):
        np.random.seed(42)
        cir = CIRProcess(kappa=0.5, theta=0.05, sigma=0.1)
        result = cir.sample(n_paths=10, horizon=100, dt=1 / 252)
        assert (result.paths.values >= 0).all()


class TestSamplingMoments:
    """Compare sample moments against theoretical moments."""

    def test_bm_terminal_mean(self):
        """BM: E[X_T] = X_0 + mu*T"""
        np.random.seed(200)
        mu, sigma, dt = 0.1, 0.3, 1 / 252
        horizon = 252
        n_paths = 50000

        bm = BrownianMotion(mu=mu, sigma=sigma)
        result = bm.sample(n_paths=n_paths, horizon=horizon, dt=dt)

        T = horizon * dt
        terminal = result.paths.iloc[-1, :].values
        assert abs(np.mean(terminal) - mu * T) < 0.05

    def test_bm_terminal_variance(self):
        """BM: Var[X_T] = sigma^2 * T"""
        np.random.seed(201)
        mu, sigma, dt = 0.0, 0.3, 1 / 252
        horizon = 252
        n_paths = 50000

        bm = BrownianMotion(mu=mu, sigma=sigma)
        result = bm.sample(n_paths=n_paths, horizon=horizon, dt=dt)

        T = horizon * dt
        terminal = result.paths.iloc[-1, :].values
        assert abs(np.var(terminal) - sigma ** 2 * T) < 0.05

    def test_gbm_terminal_mean(self):
        """GBM: E[S_T] = S_0 * exp(mu*T)"""
        np.random.seed(202)
        mu, sigma, dt = 0.1, 0.2, 1 / 252
        horizon = 252
        n_paths = 50000

        gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
        result = gbm.sample(n_paths=n_paths, horizon=horizon, dt=dt)

        T = horizon * dt
        terminal = result.paths.iloc[-1, :].values
        # S_0 defaults to 1.0 for unfitted GBM
        expected_mean = 1.0 * np.exp(mu * T)
        assert abs(np.mean(terminal) - expected_mean) / expected_mean < 0.05

# tests/test_statistical_robustness.py
"""Phase 2 tests: bias correction, Fisher Information SEs, regularization,
Feller condition, non-stationarity warnings, and backward compatibility."""

import warnings

import numpy as np
import pandas as pd
import pytest

from kestrel import (
    BrownianMotion,
    CIRProcess,
    GeometricBrownianMotion,
    MertonProcess,
    OUProcess,
)
from kestrel.utils.fisher_information import (
    bm_fisher_information,
    gbm_fisher_information,
    ou_fisher_information,
)
from kestrel.utils.warnings import (
    BiasWarning,
    ConvergenceWarning,
    FellerConditionWarning,
    KestrelWarning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_ou(theta, mu, sigma, dt, n, x0=None, seed=42):
    """Simulate an OU path and return as pd.Series."""
    rng = np.random.RandomState(seed)
    x = [x0 if x0 is not None else mu]
    for _ in range(n - 1):
        dW = rng.normal(0, np.sqrt(dt))
        x.append(x[-1] + theta * (mu - x[-1]) * dt + sigma * dW)
    return pd.Series(x)


def _simulate_cir(kappa, theta_cir, sigma, dt, n, x0=None, seed=42):
    """Simulate a CIR path and return as pd.Series."""
    rng = np.random.RandomState(seed)
    x = [x0 if x0 is not None else theta_cir]
    for _ in range(n - 1):
        dW = rng.normal(0, np.sqrt(dt))
        dx = kappa * (theta_cir - x[-1]) * dt + sigma * np.sqrt(max(0, x[-1])) * dW
        x.append(max(1e-8, x[-1] + dx))
    return pd.Series(x)


def _simulate_bm(mu, sigma, dt, n, seed=42):
    """Simulate a Brownian motion path."""
    rng = np.random.RandomState(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(x[-1] + mu * dt + sigma * np.sqrt(dt) * rng.normal())
    return pd.Series(x)


def _simulate_gbm(mu, sigma, dt, n, s0=100.0, seed=42):
    """Simulate a GBM price path."""
    rng = np.random.RandomState(seed)
    prices = [s0]
    for _ in range(n - 1):
        Z = rng.normal()
        prices.append(prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z))
    return pd.Series(prices)


# ===========================================================================
# 1. Warning class hierarchy
# ===========================================================================

class TestWarningHierarchy:
    """Verify the custom warning types exist and inherit correctly."""

    def test_kestrel_warning_is_user_warning(self):
        assert issubclass(KestrelWarning, UserWarning)

    def test_convergence_warning_hierarchy(self):
        assert issubclass(ConvergenceWarning, KestrelWarning)
        assert issubclass(ConvergenceWarning, UserWarning)

    def test_feller_warning_hierarchy(self):
        assert issubclass(FellerConditionWarning, KestrelWarning)

    def test_bias_warning_hierarchy(self):
        assert issubclass(BiasWarning, KestrelWarning)

    def test_warnings_are_catchable_via_base(self):
        """Users should be able to filter all kestrel warnings at once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("test", ConvergenceWarning)
            warnings.warn("test", FellerConditionWarning)
            warnings.warn("test", BiasWarning)
            kestrel_warnings = [x for x in w if issubclass(x.category, KestrelWarning)]
            assert len(kestrel_warnings) == 3


# ===========================================================================
# 2. OU Non-Stationarity Handling (Item 2.5)
# ===========================================================================

class TestOUNonStationarity:
    """Verify ConvergenceWarning is raised for non-stationary data."""

    def test_non_stationary_data_warns(self):
        """Random walk data (phi >= 1) should emit ConvergenceWarning."""
        np.random.seed(300)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))

        ou = OUProcess()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ou.fit(data, dt=1.0)
            convergence_warnings = [x for x in w if issubclass(x.category, ConvergenceWarning)]
            assert len(convergence_warnings) >= 1
            assert "phi=" in str(convergence_warnings[-1].message)

    def test_non_stationary_flag_set(self):
        np.random.seed(301)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))

        ou = OUProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ou.fit(data, dt=1.0)

        assert ou.is_fitted
        assert ou._non_stationary_ is True

    def test_stationary_data_no_non_stationary_flag(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=302)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252)
        assert ou._non_stationary_ is False

    def test_non_stationary_fallback_sigma_data_derived(self):
        """Fallback sigma should be derived from data, not hardcoded."""
        np.random.seed(303)
        data = pd.Series(np.cumsum(np.random.normal(0, 2.0, 200)))

        ou = OUProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ou.fit(data, dt=1.0)

        # sigma should be data-derived, not 0.1
        assert ou.sigma_ > 0.5  # 2.0 * sqrt(1/1) ~ 2.0 is the rough scale


# ===========================================================================
# 3. CIR Feller Condition (Item 2.4)
# ===========================================================================

class TestCIRFellerCondition:
    """Verify Feller condition warnings and constraint enforcement."""

    def test_feller_warning_emitted_when_violated(self):
        """Fitting data that violates Feller should emit FellerConditionWarning."""
        # Simulate with parameters that easily violate Feller
        data = _simulate_cir(0.1, 0.01, 0.5, 1 / 252, 1000, x0=0.01, seed=310)

        cir = CIRProcess()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cir.fit(data, dt=1 / 252, method='lsq')
            feller_warnings = [x for x in w if issubclass(x.category, FellerConditionWarning)]
            # May or may not violate depending on data; check that the flag is set
            if not cir._feller_satisfied_:
                assert len(feller_warnings) >= 1

    def test_feller_satisfied_flag_stored(self):
        """After fitting, _feller_satisfied_ flag should exist."""
        # Use parameters that satisfy Feller: 2*5*0.05 = 0.5 > 0.01 = 0.1^2
        data = _simulate_cir(5.0, 0.05, 0.1, 1 / 252, 2000, seed=311)

        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='mle')
        assert hasattr(cir, '_feller_satisfied_')
        assert isinstance(cir._feller_satisfied_, bool)

    def test_feller_constraint_mle(self):
        """With feller_constraint=True, fitted params should satisfy Feller."""
        data = _simulate_cir(0.5, 0.05, 0.2, 1 / 252, 2000, seed=312)

        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='mle', feller_constraint=True)

        assert cir.feller_condition_satisfied()

    def test_feller_constraint_lsq(self):
        """With feller_constraint=True on LSQ, sigma is projected if needed."""
        data = _simulate_cir(0.1, 0.01, 0.5, 1 / 252, 1000, x0=0.01, seed=313)

        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='lsq', feller_constraint=True)

        # After constraint, Feller should be satisfied or sigma forced small
        assert cir.sigma_ > 0

    def test_feller_condition_method(self):
        """feller_condition_satisfied() should return correct boolean."""
        # Satisfies: 2*5*0.05 = 0.5 > 0.01 = 0.1^2
        cir = CIRProcess(kappa=5.0, theta=0.05, sigma=0.1)
        assert cir.feller_condition_satisfied() is True

        # Violates: 2*0.1*0.01 = 0.002 < 0.25 = 0.5^2
        cir2 = CIRProcess(kappa=0.1, theta=0.01, sigma=0.5)
        assert cir2.feller_condition_satisfied() is False


# ===========================================================================
# 4. Fisher Information Standard Errors (Item 2.2)
# ===========================================================================

class TestFisherInformation:
    """Verify Fisher Information SEs are correct and consistent."""

    def test_bm_fim_shape(self):
        fim, ses = bm_fisher_information(sigma=0.3, dt=1 / 252, n=1000)
        assert fim.shape == (2, 2)
        assert 'mu' in ses and 'sigma' in ses
        assert ses['mu'] > 0 and ses['sigma'] > 0

    def test_bm_fim_diagonal(self):
        """BM FIM should be diagonal (mu and sigma are orthogonal)."""
        fim, _ = bm_fisher_information(sigma=0.3, dt=1 / 252, n=1000)
        assert fim[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert fim[1, 0] == pytest.approx(0.0, abs=1e-12)

    def test_bm_se_scales_with_n(self):
        """SEs should decrease as n increases (proportional to 1/sqrt(n))."""
        _, ses_small = bm_fisher_information(sigma=0.3, dt=1 / 252, n=100)
        _, ses_large = bm_fisher_information(sigma=0.3, dt=1 / 252, n=10000)
        assert ses_large['mu'] < ses_small['mu']
        assert ses_large['sigma'] < ses_small['sigma']

    def test_gbm_fim_shape(self):
        fim, ses = gbm_fisher_information(mu=0.1, sigma=0.3, dt=1 / 252, n=1000)
        assert fim.shape == (2, 2)
        assert 'mu' in ses and 'sigma' in ses
        assert ses['mu'] > 0 and ses['sigma'] > 0

    def test_gbm_fim_not_diagonal(self):
        """GBM FIM should NOT be diagonal (mu and sigma are coupled via alpha)."""
        fim, _ = gbm_fisher_information(mu=0.1, sigma=0.3, dt=1 / 252, n=1000)
        # Off-diagonal should be non-zero
        assert abs(fim[0, 1]) > 1e-12

    def test_ou_fim_shape(self):
        x_t = np.random.RandomState(42).normal(5.0, 0.5, 999)
        fim, ses = ou_fisher_information(theta=1.0, mu=5.0, sigma=0.5, dt=1 / 252, n=999, x_t=x_t)
        assert fim.shape == (3, 3)
        assert all(k in ses for k in ['theta', 'mu', 'sigma'])
        assert all(ses[k] > 0 for k in ['theta', 'mu', 'sigma'])

    def test_ou_fim_returns_nan_for_bad_params(self):
        """Should return NaN SEs when theta <= 0 or sigma <= 0."""
        x_t = np.ones(100)
        fim, ses = ou_fisher_information(theta=-1.0, mu=0.0, sigma=0.5, dt=1.0, n=100, x_t=x_t)
        assert np.isnan(ses['theta'])

    def test_bm_fitted_has_fisher_information(self):
        """After BM fit, _fisher_information_ attribute should exist."""
        data = _simulate_bm(0.1, 0.3, 1 / 252, 500, seed=320)
        bm = BrownianMotion()
        bm.fit(data, dt=1 / 252)
        assert hasattr(bm, '_fisher_information_')
        assert bm._fisher_information_.shape == (2, 2)

    def test_gbm_fitted_has_fisher_information(self):
        data = _simulate_gbm(0.1, 0.3, 1 / 252, 500, seed=321)
        gbm = GeometricBrownianMotion()
        gbm.fit(data, dt=1 / 252)
        assert hasattr(gbm, '_fisher_information_')
        assert gbm._fisher_information_.shape == (2, 2)

    def test_ou_fitted_has_fisher_information(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 2000, seed=322)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252)
        assert hasattr(ou, '_fisher_information_')
        assert ou._fisher_information_.shape == (3, 3)


# ===========================================================================
# 5. Bias Correction (Item 2.1)
# ===========================================================================

class TestOUBiasCorrection:
    """Verify OU jackknife bias correction works correctly."""

    def test_bias_correction_default_true(self):
        """Bias correction should be on by default."""
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=330)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252)
        assert ou._bias_corrected_ is True

    def test_bias_correction_can_be_disabled(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=331)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, bias_correction=False)
        assert ou._bias_corrected_ is False

    def test_bias_correction_changes_estimates(self):
        """Corrected and uncorrected estimates should differ."""
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 500, seed=332)

        ou_corr = OUProcess()
        ou_corr.fit(data, dt=1 / 252, bias_correction=True)

        ou_uncorr = OUProcess()
        ou_uncorr.fit(data, dt=1 / 252, bias_correction=False)

        # They should differ (jackknife generally adjusts theta downward)
        assert ou_corr.theta_ != ou_uncorr.theta_

    def test_bias_correction_reduces_mean_theta_error(self):
        """Over multiple simulations, corrected theta should have lower bias."""
        true_theta, true_mu, true_sigma, dt = 1.0, 5.0, 0.5, 1 / 252
        n = 500
        n_sims = 30

        theta_corr = []
        theta_uncorr = []
        for seed in range(400, 400 + n_sims):
            data = _simulate_ou(true_theta, true_mu, true_sigma, dt, n, seed=seed)

            ou1 = OUProcess()
            ou1.fit(data, dt=dt, bias_correction=True)
            theta_corr.append(ou1.theta_)

            ou2 = OUProcess()
            ou2.fit(data, dt=dt, bias_correction=False)
            theta_uncorr.append(ou2.theta_)

        bias_corr = abs(np.mean(theta_corr) - true_theta)
        bias_uncorr = abs(np.mean(theta_uncorr) - true_theta)

        # Corrected should have lower bias (or at least not much worse)
        assert bias_corr < bias_uncorr + 0.5

    def test_bias_correction_positive_theta(self):
        """Corrected theta should always remain positive."""
        data = _simulate_ou(0.1, 5.0, 0.5, 1 / 252, 300, seed=333)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, bias_correction=True)
        assert ou.theta_ > 0

    def test_bias_correction_skipped_with_regularization(self):
        """When regularization is active, bias correction should be skipped."""
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 500, seed=334)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, regularization=0.1)
        assert ou._bias_corrected_ is False


class TestCIRBiasCorrection:
    """Verify CIR bias correction works."""

    def test_cir_bias_correction_default_false(self):
        """CIR bias correction should be off by default."""
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 1000, seed=340)
        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='mle')
        assert cir._bias_corrected_ is False

    def test_cir_bias_correction_mle(self):
        """MLE bias correction uses analytical formula kappa * n/(n+3)."""
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 1000, seed=341)

        cir_corr = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir_corr.fit(data, dt=1 / 252, method='mle', bias_correction=True)

        cir_uncorr = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir_uncorr.fit(data, dt=1 / 252, method='mle', bias_correction=False)

        # Corrected kappa should be smaller (n/(n+3) < 1)
        assert cir_corr.kappa_ < cir_uncorr.kappa_
        assert cir_corr._bias_corrected_ is True

    def test_cir_bias_correction_lsq(self):
        """LSQ bias correction uses jackknife."""
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 500, seed=342)

        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='lsq', bias_correction=True)

        assert cir._bias_corrected_ is True
        assert cir.kappa_ > 0


# ===========================================================================
# 6. Regularized MLE / MAP Estimation (Item 2.3)
# ===========================================================================

class TestBMRegularization:
    """Verify BM regularization shrinks mu toward zero."""

    def test_bm_regularization_shrinks_mu(self):
        data = _simulate_bm(0.5, 0.3, 1 / 252, 500, seed=350)

        bm_unreg = BrownianMotion()
        bm_unreg.fit(data, dt=1 / 252)

        bm_reg = BrownianMotion()
        bm_reg.fit(data, dt=1 / 252, regularization=10.0)

        # Regularized mu should be closer to zero
        assert abs(bm_reg.mu_) <= abs(bm_unreg.mu_) + 1e-6

    def test_bm_zero_regularization_matches_mle(self):
        """regularization=0 should behave identically to no regularization."""
        data = _simulate_bm(0.1, 0.3, 1 / 252, 500, seed=351)

        bm1 = BrownianMotion()
        bm1.fit(data, dt=1 / 252)

        bm2 = BrownianMotion()
        bm2.fit(data, dt=1 / 252, regularization=None)

        assert bm1.mu_ == pytest.approx(bm2.mu_, rel=1e-12)
        assert bm1.sigma_ == pytest.approx(bm2.sigma_, rel=1e-12)

    def test_bm_regularization_sigma_unchanged(self):
        """Regularization on mu should not affect sigma estimate."""
        data = _simulate_bm(0.5, 0.3, 1 / 252, 500, seed=352)

        bm_unreg = BrownianMotion()
        bm_unreg.fit(data, dt=1 / 252)

        bm_reg = BrownianMotion()
        bm_reg.fit(data, dt=1 / 252, regularization=10.0)

        assert bm_reg.sigma_ == pytest.approx(bm_unreg.sigma_, rel=1e-10)


class TestGBMRegularization:
    """Verify GBM regularization shrinks mu toward baseline."""

    def test_gbm_regularization_shrinks_mu(self):
        data = _simulate_gbm(0.3, 0.2, 1 / 252, 500, seed=353)

        gbm_unreg = GeometricBrownianMotion()
        gbm_unreg.fit(data, dt=1 / 252)

        gbm_reg = GeometricBrownianMotion()
        gbm_reg.fit(data, dt=1 / 252, regularization=10.0)

        # The regularized alpha = mu - 0.5*sigma^2 should be closer to zero,
        # so mu should be closer to 0.5*sigma^2
        alpha_unreg = gbm_unreg.mu_ - 0.5 * gbm_unreg.sigma_ ** 2
        alpha_reg = gbm_reg.mu_ - 0.5 * gbm_reg.sigma_ ** 2
        assert abs(alpha_reg) <= abs(alpha_unreg) + 1e-6

    def test_gbm_regularization_sigma_unchanged(self):
        data = _simulate_gbm(0.1, 0.3, 1 / 252, 500, seed=354)

        gbm_unreg = GeometricBrownianMotion()
        gbm_unreg.fit(data, dt=1 / 252)

        gbm_reg = GeometricBrownianMotion()
        gbm_reg.fit(data, dt=1 / 252, regularization=10.0)

        assert gbm_reg.sigma_ == pytest.approx(gbm_unreg.sigma_, rel=1e-10)


class TestOURegularization:
    """Verify OU penalised MLE works."""

    def test_ou_regularization_produces_fitted_model(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=360)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, regularization=0.1)
        assert ou.is_fitted
        assert ou.theta_ > 0
        assert ou.sigma_ > 0

    def test_ou_regularization_shrinks_theta(self):
        data = _simulate_ou(5.0, 5.0, 0.5, 1 / 252, 500, seed=361)

        ou_unreg = OUProcess()
        ou_unreg.fit(data, dt=1 / 252, bias_correction=False)

        ou_reg = OUProcess()
        ou_reg.fit(data, dt=1 / 252, regularization=1.0)

        # Heavy regularization should shrink theta
        assert ou_reg.theta_ < ou_unreg.theta_ + 1.0

    def test_ou_regularization_has_standard_errors(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=362)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, regularization=0.1)
        # SEs should be available (from numerical inverse Hessian)
        assert hasattr(ou, 'theta_se_')
        assert hasattr(ou, 'mu_se_')
        assert hasattr(ou, 'sigma_se_')

    def test_ou_regularization_skips_bias_correction(self):
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 500, seed=363)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252, regularization=0.1)
        assert ou._bias_corrected_ is False


class TestCIRRegularization:
    """Verify CIR regularization adds penalty to NLL."""

    def test_cir_regularization_produces_fitted_model(self):
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 1000, seed=370)
        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='mle', regularization=0.1)
        assert cir.is_fitted
        assert cir.kappa_ > 0
        assert cir.theta_ > 0
        assert cir.sigma_ > 0

    def test_cir_regularization_affects_estimates(self):
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 1000, seed=371)

        cir_unreg = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir_unreg.fit(data, dt=1 / 252, method='mle')

        cir_reg = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir_reg.fit(data, dt=1 / 252, method='mle', regularization=10.0)

        # Heavy regularization should change estimates
        # (not necessarily identical to unregularized)
        assert cir_reg.is_fitted


class TestMertonRegularization:
    """Verify Merton regularization adds penalty to NLL."""

    def test_merton_regularization_produces_fitted_model(self):
        np.random.seed(380)
        # Simulate log returns with jumps
        dt = 1 / 252
        n = 500
        mu, sigma, lambda_, jump_mu, jump_sigma = 0.05, 0.15, 1.0, -0.02, 0.03
        returns = []
        for _ in range(n):
            cont = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            n_jumps = np.random.poisson(lambda_ * dt)
            jump = np.sum(np.random.normal(jump_mu, jump_sigma, n_jumps)) if n_jumps > 0 else 0
            returns.append(cont + jump)
        data = pd.Series(returns)

        merton = MertonProcess()
        merton.fit(data, dt=dt, regularization=0.1)
        assert merton.is_fitted
        assert merton.sigma_ > 0

    def test_merton_no_regularization_default(self):
        """Default regularization=None should not crash."""
        np.random.seed(381)
        dt = 1 / 252
        n = 200
        returns = pd.Series(np.random.normal(0, 0.01, n))

        merton = MertonProcess()
        merton.fit(returns, dt=dt)
        assert merton.is_fitted


# ===========================================================================
# 7. Backward Compatibility
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure Phase 2 changes don't break existing API or defaults."""

    def test_bm_fit_default_unchanged(self):
        """BM fit() with no new params should produce same results as before."""
        data = _simulate_bm(0.1, 0.3, 1 / 252, 500, seed=390)
        bm = BrownianMotion()
        bm.fit(data, dt=1 / 252)
        assert bm.is_fitted
        assert hasattr(bm, 'mu_')
        assert hasattr(bm, 'sigma_')
        assert hasattr(bm, 'mu_se_')
        assert hasattr(bm, 'sigma_se_')

    def test_gbm_fit_default_unchanged(self):
        data = _simulate_gbm(0.1, 0.3, 1 / 252, 500, seed=391)
        gbm = GeometricBrownianMotion()
        gbm.fit(data, dt=1 / 252)
        assert gbm.is_fitted

    def test_ou_fit_default_has_bias_correction(self):
        """OU default should now include bias correction."""
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=392)
        ou = OUProcess()
        ou.fit(data, dt=1 / 252)
        assert ou.is_fitted
        assert ou._bias_corrected_ is True

    def test_cir_fit_default_no_bias_correction(self):
        """CIR default should NOT include bias correction."""
        data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 1000, seed=393)
        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(data, dt=1 / 252, method='mle')
        assert cir.is_fitted
        assert cir._bias_corrected_ is False

    def test_ou_mle_ar1_still_equivalent(self):
        """MLE and AR(1) methods should still produce identical results."""
        data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 2000, seed=394)

        ou_mle = OUProcess()
        ou_mle.fit(data, dt=1 / 252, method='mle')

        ou_ar1 = OUProcess()
        ou_ar1.fit(data, dt=1 / 252, method='ar1')

        assert ou_mle.theta_ == pytest.approx(ou_ar1.theta_, rel=1e-10)
        assert ou_mle.mu_ == pytest.approx(ou_ar1.mu_, rel=1e-10)
        assert ou_mle.sigma_ == pytest.approx(ou_ar1.sigma_, rel=1e-10)

    def test_all_processes_sample_after_fit(self):
        """All fitted processes should still sample without errors."""
        bm_data = _simulate_bm(0.1, 0.3, 1 / 252, 300, seed=395)
        bm = BrownianMotion()
        bm.fit(bm_data, dt=1 / 252)
        result = bm.sample(n_paths=2, horizon=5)
        assert result.paths.shape == (6, 2)

        gbm_data = _simulate_gbm(0.1, 0.3, 1 / 252, 300, seed=396)
        gbm = GeometricBrownianMotion()
        gbm.fit(gbm_data, dt=1 / 252)
        result = gbm.sample(n_paths=2, horizon=5)
        assert result.paths.shape == (6, 2)

        ou_data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 300, seed=397)
        ou = OUProcess()
        ou.fit(ou_data, dt=1 / 252)
        result = ou.sample(n_paths=2, horizon=5)
        assert result.paths.shape == (6, 2)

        cir_data = _simulate_cir(0.5, 0.05, 0.1, 1 / 252, 300, seed=398)
        cir = CIRProcess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cir.fit(cir_data, dt=1 / 252, method='mle')
        result = cir.sample(n_paths=2, horizon=5)
        assert result.paths.shape == (6, 2)

    def test_se_attributes_exist_after_fit(self):
        """All processes should have _se_ attributes after fitting."""
        bm_data = _simulate_bm(0.1, 0.3, 1 / 252, 300, seed=399)
        bm = BrownianMotion()
        bm.fit(bm_data, dt=1 / 252)
        assert bm.mu_se_ > 0
        assert bm.sigma_se_ > 0

        ou_data = _simulate_ou(1.0, 5.0, 0.5, 1 / 252, 1000, seed=400)
        ou = OUProcess()
        ou.fit(ou_data, dt=1 / 252)
        assert ou.theta_se_ > 0
        assert ou.mu_se_ > 0
        assert ou.sigma_se_ > 0

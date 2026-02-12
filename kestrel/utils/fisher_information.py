# kestrel/utils/fisher_information.py
"""Closed-form Fisher Information Matrices for stochastic processes."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def bm_fisher_information(
    sigma: float, dt: float, n: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fisher Information Matrix for Brownian Motion parametrised as (mu, sigma).

    The log-likelihood for n increments dx_i ~ N(mu*dt, sigma^2*dt) gives:
        I = diag(n*dt/sigma^2, 2*n/sigma^2)

    Args:
        sigma: Estimated volatility.
        dt: Time step.
        n: Number of increments (len(data) - 1).

    Returns:
        (FIM, param_ses): 2x2 Fisher Information Matrix and dict of standard errors.
    """
    fim = np.array([
        [n * dt / sigma ** 2, 0.0],
        [0.0, 2 * n / sigma ** 2],
    ])
    cov = np.linalg.inv(fim)
    param_ses = {
        'mu': float(np.sqrt(cov[0, 0])),
        'sigma': float(np.sqrt(cov[1, 1])),
    }
    return fim, param_ses


def gbm_fisher_information(
    mu: float, sigma: float, dt: float, n: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fisher Information Matrix for GBM parametrised as (mu, sigma).

    GBM log-returns r_i = log(S_{i+1}/S_i) ~ N(alpha*dt, sigma^2*dt)
    where alpha = mu - 0.5*sigma^2.

    The FIM in (alpha, sigma) space is diagonal:
        I_alpha = diag(n*dt/sigma^2, 2*n/sigma^2)

    We transform to (mu, sigma) via Jacobian d(alpha,sigma)/d(mu,sigma).

    Args:
        mu: Estimated drift.
        sigma: Estimated volatility.
        dt: Time step.
        n: Number of log-returns (len(data) - 1).

    Returns:
        (FIM, param_ses): 2x2 Fisher Information Matrix and dict of standard errors.
    """
    # FIM in (alpha, sigma) space — diagonal
    fim_alpha = np.array([
        [n * dt / sigma ** 2, 0.0],
        [0.0, 2 * n / sigma ** 2],
    ])

    # Jacobian: d(alpha, sigma) / d(mu, sigma)
    # alpha = mu - 0.5*sigma^2  =>  dalpha/dmu = 1, dalpha/dsigma = -sigma
    # sigma = sigma              =>  dsigma/dmu = 0, dsigma/dsigma = 1
    J = np.array([
        [1.0, -sigma],
        [0.0, 1.0],
    ])

    # FIM in (mu, sigma) space: J^T @ FIM_alpha @ J
    fim = J.T @ fim_alpha @ J
    cov = np.linalg.inv(fim)
    param_ses = {
        'mu': float(np.sqrt(max(0.0, cov[0, 0]))),
        'sigma': float(np.sqrt(max(0.0, cov[1, 1]))),
    }
    return fim, param_ses


def ou_fisher_information(
    theta: float, mu: float, sigma: float, dt: float, n: int,
    x_t: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fisher Information Matrix for the OU process parametrised as (theta, mu, sigma).

    Computed via the Jacobian transformation from AR(1) space (c, phi, sigma_eps)
    to continuous-time OU parameters (theta, mu, sigma).

    The AR(1) representation is:
        X_{t+dt} = c + phi*X_t + eps,  eps ~ N(0, sigma_eps^2)
    where:
        phi = exp(-theta*dt)
        c = mu*(1 - phi)
        sigma_eps^2 = sigma^2*(1 - phi^2)/(2*theta)

    The FIM in AR(1) space has the standard OLS form for (c, phi) and a
    separate block for sigma_eps.

    Args:
        theta: Estimated mean-reversion speed.
        mu: Estimated long-term mean.
        sigma: Estimated volatility.
        dt: Time step.
        n: Number of transitions (len(data) - 1).
        x_t: Array of lagged values data[:-1].values.

    Returns:
        (FIM, param_ses): 3x3 Fisher Information Matrix and dict of standard errors.
    """
    if theta <= 0 or sigma <= 0:
        nan_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        return np.full((3, 3), np.nan), nan_ses

    phi = np.exp(-theta * dt)

    # sigma_eps^2 = sigma^2 * (1 - phi^2) / (2*theta)
    one_minus_phi_sq = 1.0 - phi ** 2
    if one_minus_phi_sq <= 0:
        nan_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        return np.full((3, 3), np.nan), nan_ses

    sigma_eps_sq = sigma ** 2 * one_minus_phi_sq / (2.0 * theta)
    sigma_eps = np.sqrt(sigma_eps_sq)

    # FIM in AR(1) space: (c, phi, sigma_eps)
    # For (c, phi): I = (1/sigma_eps^2) * X^T @ X  where X = [1, x_t]
    # For sigma_eps: I = 2*n / sigma_eps^2
    X_reg = np.vstack([np.ones(n), x_t]).T
    XtX = X_reg.T @ X_reg

    fim_ar1 = np.zeros((3, 3))
    fim_ar1[:2, :2] = XtX / sigma_eps_sq
    fim_ar1[2, 2] = 2.0 * n / sigma_eps_sq

    # Jacobian: d(c, phi, sigma_eps) / d(theta, mu, sigma)
    #
    # c = mu*(1 - phi)
    #   dc/dtheta = mu * dt * phi
    #   dc/dmu = 1 - phi
    #   dc/dsigma = 0
    #
    # phi = exp(-theta*dt)
    #   dphi/dtheta = -dt * phi
    #   dphi/dmu = 0
    #   dphi/dsigma = 0
    #
    # sigma_eps = sigma * sqrt(f)  where f = (1 - exp(-2*theta*dt)) / (2*theta)
    #   dsigma_eps/dsigma = sqrt(f)
    #   dsigma_eps/dmu = 0
    #   dsigma_eps/dtheta = sigma * df_dtheta / (2*sqrt(f))

    exp_2td = np.exp(-2.0 * theta * dt)
    f = (1.0 - exp_2td) / (2.0 * theta)

    if f <= 0:
        nan_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        return np.full((3, 3), np.nan), nan_ses

    # df/dtheta = dt*exp(-2*theta*dt)/theta - (1-exp(-2*theta*dt))/(2*theta^2)
    df_dtheta = (dt * exp_2td) / theta - (1.0 - exp_2td) / (2.0 * theta ** 2)
    dsigma_eps_dtheta = sigma * df_dtheta / (2.0 * np.sqrt(f))
    dsigma_eps_dsigma = np.sqrt(f)

    J = np.array([
        [mu * dt * phi, 1.0 - phi, 0.0],                       # dc/d(theta, mu, sigma)
        [-dt * phi, 0.0, 0.0],                                  # dphi/d(theta, mu, sigma)
        [dsigma_eps_dtheta, 0.0, dsigma_eps_dsigma],            # dsigma_eps/d(theta, mu, sigma)
    ])

    # FIM in (theta, mu, sigma) space: J^T @ FIM_ar1 @ J
    fim = J.T @ fim_ar1 @ J

    try:
        cov = np.linalg.inv(fim)
        param_ses = {
            'theta': float(np.sqrt(max(0.0, cov[0, 0]))),
            'mu': float(np.sqrt(max(0.0, cov[1, 1]))),
            'sigma': float(np.sqrt(max(0.0, cov[2, 2]))),
        }
    except np.linalg.LinAlgError:
        param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}

    return fim, param_ses

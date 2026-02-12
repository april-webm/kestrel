# kestrel/utils/kestrel_result.py
"""Simulation result container with plotting and analysis methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from typing import Any, Dict, List, Optional, Tuple


class KestrelResult:
    """
    A wrapper for simulation results, providing enhanced functionality.

    This class holds simulation paths, typically a pandas DataFrame,
    and offers convenience methods for plotting and analysis.
    It can also encapsulate estimation results, including parameters,
    standard errors, log-likelihood, AIC, BIC, and residuals,
    along with methods for residual diagnostics.
    """

    _paths: Optional[pd.DataFrame]
    _initial_value: Optional[float]
    _process_name: Optional[str]
    _params: Optional[Dict[str, float]]
    _param_ses: Optional[Dict[str, float]]
    _log_likelihood: Optional[float]
    _aic: Optional[float]
    _bic: Optional[float]
    _residuals: Optional[pd.Series]
    _n_obs: Optional[int]

    def __init__(
        self,
        paths: Optional[pd.DataFrame] = None,
        initial_value: Optional[float] = None,
        process_name: Optional[str] = None,
        params: Optional[Dict[str, float]] = None,
        param_ses: Optional[Dict[str, float]] = None,
        log_likelihood: Optional[float] = None,
        residuals: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None,
    ) -> None:
        if paths is not None and not isinstance(paths, pd.DataFrame):
            raise TypeError("`paths` must be a pandas DataFrame or None.")
        self._paths = paths
        self._initial_value = initial_value
        self._process_name = process_name
        self._params = params
        self._param_ses = param_ses
        self._log_likelihood = log_likelihood
        self._residuals = pd.Series(residuals) if residuals is not None else None
        self._n_obs = n_obs

        self._aic, self._bic = self._calculate_aic_bic(log_likelihood, n_obs, params)

    @staticmethod
    def _calculate_aic_bic(
        log_likelihood: Optional[float], n_obs: Optional[int], params: Optional[Dict[str, float]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Helper to calculate AIC and BIC."""
        if log_likelihood is None or n_obs is None or params is None:
            return None, None
        k = len(params) # Number of estimated parameters
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + np.log(n_obs) * k
        return aic, bic

    @property
    def paths(self) -> Optional[pd.DataFrame]:
        """
        Returns the underlying DataFrame of simulation paths.
        """
        return self._paths

    @property
    def process_name(self) -> Optional[str]:
        """Returns the name of the fitted process."""
        return self._process_name

    @property
    def params(self) -> Optional[Dict[str, float]]:
        """Returns a dictionary of estimated parameters."""
        return self._params

    @property
    def param_ses(self) -> Optional[Dict[str, float]]:
        """Returns a dictionary of estimated parameter standard errors."""
        return self._param_ses

    @property
    def log_likelihood(self) -> Optional[float]:
        """Returns the log-likelihood of the fit."""
        return self._log_likelihood

    @property
    def aic(self) -> Optional[float]:
        """Returns the Akaike Information Criterion (AIC)."""
        return self._aic

    @property
    def bic(self) -> Optional[float]:
        """Returns the Bayesian Information Criterion (BIC)."""
        return self._bic

    @property
    def residuals(self) -> Optional[pd.Series]:
        """Returns the residuals from the fit."""
        return self._residuals

    @property
    def n_obs(self) -> Optional[int]:
        """Returns the number of observations used in the fit."""
        return self._n_obs

    def plot(
        self,
        title: str = "Simulation Paths",
        xlabel: str = "Time Step",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (10, 6),
        **kwargs: Any,
    ) -> None:
        """
        Plots the simulation paths.

        Args:
            title (str): Plot title.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            figsize (tuple): Figure size (width, height).
            **kwargs: Additional keyword arguments passed to `DataFrame.plot()`.
        """
        if self._paths is None:
            print("No simulation paths available to plot.")
            return
        ax = self._paths.plot(figsize=figsize, title=title, legend=False, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()

    def mean_path(self) -> Optional[pd.Series]:
        """
        Calculates the mean path across all simulations.
        """
        if self._paths is None:
            return None
        return self._paths.mean(axis=1)

    def percentile_paths(self, percentiles: Optional[List[float]] = None) -> Optional[pd.DataFrame]:
        """
        Calculates percentile paths for the simulations.

        Args:
            percentiles (list): List of percentiles to calculate (e.g., [25, 50, 75]).

        Returns:
            pd.DataFrame: DataFrame with percentile paths.
        """
        if self._paths is None:
            return None
        if percentiles is None:
            percentiles = [25, 50, 75]
        return self._paths.quantile(np.array(percentiles) / 100, axis=1).T

    def ljung_box(self, lags: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
        """
        Performs the Ljung-Box test for autocorrelation in residuals.

        Args:
            lags (list): A list of lags to test. If None, uses [1, 5, 10].

        Returns:
            pd.DataFrame: Results of the Ljung-Box test (statistic, p-value).
        """
        if self._residuals is None:
            print("No residuals available for Ljung-Box test.")
            return None
        if lags is None:
            lags = [1, 5, 10]
        lb_test = acorr_ljungbox(self._residuals, lags=lags, return_df=True)
        return lb_test

    def normality_test(self) -> Optional[Dict[str, Any]]:
        """
        Performs the Shapiro-Wilk test for normality on residuals.

        Returns:
            dict: Results of the Shapiro-Wilk test (statistic, p-value).
        """
        if self._residuals is None:
            print("No residuals available for normality test.")
            return None
        statistic, p_value = stats.shapiro(self._residuals)
        return {"statistic": statistic, "p-value": p_value}

    def qq_plot(self, title: str = "Q-Q Plot of Residuals", figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Generates a Q-Q plot of the residuals.

        Args:
            title (str): Title of the Q-Q plot.
            figsize (tuple): Figure size (width, height).
        """
        if self._residuals is None:
            print("No residuals available for Q-Q plot.")
            return
        fig, ax = plt.subplots(figsize=figsize)
        sm.qqplot(self._residuals, line='s', ax=ax)
        ax.set_title(title)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()

    def __repr__(self) -> str:
        return (
            f"KestrelResult(process_name={self._process_name}, "
            f"paths_shape={self._paths.shape if self._paths is not None else 'N/A'}, "
            f"initial_value={self._initial_value}, "
            f"log_likelihood={self._log_likelihood:.2f} if self._log_likelihood is not None else 'N/A', "
            f"aic={self._aic:.2f} if self._aic is not None else 'N/A', "
            f"bic={self._bic:.2f} if self._bic is not None else 'N/A')"
        )

    def __str__(self) -> str:
        return self.__repr__()
# kestrel/base.py
"""Abstract base class for all stochastic processes."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from kestrel.utils.kestrel_result import KestrelResult


class StochasticProcess(ABC):
    """
    Abstract Base Class for Kestrel's stochastic processes.
    Defines common interface for parameter fitting and path simulation.
    """

    _fitted: bool
    _params: Dict[str, Any]
    _last_data_point: float
    _dt_: float
    _freq_: Optional[str]

    def __init__(self) -> None:
        self._fitted = False
        self._params = {}

    @abstractmethod
    def fit(self, data: pd.Series, dt: Optional[float] = None, **kwargs: Any) -> None:
        """
        Estimates process parameters from time-series data.

        Args:
            data (pd.Series): Time-series data for model fitting.
            dt (float, optional): Time step between observations.
                                   If None, inferred from data; defaults to 1.0.
        """
        pass

    @abstractmethod
    def sample(self, n_paths: int = 1, horizon: int = 1, dt: Optional[float] = None) -> KestrelResult:
        """
        Simulates future process paths.

        Args:
            n_paths (int): Number of simulation paths to generate.
            horizon (int): Number of future time steps to simulate.
            dt (float, optional): Simulation time step.
                                   If None, uses fitted dt; defaults to 1.0.

        Returns:
            KestrelResult: Simulation results.
        """
        pass

    def _infer_dt(self, data: pd.Series) -> float:
        """Infers dt from DatetimeIndex or defaults to 1.0."""
        if isinstance(data.index, pd.DatetimeIndex):
            if len(data.index) < 2:
                return 1.0

            inferred_timedelta = data.index[1] - data.index[0]
            current_freq = pd.infer_freq(data.index)
            if current_freq is None:
                current_freq = 'B'

            if current_freq in ['B', 'C', 'D']:
                dt = inferred_timedelta / pd.Timedelta(days=252.0)
            elif current_freq.startswith('W'):
                dt = inferred_timedelta / pd.Timedelta(weeks=52)
            elif current_freq in ['M', 'MS', 'BM', 'BMS']:
                dt = inferred_timedelta / pd.Timedelta(days=365 / 12)
            elif current_freq in ['Q', 'QS', 'BQ', 'BQS']:
                dt = inferred_timedelta / pd.Timedelta(days=365 / 4)
            elif current_freq in ['A', 'AS', 'BA', 'BAS', 'Y', 'YS', 'BY', 'BYS']:
                dt = inferred_timedelta / pd.Timedelta(days=365)
            else:
                dt = inferred_timedelta.total_seconds() / (365 * 24 * 3600)

            return max(dt, 1e-10)
        return 1.0

    def _set_params(
        self,
        last_data_point: Optional[float] = None,
        dt: Optional[float] = None,
        freq: Optional[str] = None,
        param_ses: Optional[Dict[str, float]] = None,
        **kwargs: float,
    ) -> None:
        """
        Sets estimated parameters, their standard errors, and marks model as fitted.
        """
        for k, v in kwargs.items():
            attr_name = k if k.endswith('_') else f"{k}_"
            setattr(self, attr_name, v)
            self._params[k] = v
        self._fitted = True
        if last_data_point is not None:
            self._last_data_point = last_data_point
        if dt is not None:
            self._dt_ = dt
        if freq is not None:
            self._freq_ = freq
        if param_ses is not None:
            for k, v in param_ses.items():
                base_key = k.rstrip('_')
                setattr(self, f"{base_key}_se_", v)
                self._params[f"{base_key}_se"] = v

    @property
    def is_fitted(self) -> bool:
        """
        Returns True if model fitted, False otherwise.
        """
        return self._fitted

    @property
    def params(self) -> Dict[str, Any]:
        """
        Returns dictionary of estimated parameters.
        """
        return self._params

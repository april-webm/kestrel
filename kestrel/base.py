# kestrel/base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from kestrel.utils.kestrel_result import KestrelResult
from typing import Optional

class StochasticProcess(ABC):
    """
    Abstract Base Class for Kestrel's stochastic processes.
    Defines common interface for parameter fitting and path simulation.
    """

    def __init__(self):
        self._fitted = False
        self._params = {} # Stores estimated parameters

    @abstractmethod
    def fit(self, data: pd.Series, dt: float = None):
        """
        Estimates process parameters from time-series data.

        Args:
            data (pd.Series): Time-series data for model fitting.
            dt (float, optional): Time step between observations.
                                   If None, inferred from data; defaults to 1.0.
        """
        pass

    @abstractmethod
    def sample(self, n_paths: int = 1, horizon: int = 1, dt: float = None) -> KestrelResult:
        """
        Simulates future process paths.

        Args:
            n_paths (int): Number of simulation paths to generate.
            horizon (int): Number of future time steps to simulate.
            dt (float, optional): Simulation time step.
                                   If None, uses fitted dt; defaults to 1.0.

        Returns:
            pd.DataFrame: DataFrame where each column is a simulated path.
        """
        pass

    def _set_params(self, last_data_point: float = None, dt: float = None, freq: str = None, param_ses: dict = None, **kwargs):
        """
        Sets estimated parameters, their standard errors, and marks model as fitted.
        """
        for k, v in kwargs.items():
            setattr(self, f"{k}_", v) # Underscore denotes estimated parameters
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
                setattr(self, f"{k}_se_", v) # Store standard errors
                self._params[f"{k}_se"] = v # Also add to params dictionary

    @property
    def is_fitted(self) -> bool:
        """
        Returns True if model fitted, False otherwise.
        """
        return self._fitted

    @property
    def params(self) -> dict:
        """
        Returns dictionary of estimated parameters.
        """
        return self._params

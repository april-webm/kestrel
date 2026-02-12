# kestrel/utils/kestrel_result.py
"""Simulation result container with plotting and analysis methods."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KestrelResult:
    """
    A wrapper for simulation results, providing enhanced functionality.

    This class holds simulation paths, typically a pandas DataFrame,
    and offers convenience methods for plotting and analysis.
    """

    _paths: pd.DataFrame
    _initial_value: Optional[float]

    def __init__(self, paths: pd.DataFrame, initial_value: Optional[float] = None) -> None:
        if not isinstance(paths, pd.DataFrame):
            raise TypeError("`paths` must be a pandas DataFrame.")
        self._paths = paths
        self._initial_value = initial_value

    @property
    def paths(self) -> pd.DataFrame:
        """
        Returns the underlying DataFrame of simulation paths.
        """
        return self._paths

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
        ax = self._paths.plot(figsize=figsize, title=title, legend=False, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()

    def mean_path(self) -> pd.Series:
        """
        Calculates the mean path across all simulations.
        """
        return self._paths.mean(axis=1)

    def percentile_paths(self, percentiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Calculates percentile paths for the simulations.

        Args:
            percentiles (list): List of percentiles to calculate (e.g., [25, 50, 75]).

        Returns:
            pd.DataFrame: DataFrame with percentile paths.
        """
        if percentiles is None:
            percentiles = [25, 50, 75]
        return self._paths.quantile(np.array(percentiles) / 100, axis=1).T

    def __repr__(self) -> str:
        return f"KestrelResult(paths_shape={self._paths.shape}, initial_value={self._initial_value})"

    def __str__(self) -> str:
        return self.__repr__()

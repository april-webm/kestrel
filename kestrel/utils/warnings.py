# kestrel/utils/warnings.py
"""Custom warning classes for Kestrel."""

from __future__ import annotations


class KestrelWarning(UserWarning):
    """Base class for all Kestrel warnings."""

    pass


class ConvergenceWarning(KestrelWarning):
    """Warning raised when estimation encounters convergence issues.

    Examples: AR(1) coefficient >= 1 in OU (non-stationarity),
    optimizer failure in CIR MLE.
    """

    pass


class FellerConditionWarning(KestrelWarning):
    """Warning raised when fitted CIR parameters violate the Feller condition.

    The Feller condition (2*kappa*theta > sigma^2) ensures the CIR process
    remains strictly positive. Violation means the process can reach zero.
    """

    pass


class BiasWarning(KestrelWarning):
    """Warning raised when estimates may suffer from significant small-sample bias.

    Raised when jackknife bias correction fails or the sample size is small
    enough that Hurwicz bias in mean-reversion speed estimates is likely material.
    """

    pass

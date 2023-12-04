from typing import Sequence, Union, Tuple

import numpy as np
from scipy.optimize import curve_fit


def fit_logical_error_rate_per_round(
        rounds: Sequence[int],
        logical_fidelities: Sequence[float],
        num_samples: Union[int, Sequence[int], None] = None,
        weighted_fit: bool = True,
) -> Tuple[float, float]:
    """Fit the logical error rate per round.

    Args:
        rounds: The number of error correction rounds.
        logical_fidelities: The logical fidelities with respect to each round.
        num_samples: The number of samples used to compute the std error of logical
            fidelities. Must be specified if weighted_fit is True. If a single value
            is specified, it is used for all rounds. If a sequence of values is
            specified, it must have the same length as rounds.
        weighted_fit: Whether to use the variance to weight the squared error fit.

    Returns:
        The fitted logical error rate per round epsilon and the SPAM fidelity A, i.e.
        F(t) = A*(1-2\epsilon)^t
    """
    if len(rounds) != len(logical_fidelities):
        raise ValueError("The number of rounds and logical fidelities must match.")
    if weighted_fit and num_samples is None:
        raise ValueError("num_samples must be specified if weighted_fit is True.")
    if weighted_fit and not isinstance(num_samples, int) and len(num_samples) != len(rounds):
        raise ValueError("num_samples must be a single value or a sequence of values "
                         "with the same length as rounds.")

    logical_fidelities = np.array(logical_fidelities)
    log10_fidelities = np.log10(logical_fidelities)
    if not weighted_fit:
        log10_sigma = None
    else:
        sigma = np.sqrt(logical_fidelities * (1 - logical_fidelities) / num_samples)
        log10_sigma = np.log10(sigma + logical_fidelities) - log10_fidelities
    popt, _ = curve_fit(_fit_func, rounds, log10_fidelities, sigma=log10_sigma, absolute_sigma=True)
    k, b = popt
    epsilon = (1 - 10**k) / 2
    return epsilon, 10**b


def _fit_func(t, k, b):
    """Fit logical fidelity at round t: F(t) = A*(1-2\epsilon)^t

    After log transform: log(F) = log(1-2\epsilon) * t + log(A)

    Then k = log(1-2\epsilon); b = log(A)
    """
    return k * t + b

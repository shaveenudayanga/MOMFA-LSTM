import math
import numpy as np


def levy_flight(size: int | tuple, lambda_: float = 1.5) -> np.ndarray:
    """Levy flight step via Mantegna's algorithm.

    Args:
        size:    Shape of the output step array (int or tuple).
        lambda_: Levy stability index. 1.5 per CONTEXT.md §3.3.

    Returns:
        Step vector of given shape. Heavy-tailed: occasional large jumps.
    """
    # Mantegna 1994 - numerically stable approximation of Levy stable distribution
    num = math.gamma(1 + lambda_) * math.sin(math.pi * lambda_ / 2)
    den = math.gamma((1 + lambda_) / 2) * lambda_ * (2 ** ((lambda_ - 1) / 2))
    sigma = (num / den) ** (1 / lambda_)

    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1,     size)
    return u / (np.abs(v) ** (1 / lambda_))

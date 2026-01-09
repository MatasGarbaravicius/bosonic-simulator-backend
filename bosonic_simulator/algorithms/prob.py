import numpy as np
import bosonic_simulator.coherent_state_tools as coherent_state_tools
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from scipy.linalg import solve  # type: ignore (comment for the IDE since it raises a warning although it shouldn't)
from numpy.typing import NDArray


def prob(
    gaussian_state_description: GaussianStateDescription,
    amplitude: NDArray[np.complex128],
    wires: list[int],
) -> np.float64:
    r"""
    Computes the probability density of observing a specific outcome in a heterodyne measurement.

    This implements the `prob` subroutine described in Section 3.4 of the reference paper.
    It takes as input a description $\Delta \in \text{Desc}_n$ of a Gaussian state and an
    outcome vector $\beta \in \mathbb{C}^k$, and outputs the probability of obtaining
    measurement outcome $\beta$ when performing a heterodyne measurement on a subset
    of $k$ modes.

    The algorithm runs in time $O(k^3)$, where $k$ is the number of measured modes (length of `wires`).

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The description of the state being measured.
    amplitude : NDArray[np.complex128]
        The measurement outcome vector $\beta \in \mathbb{C}^k$.
    wires : list[int]
        The list of mode indices (0-based) to be measured.

    Returns
    -------
    np.float64
        The probability density $\text{prob}(\Delta, \beta)$.
    """
    # precompute quantities and introduce shorter notation

    gamma = gaussian_state_description.covariance_matrix
    alpha = amplitude
    k = amplitude.size
    dhat_alpha = coherent_state_tools.displacement_vector(alpha)
    s_A = coherent_state_tools.displacement_vector(
        gaussian_state_description.amplitude[wires]
    )

    # prepare a mask for subsystem A (described by wires)

    mask_A = np.repeat(wires, 2)  # [a, b, c, ...] -> [a, a, b, b, c, c, ...]
    # start::step means start at index 'start' and pick elements every 'step' positions
    mask_A[0::2] = 2 * mask_A[0::2]
    mask_A[1::2] = 2 * mask_A[1::2] + 1

    # further precomputation

    gamma_A = gamma[np.ix_(mask_A, mask_A)]
    gamma_A_plus_I = gamma_A + np.eye(2 * k, dtype=gamma.dtype)

    # Final evaluation in log-space and via a linear solve (avoiding explicit matrix inversion),
    # both for numerical stability.

    exponent = -np.dot(dhat_alpha - s_A, solve(gamma_A_plus_I, dhat_alpha - s_A))
    (_, logabsdet) = np.linalg.slogdet((gamma_A_plus_I) / 2)  # log(|det(gamma_A + I)|)
    log_val = exponent - k * np.log(np.pi) - 0.5 * logabsdet
    return np.exp(log_val)

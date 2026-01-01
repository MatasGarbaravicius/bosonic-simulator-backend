import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.coherent_state import CoherentState
from scipy.linalg import solve  # type: ignore (comment for the IDE since it raises a warning although it shouldn't)


def prob(
    gaussian_state_description: GaussianStateDescription,
    amplitude: np.ndarray,
    wires: list[int],
):
    # precompute quantities and introduce shorter notation

    gamma = gaussian_state_description.covariance_matrix
    alpha = amplitude
    k = amplitude.size
    dhat_alpha = CoherentState.displacement_vector(alpha)
    s_A = CoherentState.displacement_vector(gaussian_state_description.amplitude[wires])

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

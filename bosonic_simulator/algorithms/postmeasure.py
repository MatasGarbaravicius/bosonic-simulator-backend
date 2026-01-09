import numpy as np
import bosonic_simulator.coherent_state_tools as coherent_state_tools
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.prob import prob
from bosonic_simulator.algorithms.overlaptriple import overlaptriple
from scipy.linalg import solve  # type: ignore (comment for the IDE since it raises a warning although it shouldn't)
from numpy.typing import NDArray


def postmeasure(
    gaussian_state_description: GaussianStateDescription,
    amplitude: NDArray[np.complex128],
    wires: list[int],
) -> GaussianStateDescription:
    r"""
    Computes the description of the post-measurement state after a heterodyne measurement.

    This implements the `postmeasure` algorithm described in Section 3.4 of the
    reference paper. Given an initial state description $\Delta$ and a measurement
    outcome vector $\beta \in \mathbb{C}^k$ associated with $k$ specific modes,
    it computes the description $\Delta'$ of the normalized post-measurement state.

    The algorithm runs in time $O(n^3)$.

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The description $\Delta$ of the state before measurement.
    amplitude : NDArray[np.complex128]
        The vector of measurement outcomes $\beta$. Its length must match the
        number of wires being measured.
    wires : list[int]
        The list of mode indices (0-based) on which the measurement is performed.

    Returns
    -------
    GaussianStateDescription
        The description $\Delta'$ of the post-measurement state.
    """
    # precompute quantities and introduce shorter notation

    gamma = gaussian_state_description.covariance_matrix
    alpha = gaussian_state_description.amplitude
    s = coherent_state_tools.displacement_vector(alpha)
    r = gaussian_state_description.overlap
    n = gaussian_state_description.number_of_modes
    beta = amplitude
    dhat_beta = coherent_state_tools.displacement_vector(beta)
    k = beta.size

    # prepare masks for the subsystems

    mask_A = np.zeros(n, dtype=bool)
    mask_A[wires] = True
    mask_B = np.ones(n, dtype=bool)
    mask_B[wires] = False
    mask_A = np.repeat(mask_A, 2)  # expand to phase-space
    mask_B = np.repeat(mask_B, 2)  # expand to phase-space

    # further precomputation

    s_A = s[mask_A]
    s_B = s[mask_B]
    gamma_A = gamma[np.ix_(mask_A, mask_A)]
    gamma_B = gamma[np.ix_(mask_B, mask_B)]
    gamma_AB = gamma[np.ix_(mask_A, mask_B)]
    gamma_A_plus_I = gamma_A + np.eye(2 * k, dtype=gamma_A.dtype)

    # compute gamma_prime
    # avoid explicit matrix inversion via linear solve (for numerical stability)

    gamma_prime_B = gamma_B - gamma_AB.T @ solve(gamma_A_plus_I, gamma_AB)
    gamma_prime = np.eye(2 * n, dtype=gamma.dtype)
    gamma_prime[np.ix_(mask_B, mask_B)] = gamma_prime_B

    # compute alpha_prime

    s_prime_A = dhat_beta
    s_prime_B = s_B + gamma_AB.T @ solve(gamma_A_plus_I, dhat_beta - s_A)
    s_prime = np.concatenate([s_prime_A, s_prime_B])
    alpha_prime = coherent_state_tools.displacement_vec_to_amplitude(s_prime)

    # compute r_prime

    gamma1, gamma2, gamma3 = gamma, gamma_prime, np.eye(2 * n, dtype=gamma.dtype)
    d1, d2, d3 = s, s_prime, s_prime
    lambda_ = alpha - alpha_prime
    u = np.exp(1j * np.dot(alpha_prime, np.conj(alpha)).imag) * r
    p = prob(gaussian_state_description, beta, wires)
    v = np.power(np.pi, k / 2) * np.sqrt(p)
    r_prime = np.conj(overlaptriple(gamma1, gamma2, gamma3, d1, d2, d3, u, v, lambda_))

    return GaussianStateDescription(gamma_prime, alpha_prime, r_prime)

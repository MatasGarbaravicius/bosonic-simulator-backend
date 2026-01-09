import numpy as np
from numpy.typing import NDArray
from scipy.linalg import (
    solve,
)  # type: ignore (comment for the IDE since it raises a warning although it shouldn't)

import bosonic_simulator.utils.coherent_state_utils as coherent_state_utils
from bosonic_simulator.algorithms.norm import exactnorm, fastnorm
from bosonic_simulator.algorithms.overlaps import overlaptriple
from bosonic_simulator.types.gaussian_state_description import GaussianStateDescription


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
    dhat_alpha = coherent_state_utils.displacement_vector(alpha)
    s_A = coherent_state_utils.displacement_vector(
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
    s = coherent_state_utils.displacement_vector(alpha)
    r = gaussian_state_description.overlap
    n = gaussian_state_description.number_of_modes
    beta = amplitude
    dhat_beta = coherent_state_utils.displacement_vector(beta)
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
    alpha_prime = coherent_state_utils.displacement_vec_to_amplitude(s_prime)

    # compute r_prime

    gamma1, gamma2, gamma3 = gamma, gamma_prime, np.eye(2 * n, dtype=gamma.dtype)
    d1, d2, d3 = s, s_prime, s_prime
    lambda_ = alpha - alpha_prime
    u = np.exp(1j * np.dot(alpha_prime, np.conj(alpha)).imag) * r
    p = prob(gaussian_state_description, beta, wires)
    v = np.power(np.pi, k / 2) * np.sqrt(p)
    r_prime = np.conj(overlaptriple(gamma1, gamma2, gamma3, d1, d2, d3, u, v, lambda_))

    return GaussianStateDescription(gamma_prime, alpha_prime, r_prime)


def measureprobexact(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    amplitude: NDArray[np.complex128],
    wires: list[int],
) -> np.float64:
    r"""
    Computes the exact probability density of observing a specific outcome in a heterodyne measurement on a superposition state.

    This implements the `measureprobexact` algorithm
    described in Section 4.3 (Lemma 4.4) of the reference paper. Given a superposition
    $|\Psi\rangle = \sum_{j=1}^{\chi} c_j |\psi_j\rangle$ of Gaussian states, it computes the probability
    density $p(\beta)$ for measuring the outcome $\beta$ on a specified subset of modes.

    **Runtime:**
    $O(\chi^2 n^3)$, where $\chi$ is the number of terms in the superposition and $n$
    is the number of modes.

    Parameters
    ----------
    superposition_terms : list[tuple[np.complex128, GaussianStateDescription]]
        A list of terms defining the superposition, where each term is a tuple
        $(c_j, \Delta_j)$ containing the complex coefficient $c_j$ and the
        description $\Delta_j$ of the Gaussian state $|\psi_j\rangle$.
    amplitude : NDArray[np.complex128]
        The measurement outcome vector $\beta \in \mathbb{C}^k$.
    wires : list[int]
        The list of mode indices (0-based) to be measured.

    Returns
    -------
    np.float64
        The exact probability density $p(\beta)$.
    """
    # precompute values and introduce shorter notation

    alpha = amplitude
    k = alpha.size

    # execute the algorithm

    exactnorm_input = []
    for c_j, psi_j in superposition_terms:

        pi_alpha_psi_j_norm_squared = (np.pi**k) * prob(psi_j, alpha, wires)
        c_prime_j = c_j * np.sqrt(pi_alpha_psi_j_norm_squared)

        psi_prime_j = postmeasure(psi_j, alpha, wires)

        exactnorm_input.append((c_prime_j, psi_prime_j))

    return (exactnorm(exactnorm_input) ** 2) / (np.pi**k)


def measureprobapproximate(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    amplitude: NDArray[np.complex128],
    wires: list[int],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64 | None = None,
) -> np.float64:
    r"""
    Estimates the probability density of observing a specific outcome in a heterodyne measurement of a normalized superposition of Gaussian states.

    This implements the `measureprobapproximate` algorithm described in Section 4.3 (Lemma 4.4) of the
    reference paper.

    Assume that the normalized post-measurement state $\Psi'_\beta = \Pi_\beta \Psi / \|\Pi_\beta \Psi\|$
    satisfies the energy bound:

    $$
    \langle \Psi'_\beta, H \Psi'_\beta \rangle \le E
    $$

    The output is a value $\overline{Y}$ that satisfies:

    $$
    \frac{\overline{Y}}{p_{\Psi}(\beta)} \in [1 - \epsilon, 1 + \epsilon]
    $$

    except with probability $p_f$.

    **Placeholder Energy Bound:**
    Lemma 4.4 requires an upper bound $E$ on the energy of the *post-measurement*
    state. If `energy_upper_bound` is not provided (is `None`), this implementation calculates
    a placeholder bound based on the energy of the *pre-measurement* superposition $\Psi$.
    This is calculated using the Cauchy-Schwarz inequality:

    $$
    E_{\Psi} = \langle \Psi | H | \Psi \rangle \le \left( \sum_j |c_j| \sqrt{\langle \psi_j | H | \psi_j \rangle} \right)^2
    $$

    **Runtime:**
    $O\left(\frac{\chi n^3 E}{p_f \epsilon^3}\right)$. This scales linearly with the number of terms $\chi$
    in the superposition, offering a significant speedup over the quadratic `measureprobexact`
    method for states with many terms.

    Parameters
    ----------
    superposition_terms : list[tuple[np.complex128, GaussianStateDescription]]
        A list of terms defining the superposition, where each term is a tuple
        $(c_j, \Delta_j)$.
    amplitude : NDArray[np.complex128]
        The measurement outcome vector $\beta \in \mathbb{C}^k$.
    wires : list[int]
        The list of mode indices (0-based) to be measured.
    multiplicative_error : np.float64
        The multiplicative error tolerance $\epsilon > 0$.
    max_failure_probability : np.float64
        The maximum allowable failure probability $p_f \in (0, 1)$.
    energy_upper_bound : np.float64 | None, optional
        An upper bound $E$ on the energy of the normalized post-measurement state.
        If None, a placeholder bound based on the pre-measurement state is used.

    Returns
    -------
    np.float64
        The estimated probability density $p(\beta)$.
    """
    if energy_upper_bound is None:
        energy_upper_bound = np.float64(
            np.square(
                np.sum(
                    [
                        np.abs(c) * np.sqrt(psi.energy())
                        for (c, psi) in superposition_terms
                    ]
                )
            )
        )  # bound energy of pre-measurement superposition with Cauchy-Schwarz

    # precompute values and introduce shorter notation

    alpha = amplitude
    k = alpha.size

    # execute the algorithm

    pi_alpha_psi = []
    for c_j, psi_j in superposition_terms:

        pi_alpha_psi_j_norm_squared = (np.pi**k) * prob(psi_j, alpha, wires)
        c_prime_j = c_j * np.sqrt(pi_alpha_psi_j_norm_squared)

        psi_prime_j = postmeasure(psi_j, alpha, wires)

        pi_alpha_psi.append((c_prime_j, psi_prime_j))

    norm_pi_alpha_psi = fastnorm(
        pi_alpha_psi,
        multiplicative_error,
        max_failure_probability,
        energy_upper_bound,
    )

    return (norm_pi_alpha_psi**2) / (np.pi**k)

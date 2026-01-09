import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.exactnorm import exactnorm
from bosonic_simulator.algorithms.prob import prob
from bosonic_simulator.algorithms.postmeasure import postmeasure
from numpy.typing import NDArray


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

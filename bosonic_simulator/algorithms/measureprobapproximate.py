import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.fastnorm import fastnorm
from bosonic_simulator.algorithms.prob import prob
from bosonic_simulator.algorithms.postmeasure import postmeasure
from numpy.typing import NDArray


def measureprobapproximate(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    amplitude: NDArray[np.complex128],
    wires: list[int],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64 | None = None,
) -> np.float64:
    r"""
    Estimates the probability density of observing a specific outcome in a heterodyne measurement.

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

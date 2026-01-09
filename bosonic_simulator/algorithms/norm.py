import numpy as np

from bosonic_simulator.algorithms.overlaps import overlap
from bosonic_simulator.types.gaussian_state_description import GaussianStateDescription


def exactnorm(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
) -> np.float64:
    r"""
    Computes the exact norm of a superposition of Gaussian states.

    This implements the `exactnorm` algorithm described in Section 4.1 (Lemma 4.1) of the
    reference paper. Given a superposition $|\Psi\rangle = \sum_{j=1}^{\chi} c_j |\psi_j\rangle$,
    it computes the norm $\|\Psi\|$ by evaluating:

    $$
    \|\Psi\| = \sqrt{\sum_{j,k=1}^{\chi} \overline{c_k} c_j \langle \psi_k, \psi_j \rangle}
    $$

    The algorithm computes $O(\chi^2)$ pairwise overlaps between the Gaussian states
    in the superposition.

    The runtime is $O(\chi^2 n^3)$, where $\chi$ is the number of terms in the
    superposition and $n$ is the number of modes.

    Parameters
    ----------
    superposition_terms : list[tuple[np.complex128, GaussianStateDescription]]
        A list of terms defining the superposition, where each term is a tuple
        $(c_j, \Delta_j)$ containing the complex coefficient $c_j$ and the
        description $\Delta_j$ of the Gaussian state $|\psi_j\rangle$.

    Returns
    -------
    np.float64
        The exact norm $\|\Psi\|$.
    """
    norm_squared = np.float64(
        np.sum(
            [
                np.conj(c_k) * c_j * overlap(psi_k, psi_j)
                for (c_k, psi_k) in superposition_terms
                for (c_j, psi_j) in superposition_terms
            ]
        )
    )
    return np.sqrt(np.real(norm_squared))


def fastnorm(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64,
) -> np.float64:
    r"""
    Estimates the norm of a superposition of Gaussian states.

    This implements **a modified version** of the `fastnorm_E` algorithm described in
    Section 4.2 of the reference paper. While the original algorithm computes an
    estimator $\overline{X}$ for the squared norm $\|\Psi\|^2$, this function returns
    $\sqrt{\overline{X}}$ to provide an estimate of the norm $\|\Psi\|$.

    According to the reference paper, the estimator $\overline{X}$ satisfies:

    $$
    (1-\epsilon)\|\Psi\|^2 \le \overline{X} \le (1+\epsilon)\|\Psi\|^2
    $$

    with probability at least $1 - p_f$.

    **Runtime:**
    $O\left(\frac{\chi n^3 E}{p_f \epsilon^3}\right)$. This scales linearly with the number of terms $\chi$
    in the superposition, offering a significant speedup over the quadratic `exactnorm`
    algorithm for states with many terms.

    Parameters
    ----------
    superposition_terms : list[tuple[np.complex128, GaussianStateDescription]]
        A list of terms defining the superposition, where each term is a tuple
        $(c_j, \Delta_j)$ containing the complex coefficient $c_j$ and the
        description $\Delta_j$ of the Gaussian state $|\psi_j\rangle$.
    multiplicative_error : np.float64
        The multiplicative error tolerance $\epsilon > 0$ for the squared norm estimate.
    max_failure_probability : np.float64
        The maximum allowable failure probability $p_f \in (0, 1)$.
    energy_upper_bound : np.float64
        An upper bound $E$ on the energy $\langle \Psi', H \Psi' \rangle$ of the
        normalized state $\Psi' = \Psi / \|\Psi\|$.

    Returns
    -------
    np.float64
        The estimated norm $\|\Psi\|$.
    """
    # introduce shorter notation

    n = superposition_terms[0][1].number_of_modes
    eps = multiplicative_error
    p_f = max_failure_probability
    capital_e = energy_upper_bound

    # execute the algorithm

    capital_r = np.sqrt(capital_e / eps)
    capital_l = np.ceil(4 * np.pi * capital_e / (p_f * (eps**3))).astype(int)
    sample_sum = np.float64(0)
    for _ in range(capital_l):
        # 1. sample alpha according to the Lebesgue measure on B_R(0)

        # 1.1 sample a random direction uniformly on the unit sphere in C^n by
        # normalizing a standard normal vector (the density is rotationally invariant)

        v = np.random.standard_normal(n) + 1j * np.random.standard_normal(n)
        v = v / np.linalg.norm(v)

        # 1.2 sample radius r so that the cumulative distribution fct is F(r) = (r/R)^{2n}

        r = capital_r * np.power(np.random.uniform(0, 1), 1 / (2 * n))

        alpha = r * v

        # 2. update sample_sum

        delta = GaussianStateDescription.coherent_state(alpha)
        sum_cjoj = np.sum([c * overlap(delta, psi) for (c, psi) in superposition_terms])
        sample_sum += (capital_r**2) * (np.abs(sum_cjoj) ** 2)

    return np.sqrt(sample_sum / capital_l)

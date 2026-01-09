import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.overlap import overlap


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

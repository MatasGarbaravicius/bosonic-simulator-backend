import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.measureprobexact import measureprobexact
from bosonic_simulator.gaussian_unitary_description import GaussianUnitaryDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary_inplace
from numpy.typing import NDArray


def simulateexactly(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    unitary_descriptions: list[GaussianUnitaryDescription],
    amplitude: NDArray[np.complex128],
    wires: list[int],
) -> np.float64:
    r"""
    Simulates the evolution and measurement of a superposition of Gaussian states.

    This implements the exact strong simulation routine summarized in Theorem 1.1 of the
    reference paper. It applies a sequence of Gaussian unitaries to a superposition state
    and computes the exact probability density of observing a specific heterodyne outcome.

    Given an initial superposition $|\Psi\rangle = \sum c_j |\psi_j\rangle$ and a
    sequence of unitaries $U_1, \dots, U_T$, it computes the probability density
    $p(\beta)$ for the evolved state:

    $$
    |\Psi_{\text{final}}\rangle = U_T \dots U_1 |\Psi\rangle
    $$

    **Runtime:**
    $O(T \chi n^3 + \chi^2 n^3)$, where $T$ is the number of unitaries, $\chi$ is the
    number of terms in the superposition, and $n$ is the number of modes.
    The term $T \chi n^3$ arises from the state evolution,
    while $\chi^2 n^3$ arises from the exact probability calculation which accounts for
    interference between all pairs of terms.

    Parameters
    ----------
    superposition_terms : list[tuple[np.complex128, GaussianStateDescription]]
        The initial superposition state, described by a list of terms $(c_j, \Delta_j)$.
    unitary_descriptions : list[GaussianUnitaryDescription]
        The sequence of Gaussian unitaries to apply.
    amplitude : NDArray[np.complex128]
        The measurement outcome vector $\beta \in \mathbb{C}^k$.
    wires : list[int]
        The list of mode indices (0-based) to be measured.

    Returns
    -------
    np.float64
        The exact probability density $p(\beta)$.
    """
    superposition_terms = copy.deepcopy(superposition_terms)
    for unitary_description in unitary_descriptions:
        for _, gaussian_state_description in superposition_terms:
            applyunitary_inplace(gaussian_state_description, unitary_description)

    return measureprobexact(superposition_terms, amplitude, wires)

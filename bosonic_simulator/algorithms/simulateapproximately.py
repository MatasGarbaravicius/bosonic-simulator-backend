import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.measureprobapproximate import measureprobapproximate
from bosonic_simulator.gaussian_unitary_description import GaussianUnitaryDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary_inplace
from numpy.typing import NDArray


def simulateapproximately(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    unitary_descriptions: list[GaussianUnitaryDescription],
    amplitude: NDArray[np.complex128],
    wires: list[int],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64 | None = None,
) -> np.float64:
    r"""
    Simulates the evolution and measurement of a normalized superposition of Gaussian states using an approximate method.

    This implements **a slightly modified version** of the approximate strong simulation routine summarized in Theorem 1.2
    of the reference paper. It applies a sequence of Gaussian unitaries to a superposition state
    and estimates the probability density of observing a specific heterodyne outcome.

    If the energy of the normalized post-measurement state is bounded by `energy_upper_bound`,
    the probability estimate $\overline{Y}$ satisfies:

    $$
    (1-\epsilon)p(\beta) \le \overline{Y} \le (1+\epsilon)p(\beta)
    $$

    with probability at least $1 - p_f$.

    **This implementation differs from the paper:**
    In the reference paper, a specific energy bound is used.
    In contrast:

    * This function allows the user to manually specify the `energy_upper_bound` $E$ for the post-measurement state.
    * If `energy_upper_bound` is `None`, it relies on the placeholder bound implemented in `measureprobapproximate`.
      This placeholder bound can be
      distinct from the one used in the paper.

    **Runtime:**
    $O\left(T \chi n^3 + \frac{\chi n^3 E}{p_f \epsilon^3}\right)$, where $T$ is the number of unitaries, $\chi$ is the number
    of superposition terms, and $n$ is the number of modes.
    Crucially, the dependence on $\chi$ is linear, making this method faster than the quadratic
    `simulateexactly` routine for large superpositions.

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
    multiplicative_error : np.float64
        The multiplicative error tolerance $\epsilon > 0$.
    max_failure_probability : np.float64
        The maximum allowable failure probability $p_f \in (0, 1)$.
    energy_upper_bound : np.float64 | None, optional
        An upper bound $E$ on the energy of the normalized post-measurement state.
        If None, the placeholder bound from `measureprobapproximate` is used.

    Returns
    -------
    np.float64
        The estimated probability density $p(\beta)$.
    """
    superposition_terms = copy.deepcopy(superposition_terms)
    for unitary_description in unitary_descriptions:
        for _, gaussian_state_description in superposition_terms:
            applyunitary_inplace(gaussian_state_description, unitary_description)

    return measureprobapproximate(
        superposition_terms,
        amplitude,
        wires,
        multiplicative_error,
        max_failure_probability,
        energy_upper_bound,
    )

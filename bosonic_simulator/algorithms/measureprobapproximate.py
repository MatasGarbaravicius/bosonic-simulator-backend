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

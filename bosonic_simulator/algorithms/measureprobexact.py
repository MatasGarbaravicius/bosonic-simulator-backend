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

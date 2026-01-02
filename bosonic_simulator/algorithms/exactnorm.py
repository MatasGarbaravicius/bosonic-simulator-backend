import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.overlap import overlap


def exactnorm(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
):
    norm_squared = np.sum(
        [
            np.conj(c_k) * c_j * overlap(psi_k, psi_j)
            for (c_k, psi_k) in superposition_terms
            for (c_j, psi_j) in superposition_terms
        ]
    )
    return np.sqrt(np.real(norm_squared))

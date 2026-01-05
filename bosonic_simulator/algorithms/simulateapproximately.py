import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.measureprobapproximate import measureprobapproximate
from bosonic_simulator.gaussian_unitary_description import GaussianUnitaryDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary_inplace


def simulateapproximately(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    unitary_descriptions: list[GaussianUnitaryDescription],
    amplitude: np.ndarray,
    wires: list[int],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64 | None = None,
):
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

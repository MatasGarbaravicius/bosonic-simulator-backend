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
    superposition_terms = copy.deepcopy(superposition_terms)
    for unitary_description in unitary_descriptions:
        for _, gaussian_state_description in superposition_terms:
            applyunitary_inplace(gaussian_state_description, unitary_description)

    return measureprobexact(superposition_terms, amplitude, wires)

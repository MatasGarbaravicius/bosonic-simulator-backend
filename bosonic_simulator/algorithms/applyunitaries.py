import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.gaussian_unitary_description import GaussianUnitaryDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary


def applyunitaries(
    gaussian_state_description: GaussianStateDescription,
    gaussian_unitary_descriptions: list[GaussianUnitaryDescription],
) -> GaussianStateDescription:
    gsd = copy.deepcopy(gaussian_state_description)
    for unitary_description in gaussian_unitary_descriptions:
        gsd = applyunitary(gsd, unitary_description)

    return gsd


# import bosonic_simulator.gaussian_unitary_description as gud
# import numpy as np
# from bosonic_simulator.algorithms.plotprobexact import plotprobexact

# gsd = GaussianStateDescription.vacuum_state(1)
# gsd = applyunitaries(
#     gsd,
#     [
#         gud.DisplacementDescription(np.array([1 + 1j])),
#         # gud.DisplacementDescription(np.array([-1 + -1j])),
#     ],
# )
# plotprobexact(
#     superposition_terms=[(np.complex128(1.0), gsd)], mode_index=0, resolution=20
# )

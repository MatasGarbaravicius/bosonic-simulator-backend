import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.gaussian_unitary_description import GaussianUnitaryDescription
from bosonic_simulator.algorithms.applyunitary import applyunitary_inplace


def applyunitaries(
    gaussian_state_description: GaussianStateDescription,
    gaussian_unitary_descriptions: list[GaussianUnitaryDescription],
) -> GaussianStateDescription:
    r"""
    Applies a sequence of Gaussian unitary operations to a Gaussian state.

    This function is not described in the reference paper. It sequentially applies a
    list of Gaussian unitaries to a state, returning a new state description representing
    the evolved state:

    $$
    |\psi_{\text{final}}\rangle = U_T \dots U_1 |\psi_{\text{initial}}\rangle
    $$

    where $U_1$ is the first element of the list and $U_T$ is the last.

    **Runtime:**

    * **General case (with squeezing):** $O(T n^3)$, where $T$ is the number of unitaries and $n$ is the number of modes.
    * **Without squeezing:** $O(n^2 + T n)$. The complexity is dominated by the single initial deep copy of the state ($O(n^2)$), followed by efficient $O(n)$ in-place updates for displacement, phase shift, and beam splitter operations.

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The description of the initial state.
    gaussian_unitary_descriptions : list[GaussianUnitaryDescription]
        The ordered list of Gaussian unitaries to apply.

    Returns
    -------
    GaussianStateDescription
        The description of the final state after applying all unitaries.
    """
    gsd = copy.deepcopy(gaussian_state_description)
    for unitary_description in gaussian_unitary_descriptions:
        applyunitary_inplace(gsd, unitary_description)

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

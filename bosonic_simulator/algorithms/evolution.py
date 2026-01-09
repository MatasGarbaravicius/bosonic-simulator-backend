import copy

import numpy as np

import bosonic_simulator.utils.coherent_state_utils as coherent_state_utils
import bosonic_simulator.types.gaussian_unitary_description as gu_description
from bosonic_simulator.algorithms.overlaps import overlaptriple
from bosonic_simulator.types.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.types.gaussian_unitary_description import (
    GaussianUnitaryDescription,
)


def squeezing(
    gaussian_state_description: GaussianStateDescription,
    squeezing_parameter: np.float64,
    mode_index: int,
) -> GaussianStateDescription:
    r"""
    Returns a new description of a Gaussian state after applying a single-mode squeezing operation.

    This implements the `squeezing` algorithm described in Section 3.3 of the
    reference paper. Given descriptions of a state $\Delta$ and a squeezing operation
    $S_j(z)$, it returns the description $\Delta'$ of the evolved state:


    $$
    |\psi(\Delta')\rangle = S_j(z) |\psi(\Delta)\rangle
    $$

    The algorithm runs in time $O(n^3)$, where $n$ is the number of modes.

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The description $\Delta$ of the initial Gaussian state.
    squeezing_parameter : np.float64
        The squeezing parameter $z \in (0, +\infty)$.
    mode_index : int
        The index of the mode $j$ to be squeezed.
        Note that this implementation uses 0-based indexing.

    Returns
    -------
    GaussianStateDescription
        The description $\Delta'$ of the squeezed state.
    """
    # precompute quantities and introduce shorter notation

    gamma = gaussian_state_description.covariance_matrix
    alpha = gaussian_state_description.amplitude
    r = gaussian_state_description.overlap
    n = gaussian_state_description.number_of_modes
    z = squeezing_parameter
    j = mode_index
    s_block = np.array([[np.exp(-z), 0], [0, np.exp(z)]])
    identity_n = np.eye(2 * n, dtype=gamma.dtype)

    # compute gamma_prime

    gamma_prime = copy.deepcopy(gamma)
    block_slice = slice(2 * j, 2 * j + 2)
    # multiply by S from the left
    gamma_prime[block_slice, :] = s_block @ gamma_prime[block_slice, :]
    # multiply by S^T from the right
    gamma_prime[:, block_slice] = gamma_prime[:, block_slice] @ s_block.T

    # compute gamma_prime_prime

    gamma_prime_prime = identity_n
    gamma_prime_prime[block_slice, block_slice] = s_block @ s_block.T

    # compute alpha_prime

    alpha_prime = copy.deepcopy(alpha)
    alpha_prime[j] = alpha[j] * np.cosh(z) - np.conj(alpha[j]) * np.sinh(z)

    # compute r_prime

    dhat_alpha_prime = coherent_state_utils.displacement_vector(alpha_prime)
    d1 = d2 = d3 = dhat_alpha_prime
    gamma1, gamma2, gamma3 = gamma_prime_prime, identity_n, gamma_prime
    u = np.conj(r)
    v = 1 / np.sqrt(np.cosh(z))
    zero_vector = np.zeros(n, dtype=np.complex128)
    r_prime = overlaptriple(gamma1, gamma2, gamma3, d1, d2, d3, u, v, zero_vector)

    return GaussianStateDescription(gamma_prime, alpha_prime, r_prime)


def applyunitary_inplace(
    gaussian_state_description: GaussianStateDescription,
    gaussian_unitary_description: gu_description.GaussianUnitaryDescription,
) -> None:
    r"""
    Overwrites in-place the description of a Gaussian state to reflect the application of a Gaussian unitary.

    This implements the `applyunitary` algorithm described in Section 3.3 of the
    reference paper. It updates the state description $\Delta$ to correspond to
    the evolved state $U |\psi(\Delta)\rangle$.

    The behavior depends on the type of `gaussian_unitary_description`:

    * **Squeezing:** Delegates to the `squeezing` routine.
    * **Displacement, Phase Shift, Beam Splitter:** Updates the covariance matrix
      and displacement vector directly using the symplectic transformations described
      in Table 1 of the reference paper.

    **Runtime:**

    * **Squeezing:** $O(n^3)$ due to the `squeezing` subroutine.
    * **Others:** $O(n)$.

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The description of the state to be updated in-place.
    gaussian_unitary_description : GaussianUnitaryDescription
        The description of the Gaussian unitary operation to apply.
    """
    gsd = gaussian_state_description  # alias for convenience
    match gaussian_unitary_description:
        case gu_description.SqueezingDescription(z, j):
            new_description = squeezing(gsd, z, j)
            gsd.__dict__.update(new_description.__dict__)

        case gu_description.DisplacementDescription(beta):
            alpha = gsd.amplitude
            exponent = 1j * (np.dot(alpha, np.conj(beta)).imag)
            alpha_prime = np.exp(exponent) * (alpha - beta)
            gsd.amplitude = alpha_prime

        case gu_description.PhaseShiftDescription(phi, j):
            # --- update amplitude ---
            gsd.amplitude[j] *= np.exp(-1j * phi)

            # --- update covariance matrix ---
            gamma = gsd.covariance_matrix
            s_block = np.array(
                [[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]
            )
            block_slice = slice(2 * j, 2 * j + 2)
            # multiply by S from the left
            gamma[block_slice, :] = s_block @ gamma[block_slice, :]
            # multiply by S^T from the right
            gamma[:, block_slice] = gamma[:, block_slice] @ s_block.T

        case gu_description.BeamSplitterDescription(w, j, k):
            # --- update amplitude ---
            alpha = gsd.amplitude
            alpha[[j, k]] = [
                alpha[j] * np.cos(w) - 1j * alpha[k] * np.sin(w),
                -1j * alpha[j] * np.sin(w) + alpha[k] * np.cos(w),
            ]

            # --- update covariance matrix ---
            gamma = gsd.covariance_matrix
            s_block = np.array(
                [
                    [np.cos(w), 0, 0, np.sin(w)],
                    [0, np.sin(w), np.cos(w), 0],
                    [0, np.cos(w), -np.sin(w), 0],
                    [-np.sin(w), 0, 0, np.cos(w)],
                ]
            )
            block_indices = [2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
            # multiply by S from the left
            gamma[block_indices, :] = s_block @ gamma[block_indices, :]
            # multiply by S^T from the right
            gamma[:, block_indices] = gamma[:, block_indices] @ s_block.T


def applyunitary(
    gaussian_state_description: GaussianStateDescription,
    gaussian_unitary_description: gu_description.GaussianUnitaryDescription,
) -> GaussianStateDescription:
    r"""
    Returns a new description of a Gaussian state after applying a Gaussian unitary.

    This is a wrapper around `applyunitary_inplace` that preserves the original
    state description by performing a deep copy before applying the operation.

    **Runtime:**

    * **Squeezing:** $O(n^3)$.
    * **Others:** $O(n^2)$. While the arithmetic updates for displacements, phase shifts,
      and beam splitters are $O(n)$, this function requires $O(n^2)$ time to allocate
      and copy the $2n \times 2n$ covariance matrix.

    Parameters
    ----------
    gaussian_state_description : GaussianStateDescription
        The initial state description.
    gaussian_unitary_description : GaussianUnitaryDescription
        The unitary operation to apply.

    Returns
    -------
    GaussianStateDescription
        The new description of the evolved state.
    """
    gsd_copy = copy.deepcopy(gaussian_state_description)
    applyunitary_inplace(gsd_copy, gaussian_unitary_description)
    return gsd_copy


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

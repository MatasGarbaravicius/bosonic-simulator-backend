import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
import bosonic_simulator.gaussian_unitary_description as gu_description
from bosonic_simulator.algorithms.squeezing import squeezing


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

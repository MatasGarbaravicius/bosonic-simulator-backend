import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
import bosonic_simulator.gaussian_unitary_description as gu_description
from bosonic_simulator.algorithms.squeezing import squeezing


def applyunitary_inplace(
    gaussian_state_description: GaussianStateDescription,
    gaussian_unitary_description: gu_description.GaussianUnitaryDescription,
):
    gsd = gaussian_state_description  # alias for convenience
    match gaussian_unitary_description:
        case gu_description.SqueezingDescription(z, j):
            # Squeezing is actually not implemented in-place, since an in-place
            # implementation would not improve runtime complexity.
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
):
    gsd_copy = copy.deepcopy(gaussian_state_description)
    applyunitary_inplace(gsd_copy, gaussian_unitary_description)
    return gsd_copy

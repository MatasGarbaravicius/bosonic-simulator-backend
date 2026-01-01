import numpy as np
import copy
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.overlaptriple import overlaptriple
from bosonic_simulator.coherent_state import CoherentState


def squeezing(
    gaussian_state_description: GaussianStateDescription,
    squeezing_parameter: np.float64,
    mode_index: int,
):
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

    dhat_alpha_prime = CoherentState.displacement_vector(alpha_prime)
    d1 = d2 = d3 = dhat_alpha_prime
    gamma1, gamma2, gamma3 = gamma_prime_prime, identity_n, gamma_prime
    u = np.conj(r)
    v = 1 / np.cosh(z)
    zero_vector = np.zeros(n, dtype=complex)
    r_prime = overlaptriple(gamma1, gamma2, gamma3, d1, d2, d3, u, v, zero_vector)

    return GaussianStateDescription(gamma_prime, alpha_prime, r_prime)

import numpy as np
import bosonic_simulator.coherent_state_tools as coherent_state_tools
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.overlaptriple import overlaptriple


def overlap(
    gaussian_state_description1: GaussianStateDescription,
    gaussian_state_description2: GaussianStateDescription,
) -> np.complex128:
    r"""
    Computes the inner product between two Gaussian states.

    This implements the `overlap` algorithm described in Section 3.2 of the
    reference paper. Given the extended descriptions $\Delta_1$ and $\Delta_2$ of two
    Gaussian states $|\psi(\Delta_1)\rangle$ and $|\psi(\Delta_2)\rangle$, it computes
    the scalar overlap:

    $$
    w = \langle \psi(\Delta_1), \psi(\Delta_2) \rangle
    $$

    The algorithm runs in time $O(n^3)$, where $n$ is the number of modes.

    Parameters
    ----------
    gaussian_state_description1 : GaussianStateDescription
        The extended description $\Delta_1$ of the first state.
    gaussian_state_description2 : GaussianStateDescription
        The extended description $\Delta_2$ of the second state.

    Returns
    -------
    np.complex128
        The complex inner product $w$.
    """
    # introducte shorter notation

    gamma1 = gaussian_state_description1.covariance_matrix
    gamma2 = gaussian_state_description2.covariance_matrix
    alpha1 = gaussian_state_description1.amplitude
    alpha2 = gaussian_state_description2.amplitude
    r1 = gaussian_state_description1.overlap
    r2 = gaussian_state_description2.overlap
    n = gaussian_state_description1.number_of_modes

    # precompute quantities

    gamma1_p = np.eye(2 * n, dtype=gamma1.dtype)
    gamma2_p = gamma1
    gamma3_p = gamma2
    d1_p = coherent_state_tools.displacement_vector(alpha1)
    d2_p = coherent_state_tools.displacement_vector(alpha1)
    d3_p = coherent_state_tools.displacement_vector(alpha2)
    lambda_ = alpha1 - alpha2
    u = np.exp(-1j * np.dot(alpha1, np.conj(alpha2)).imag) * np.conj(r2)
    v = r1

    return overlaptriple(gamma1_p, gamma2_p, gamma3_p, d1_p, d2_p, d3_p, u, v, lambda_)


# from bosonic_simulator.coherent_state_tools import CoherentState

# a1 = np.array([1 + 0j, 1 + 0j])
# a2 = np.array([-1 + 1j, 1 - 1j])
# print(coherent_state_tools.overlap(a1, a2))
# print(
#     overlap(
#         GaussianStateDescription.coherent_state(a1),
#         GaussianStateDescription.coherent_state(a2),
#     )
# )

# a3 = np.random.rand(6) + 1j * np.random.rand(6)
# a4 = np.random.rand(6) + 1j * np.random.rand(6)
# print(coherent_state_tools.overlap(a3, a4))
# print(
#     overlap(
#         GaussianStateDescription.coherent_state(a3),
#         GaussianStateDescription.coherent_state(a4),
#     )
# )

# a5 = np.random.randn(6) + 1j * np.random.randn(6)
# a6 = np.random.randn(6) + 1j * np.random.randn(6)
# print(coherent_state_tools.overlap(a5, a6))
# print(
#     overlap(
#         GaussianStateDescription.coherent_state(a5),
#         GaussianStateDescription.coherent_state(a6),
#     )
# )

# a7 = np.random.rand(6) + 1j * np.random.rand(6)
# a8 = np.random.rand(6) + 1j * np.random.rand(6)
# print(coherent_state_tools.overlap(a7, a8))
# print(
#     overlap(
#         GaussianStateDescription.coherent_state(a7),
#         GaussianStateDescription.coherent_state(a8),
#     )
# )

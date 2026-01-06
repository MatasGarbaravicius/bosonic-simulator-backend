import numpy as np  # package for scientific computing with Python
import bosonic_simulator.block_diagonal_symplectic_form_tools as omega
import bosonic_simulator.coherent_state_tools as coherent_state_tools
from scipy.linalg import solve  # type: ignore # Suppress incorrect VS Code warning
from numpy.typing import NDArray


def overlaptriple(
    gamma1: NDArray[np.float64],
    gamma2: NDArray[np.float64],
    gamma3: NDArray[np.float64],
    d1: NDArray[np.float64],
    d2: NDArray[np.float64],
    d3: NDArray[np.float64],
    u: np.complex128,
    v: np.complex128,
    lambda_: NDArray[np.complex128],
) -> np.complex128:
    # For numerical stability:
    # - Solve linear systems instead of computing explicit inverses
    # - Prefer Ax = b with vector RHS over matrix RHS

    class Inv:
        """
        A helper class that allows to write Inv(A) @ B instead of solve(A, B).

        This makes the code resemble the formula in the lemma more closely, while still
        internally avoiding the explicit computation of inverses.
        """

        def __init__(self, matrix):
            self.matrix = matrix

        def __matmul__(self, arg):
            return solve(self.matrix, arg)

    # --- precompute quantities and introduce shorter notation ---

    g1 = gamma1
    g2 = gamma2
    g3 = gamma3

    # Naming convention: gX_p_iomega / gX_m_iomega means gammaX ± i * omega.
    g1_p_iomega = omega.add_multiple_inplace_to(g1.copy(), np.complex128(1j))
    g1_m_iomega = omega.add_multiple_inplace_to(g1.copy(), np.complex128(-1j))

    g3_p_iomega = omega.add_multiple_inplace_to(g3.copy(), np.complex128(1j))
    g3_m_iomega = omega.add_multiple_inplace_to(g3.copy(), np.complex128(-1j))

    g2_p_g3 = g2 + g3
    g4 = g3 - g3_p_iomega @ (Inv(g2_p_g3) @ g3_m_iomega)
    g1_p_g4 = g1 + g4

    omega_dhat = omega.multiply_vector(
        coherent_state_tools.displacement_vector(lambda_)
    )  # omega * dhat(lambda)

    d1_prime = d1 - d3
    d2_prime = d2 - d3

    # --- evaluate terms of the exponent ---

    # Term 1: -d1'^T W1 d1'

    t1 = -np.dot(d1_prime, Inv(g1_p_g4) @ d1_prime)

    # Term 2: -d2'^T W2 d2'

    # split the matrix-vector multiplication into two separate ones
    t2_matvec1 = Inv(g2_p_g3) @ (d2_prime)
    # compute t2_matvec2 using tmp
    tmp = Inv(g2_p_g3) @ d2_prime
    tmp = g3_p_iomega @ tmp
    tmp = Inv(g1_p_g4) @ tmp
    tmp = g3_m_iomega @ tmp
    t2_matvec2 = Inv(g2_p_g3) @ tmp
    t2 = -np.dot(d2_prime, t2_matvec1 + t2_matvec2)

    # Term 3: d1'^T W3 d2'

    t3_matvec = 2 * (Inv(g1_p_g4) @ (g3_p_iomega @ (Inv(g2_p_g3) @ d2_prime)))
    t3 = np.dot(d1_prime, t3_matvec)

    # Term 4: -dhat(lambda)^T omega^T gamma5 omega dhat(lambda)

    # split the matrix multiplication by gamma5 into two separate ones
    t4_matvec1 = (gamma1 / 4) @ omega_dhat
    t4_matvec2 = (-g1_m_iomega / 4) @ (Inv(g1_p_g4) @ (g1_p_iomega @ omega_dhat))
    t4 = -np.dot(omega_dhat, t4_matvec1 + t4_matvec2)

    # Term 5: -i dhat(lambda)^T  omega^T (W4 d1' + W5 d2' + d3)

    w4_d1_prime = d1_prime - g1_m_iomega @ (Inv(g1_p_g4) @ d1_prime)
    # compute w5_d2_prime using tmp
    tmp = Inv(g2_p_g3) @ d2_prime
    tmp = g3_p_iomega @ tmp
    tmp = Inv(g1_p_g4) @ tmp
    w5_d2_prime = g1_m_iomega @ tmp
    t5 = -1j * np.dot(omega_dhat, w4_d1_prime + w5_d2_prime + d3)

    # --- evaluate the expression in the lemma in log-space (for numerical stability) ---

    exponent = t1 + t2 + t3 + t4 + t5
    (_, logabsdet1) = np.linalg.slogdet((g2_p_g3) / 2)  # log(|det(gamma2 + gamma3)|)
    (_, logabsdet2) = np.linalg.slogdet((g1_p_g4) / 2)  # log(|det(gamma1 + gamma4)|)
    log_real = exponent.real - 0.5 * (logabsdet1 + logabsdet2)
    log_imag = exponent.imag - 0.5 * (logabsdet1 + logabsdet2)

    # The result is the expression in the lemma multiplied by u^{-1} * v^{-1}.
    lemma_expression = np.exp(log_real + 1j * log_imag)
    return lemma_expression / u / v


# print(BlockDiagonalSymplecticForm.add_multiple_inplace_to(np.identity(2), 5))
# identity4 = np.identity(4, dtype=complex)
# randv4 = np.random.rand(4)
# print(
#     overlaptriple(
#         4 * identity4,
#         3 * identity4,
#         2 * identity4,
#         randv4,
#         2 * randv4,
#         3 * randv4,
#         np.complex128(2j),
#         np.complex128(1j + 1),
#         np.complex128(1j + 1),
#     )
# )

# ill_cond_mat0 = np.array([[1, 1 + 1e-16], [1, 1]])
# solve(ill_cond_mat0, np.array([1, 1]))

# ill_cond_mat1 = np.array([[1, 1 + 3e-16], [1, 1]])
# solve(ill_cond_mat1, np.array([1, 1]))

# ill_cond_mat2 = np.array([[1, 1 + 5e-16], [1, 1]])
# solve(ill_cond_mat2, np.array([1, 1]))

# ill_cond_mat3 = np.array([[1, 1 + 6e-16], [1, 1]])
# solve(ill_cond_mat3, np.array([1, 1]))

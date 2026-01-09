import numpy as np  # package for scientific computing with Python
from numpy.typing import NDArray


def multiply_vector(
    vector: NDArray[np.complex128] | NDArray[np.float64],
) -> NDArray[np.complex128] | NDArray[np.float64]:
    r"""
    Computes the product of the symplectic form matrix $\Omega$ with a vector.

    This calculates $\Omega v$ for a given vector $v$.
    The symplectic form $\Omega$ is described in Section 2 of the reference paper
    and is given by:

    $$
    \Omega = \bigoplus_{j=1}^{n} \begin{pmatrix} 0 & 1 \\\ -1 & 0 \end{pmatrix}
    $$

    Parameters
    ----------
    vector : NDArray[np.complex128] | NDArray[np.float64]
        The input vector $v$ of size $2n$.

    Returns
    -------
    NDArray[np.complex128] | NDArray[np.float64]
        The resulting vector $\Omega v$.
    """
    result = np.empty_like(vector)
    # start::step means start at index 'start' and pick elements every 'step' positions
    result[0::2] = vector[1::2]
    result[1::2] = -vector[0::2]
    return result


def add_multiple_inplace_to(
    matrix: NDArray[np.complex128] | NDArray[np.float64],
    scalar: np.complex128 | np.float64,
) -> NDArray[np.complex128] | NDArray[np.float64]:
    r"""
    Adds a scalar multiple of the symplectic form $\Omega$ to a matrix in-place.

    This performs the update $M \leftarrow M + s \Omega$, where $M$ is the
    input matrix and $s$ is the scalar. The symplectic form $\Omega$
    is described in Section 2 of the reference paper.

    Parameters
    ----------
    matrix : NDArray[np.complex128] | NDArray[np.float64]
        The matrix $M$ to be updated. It must be of shape $(2n, 2n)$.
    scalar : np.complex128 | np.float64
        The scalar multiplier $s$.

    Returns
    -------
    NDArray[np.complex128] | NDArray[np.float64]
        The updated matrix $M$.
    """
    n = matrix.shape[0] // 2  # matrix.shape[0] gives the number of rows
    even_indices = 2 * np.arange(n)  # array([ 0,  2,  4,  6, ..., 2 * (n - 1)])
    # update only the block diagonal entries
    matrix[even_indices, even_indices + 1] += scalar
    matrix[even_indices + 1, even_indices] -= scalar
    return matrix


# np.random.seed(42)
# v1 = np.random.randint(6, size=6)
# print(v1)
# print(BlockDiagonalSymplecticForm.multiply_vector(v1))
# v2 = np.ones(8)
# print(v2)
# print(BlockDiagonalSymplecticForm.multiply_vector(v2))
# print(BlockDiagonalSymplecticForm.add_multiple_inplace_to(np.identity(6, dtype=int), 5))
# rand_mat = np.random.randint(11, size=(4, 4))
# print(rand_mat)
# print(BlockDiagonalSymplecticForm.add_multiple_inplace_to(rand_mat, -11))

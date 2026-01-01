import numpy as np  # package for scientific computing with Python


class BlockDiagonalSymplecticForm:

    def multiply_vector(vector: np.ndarray):
        result = np.empty_like(vector)
        # start::step means start at index 'start' and pick elements every 'step' positions
        result[0::2] = vector[1::2]
        result[1::2] = -vector[0::2]
        return result

    def add_multiple_inplace_to(matrix: np.ndarray, scalar: np.complex128):
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

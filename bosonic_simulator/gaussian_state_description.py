import numpy as np  # package for scientific computing with Python
from numpy.typing import NDArray


class GaussianStateDescription:  # see subsection "Extended description of a Gaussian state"

    def __init__(
        self,
        covariance_matrix: NDArray[np.float64],
        amplitude: NDArray[np.complex128],
        overlap: np.complex128,
    ):
        self.covariance_matrix = covariance_matrix
        self.amplitude = amplitude
        self.overlap = overlap

    def energy(self) -> np.float64:
        tr_gamma = np.trace(self.covariance_matrix)
        d_norm_squared = 2 * (np.linalg.norm(self.amplitude))
        n = self.number_of_modes
        return np.real(0.5 * tr_gamma + d_norm_squared + n)

    @classmethod
    def coherent_state(
        cls, amplitude: NDArray[np.complex128]
    ) -> "GaussianStateDescription":
        alpha = amplitude
        identity = np.eye(
            2 * alpha.shape[0], dtype=np.float64
        )  # I matrix of needed shape & type
        return cls(identity, alpha, np.complex128(1))

    @classmethod
    def vacuum_state(cls, complex_dimension: int) -> "GaussianStateDescription":
        return cls.coherent_state(np.zeros(complex_dimension, dtype=complex))

    @property
    def number_of_modes(self) -> int:
        return self.amplitude.size


# a = GaussianStateDescription(np.tile(np.eye(2), (3, 3)), np.ones(6, dtype=complex), 5)
# print(a.covariance_matrix)
# a.apply_unitary(gu_description.PhaseShiftDescription(np.radians(30), 2))
# np.set_printoptions(precision=3, suppress=True)
# print(a.covariance_matrix)
# a.apply_unitary(gu_description.PhaseShiftDescription(np.radians(-30), 2))
# print(a.covariance_matrix)

# # b = GaussianStateDescription(np.tile(np.eye(4), (2, 2)), np.ones(6, dtype=complex), 5)
# b = GaussianStateDescription(np.tile(np.eye(8), (1, 1)), np.ones(6, dtype=complex), 5)
# print(b.covariance_matrix)
# b.apply_unitary(gu_description.BeamSplitterDescription(np.radians(30), 2, 4))
# print(b.covariance_matrix)
# print(
#     b.covariance_matrix[[2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1], :][
#         :, [2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1]
#     ]
# )

# c = GaussianStateDescription(np.tile(np.eye(8), (1, 1)), np.ones(6, dtype=complex), 5)
# print(c.covariance_matrix)
# c.apply_unitary(gu_description.BeamSplitterDescription(np.radians(30), 4, 2))
# print(c.covariance_matrix)
# print(
#     c.covariance_matrix[[2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1], :][
#         :, [2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1]
#     ]
# )

# d = GaussianStateDescription(np.tile(np.eye(8), (1, 1)), np.ones(6, dtype=complex), 5)
# print(d.covariance_matrix)
# d.apply_unitary(gu_description.BeamSplitterDescription(np.radians(30), 3, 3))
# print(d.covariance_matrix)
# print(
#     d.covariance_matrix[[2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1], :][
#         :, [2 * 1, 2 * 1 + 1, 2 * 3, 2 * 3 + 1]
#     ]
# )


# def modify_gaussian(gs_desc: GaussianStateDescription):
#     gs_desc.covariance_matrix = np.ones(3)


# modify_gaussian(a)
# print(a.covariance_matrix)

# print(a.number_of_modes)


# amp1 = np.repeat([(1 + 1j) / np.sqrt(10)], 5)
# gsd1 = GaussianStateDescription.coherent_state(amp1)
# print(f"norm = {np.linalg.norm(amp1)}")
# print(gsd1.prob(amp1, np.arange(1, 6)))
# print(gsd1.covariance_matrix, gsd1.amplitude, gsd1.overlap)
# gsd1.post_measure(amp1, np.arange(1, 6))
# print("hello2")
# print(gsd1.covariance_matrix, gsd1.amplitude, gsd1.overlap)

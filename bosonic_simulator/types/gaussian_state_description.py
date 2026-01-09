import numpy as np  # package for scientific computing with Python
from numpy.typing import NDArray


class GaussianStateDescription:
    r"""
    A description of a pure Gaussian state including its global phase information.

    This class implements the "Extended description of a Gaussian state" as defined in
    Definition 3.1 of the reference paper. It characterizes a pure Gaussian state $|\psi\rangle$
    using a triple
    $$\Delta = (\Gamma, \alpha, r) \in \mathsf{Mat}_{2n\times 2n}(\mathbb{R}) \times \mathbb{C}^{n} \times \mathbb{C}, $$
    where:

    * $\Gamma$ is the covariance matrix.
    * $\alpha$ is the complex amplitude of a reference coherent state $|\alpha\rangle$
      that has the same displacement as $|\psi\rangle$.
    * $r = \langle \alpha, \psi \rangle$ is the overlap between the state and the
      reference coherent state.

    This description allows for tracking relative phases in superpositions of Gaussian states,
    which is not possible using the standard covariance matrix formalism alone.

    Parameters
    ----------
    covariance_matrix : NDArray[np.float64]
        The $2n \times 2n$ real symmetric covariance matrix $\Gamma$.
    amplitude : NDArray[np.complex128]
        The complex vector $\alpha \in \mathbb{C}^n$ representing the displacement.
    overlap : np.complex128
        The scalar overlap $r = \langle \alpha, \psi \rangle$.

    Notes
    -----
    **Indexing Convention:**
    While the reference paper uses 1-based indexing for modes (e.g., $j \in \{1, \dots, n\}$),
    this implementation uses standard Python 0-based indexing (e.g., indices `0` to `n-1`).

    **Real Displacement Vector:**
    The corresponding real displacement vector $d$ in phase space is determined by the mapping
    $d = \hat{d}(\alpha)$ described in Section 2.1. Explicitly:

    $$
    \hat{d}(\alpha) = \sqrt{2} (\text{Re}(\alpha_1), \text{Im}(\alpha_1), \dots, \text{Re}(\alpha_n), \text{Im}(\alpha_n))^T
    $$

    **Bijective Mapping:**
    The set of these descriptions, denoted $\text{Desc}_n$, maps bijectively to the set
    of $n$-mode pure Gaussian states $\text{Gauss}_n$. This means every description uniquely
    fixes a Gaussian state $|\psi(\Delta)\rangle$, and conversely, every pure Gaussian state
    admits a unique description.
    """

    def __init__(
        self,
        covariance_matrix: NDArray[np.float64],
        amplitude: NDArray[np.complex128],
        overlap: np.complex128,
    ):
        self.covariance_matrix = covariance_matrix
        self.amplitude = amplitude
        self.overlap = overlap

    @classmethod
    def coherent_state(
        cls, amplitude: NDArray[np.complex128]
    ) -> "GaussianStateDescription":
        r"""
        Creates a description for a coherent state $|\alpha\rangle$.

        According to Section 2.3, a coherent state is described by:

        * Covariance matrix $\Gamma = I$.
        * Displacement described by $\alpha$.

        For the extended description, the reference state is chosen to be the state itself,
        so the overlap is $r = \langle \alpha | \alpha \rangle = 1$.

        Parameters
        ----------
        amplitude : NDArray[np.complex128]
            The complex amplitude vector $\alpha$.

        Returns
        -------
        GaussianStateDescription
            The description $(I, \alpha, 1)$.
        """
        alpha = amplitude
        identity = np.eye(
            2 * alpha.shape[0], dtype=np.float64
        )  # I matrix of needed shape & type
        return cls(identity, alpha, np.complex128(1))

    @classmethod
    def vacuum_state(cls, complex_dimension: int) -> "GaussianStateDescription":
        r"""
        Creates a description for the vacuum state $|0\rangle$.

        The vacuum state is a coherent state with $\alpha = 0$.

        Parameters
        ----------
        complex_dimension : int
            The number of modes $n$.

        Returns
        -------
        GaussianStateDescription
            The description of the n-mode vacuum.
        """
        return cls.coherent_state(np.zeros(complex_dimension, dtype=complex))

    def energy(self) -> np.float64:
        r"""
        Computes the energy (mean photon number) of the Gaussian state.

        This implements the energy formula given in Section 2.1 of the reference paper.
        The energy is given by:

        $$
        \text{tr}(H \rho) = \frac{1}{2}\text{tr}(\Gamma) + d^T d + n
        $$

        where $d$ is the real displacement vector. In this implementation,
        $d^T d$ is calculated as $2 |\alpha|^2$ based on the mapping
        $\hat{d}(\alpha)$ described in Section 2.3.

        Returns
        -------
        np.float64
            The energy of the state.
        """
        tr_gamma = np.trace(self.covariance_matrix)
        d_norm_squared = 2 * (np.linalg.norm(self.amplitude) ** 2)
        n = self.number_of_modes
        return np.real(0.5 * tr_gamma + d_norm_squared + n)

    @property
    def number_of_modes(self) -> int:
        r"""
        Returns the number of modes $n$ of the system.

        Returns
        -------
        int
            The number of modes.
        """
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

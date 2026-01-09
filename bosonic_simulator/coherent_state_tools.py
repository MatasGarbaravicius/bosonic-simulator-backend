import numpy as np
from numpy.typing import NDArray


def displacement_vector(amplitude: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""
    Converts a complex amplitude vector to a real displacement vector.

    This implements the mapping $\alpha \mapsto \hat{d}(\alpha)$ described in
    Section 2.1 of the reference paper. For a vector of
    complex amplitudes $\alpha \in \mathbb{C}^n$, the real displacement
    vector $d \in \mathbb{R}^{2n}$ is constructed as:

    $$
    \hat{d}(\alpha) = \sqrt{2} (\text{Re}(\alpha_1), \text{Im}(\alpha_1), \dots, \text{Re}(\alpha_n), \text{Im}(\alpha_n))^T
    $$

    **Runtime:**
    $O(n)$.

    Parameters
    ----------
    amplitude : NDArray[np.complex128]
        The complex amplitude vector $\alpha$.

    Returns
    -------
    NDArray[np.float64]
        The real displacement vector $\hat{d}(\alpha)$.
    """
    result = np.empty(2 * amplitude.size, dtype=np.float64)
    result[0::2] = amplitude.real  # fill even indices with real parts
    result[1::2] = amplitude.imag  # fill odd indices with imaginary parts
    return np.sqrt(2) * result


def overlap(
    amplitude1: NDArray[np.complex128], amplitude2: NDArray[np.complex128]
) -> np.complex128:
    r"""
    Computes the overlap between two coherent states.

    This implements the formula for the inner product $\langle \alpha_1 | \alpha_2 \rangle$
    between two coherent states described in Section 2.5 of the reference paper.
    The overlap is given by:

    $$
    \langle \alpha_1 | \alpha_2 \rangle = \exp\left( -\frac{1}{2}|\alpha_1|^2 - \frac{1}{2}|\alpha_2|^2 + \overline{\alpha_1}^T \alpha_2 \right)
    $$

    **Runtime:**
    $O(n)$, where $n$ is the length of `amplitude1`.

    Parameters
    ----------
    amplitude1 : NDArray[np.complex128]
        The complex amplitude $\alpha_1$ of the first state.
    amplitude2 : NDArray[np.complex128]
        The complex amplitude $\alpha_2$ of the second state.

    Returns
    -------
    np.complex128
        The complex overlap $\langle \alpha_1 | \alpha_2 \rangle$.
    """
    a1 = amplitude1
    a2 = amplitude2
    return np.exp(
        -0.5 * (np.linalg.norm(a1) ** 2)
        - 0.5 * (np.linalg.norm(a2) ** 2)
        + np.vdot(a1, a2)
    )


def displacement_vec_to_amplitude(
    displacement_vector: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""
    Converts a real displacement vector back to a complex amplitude vector.

    This performs the inverse of the mapping $\hat{d}(\alpha)$ described in
    Section 2.1 of the reference paper. The mapping relates the complex
    amplitudes $\alpha$ to the real displacement vector $d$ via:

    $$
    \hat{d}(\alpha) = \sqrt{2} (\text{Re}(\alpha_1), \text{Im}(\alpha_1), \dots, \text{Re}(\alpha_n), \text{Im}(\alpha_n))^T
    $$

    **Runtime:**
    $O(n)$.

    Parameters
    ----------
    displacement_vector : NDArray[np.float64]
        The real displacement vector $d$.


    Returns
    -------
    NDArray[np.complex128]
        The complex amplitude vector $\alpha$.
    """
    # start::step means start at index 'start' and pick elements every 'step' positions
    return (displacement_vector[0::2] + 1j * displacement_vector[1::2]) / np.sqrt(2)


# alpha = np.array([1 + 2j, 3 + 4j])
# print(CoherentState.displacement_vector(alpha))

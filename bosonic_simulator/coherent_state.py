import numpy as np


class CoherentState:

    def displacement_vector(amplitude: np.ndarray):
        result = np.empty(2 * amplitude.size, dtype=np.float64)
        result[0::2] = amplitude.real  # fill even indices with real parts
        result[1::2] = amplitude.imag  # fill odd indices with imaginary parts
        return np.sqrt(2) * result

    def overlap(amplitude1: np.ndarray, amplitude2: np.ndarray):
        a1 = amplitude1
        a2 = amplitude2
        return np.exp(
            -0.5 * (np.linalg.norm(a1) ** 2)
            - 0.5 * (np.linalg.norm(a2) ** 2)
            + np.vdot(a1, a2)
        )

    def displacement_vec_to_amplitude(displacement_vector: np.ndarray):
        # start::step means start at index 'start' and pick elements every 'step' positions
        return (displacement_vector[0::2] + 1j * displacement_vector[1::2]) / np.sqrt(2)


# alpha = np.array([1 + 2j, 3 + 4j])
# print(CoherentState.displacement_vector(alpha))

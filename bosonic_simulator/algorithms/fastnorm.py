import numpy as np
from bosonic_simulator.gaussian_state_description import GaussianStateDescription
from bosonic_simulator.algorithms.overlap import overlap


def fastnorm(
    superposition_terms: list[tuple[np.complex128, GaussianStateDescription]],
    multiplicative_error: np.float64,
    max_failure_probability: np.float64,
    energy_upper_bound: np.float64,
) -> np.float64:
    # introduce shorter notation

    n = superposition_terms[0][1].number_of_modes
    eps = multiplicative_error
    p_f = max_failure_probability
    capital_e = energy_upper_bound

    # execute the algorithm

    capital_r = np.sqrt(capital_e / eps)
    capital_l = np.ceil(4 * np.pi * capital_e / (p_f * (eps**3))).astype(int)
    sample_sum = np.float64(0)
    for _ in range(capital_l):
        # 1. sample alpha according to the Lebesgue measure on B_R(0)

        # 1.1 sample a random direction uniformly on the unit sphere in C^n by
        # normalizing a standard normal vector (the density is rotationally invariant)

        v = np.random.standard_normal(n) + 1j * np.random.standard_normal(n)
        v = v / np.linalg.norm(v)

        # 1.2 sample radius r so that the cumulative distribution fct is F(r) = (r/R)^{2n}

        r = capital_r * np.power(np.random.uniform(0, 1), 1 / (2 * n))

        alpha = r * v

        # 2. update sample_sum

        delta = GaussianStateDescription.coherent_state(alpha)
        sum_cjoj = np.sum([c * overlap(delta, psi) for (c, psi) in superposition_terms])
        sample_sum += (capital_r**2) * (np.abs(sum_cjoj) ** 2)

    return np.sqrt(sample_sum / capital_l)
